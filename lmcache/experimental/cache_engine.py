# Copyright 2024-2025 LMCache Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import multiprocessing
import time
import threading
from typing import Dict, Generator, List, Optional, Union, Callable

import torch

from lmcache.config import LMCacheEngineMetadata
from lmcache.experimental.config import LMCacheEngineConfig
from lmcache.experimental.distributed_server import (
    DistributedServerInterface, NaiveDistributedServer)
from lmcache.experimental.gpu_connector import (
    GPUConnectorInterface, VLLMPagedMemLayerwiseGPUConnector)
from lmcache.experimental.lookup_server import (LookupServerInterface,
                                                RedisLookupServer)
from lmcache.experimental.memory_management import (  # noqa: E501
    AdHocMemoryAllocator, MemoryAllocatorInterface, MemoryFormat, MemoryObj,
    MixedMemoryAllocator)
from lmcache.experimental.storage_backend.storage_manager import (
    DistributedStorageManager, StorageManager)
from lmcache.experimental.token_database import (ChunkedTokenDatabase,
                                                 TokenDatabase)
from lmcache.logging import init_logger
from lmcache.observability import LMCacheStatsLogger, LMCStatsMonitor
from lmcache.usage_context import InitializeUsageContext
from lmcache.utils import CacheEngineKey, LayerCacheEngineKey, _lmcache_nvtx_annotate, NVTXContext
from lmcache.experimental.storage_backend.connector.nixl_connector_v2 import NixlObserverInterface

## TODO: remove this after nvtx is fixed
from nvtx import annotate  # type: ignore
from lmcache.utils import _get_color_for_nvtx

logger = init_logger(__name__)


class CacheEngineEndSignal:
    pass


class LMCacheEngine:
    """The main class for the cache engine. 

    When storing the KV caches into the cache engine, it takes GPU KV
    caches from the serving engine and convert them into MemoryObjs that
    resides in the CPU. The MemoryObjs are then being stored into the 
    StorageBackends in an asynchronous manner.

    When retrieving the KV caches from the cache engine, it fetches the
    MemoryObjs from the StorageBackends and convert them into GPU KV caches
    by GPUConnectors specialized for the serving engine.

    It also supports prefetching the KV caches from the StorageBackends. 
    It relies on the StorageBackends to manage the requests of prefetching
    and real retrieval and avoid the conflicts.
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
        memory_allocator: MemoryAllocatorInterface,
        token_database: TokenDatabase,
        gpu_connector: GPUConnectorInterface,
        layerwise: bool = False,
    ):
        logger.info(f"Creating LMCacheEngine with config: {config}")
        self.config = config
        self.metadata = metadata
        self.memory_allocator = memory_allocator
        self.token_database = token_database
        self.gpu_connector = gpu_connector

        self.enable_p2p = config.enable_p2p

        # NOTE: Unix systems use fork by default
        multiprocessing.set_start_method('spawn', force=True)

        self.lookup_server: Optional[LookupServerInterface] = None
        if self.enable_p2p:
            self.lookup_server = RedisLookupServer(config)

        # avoid circular import
        from lmcache.experimental.cache_controller import LMCacheWorker
        self.lmcache_worker: Optional[LMCacheWorker] = None
        if self.config.enable_controller:
            self.lmcache_worker = LMCacheWorker(config, metadata, self)

        self.use_distributed_storage_manager = False
        if config.enable_nixl:
            self.use_distributed_storage_manager = True
            self.storage_manager = DistributedStorageManager(
                config, metadata, self.memory_allocator)
        else:
            self.storage_manager = StorageManager(
                config, metadata, self.memory_allocator, self.lmcache_worker,
                self.lookup_server, layerwise)  # type: ignore[assignment]

        if self.enable_p2p:
            self.distributed_loop = asyncio.get_event_loop()
            assert self.lookup_server is not None
            assert isinstance(self.storage_manager, StorageManager)
            self.distributed_server: DistributedServerInterface = \
                NaiveDistributedServer(self.storage_manager,
                                       self.lookup_server,
                                       self.distributed_loop,
                                       config)

        InitializeUsageContext(config.to_original_config(), metadata)
        self.stats_monitor = LMCStatsMonitor.GetOrCreate()

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def store_distributed(self,
                          tokens: torch.Tensor,
                          mask: Optional[torch.Tensor] = None,
                          **kwargs) -> None:
        """Store the tokens and mask into the cache engine.
        
        This function is only for distributed storage manager.

        This function will be refactored in the future.
        """
        st = time.perf_counter()
        if mask is not None:
            num_store_tokens = torch.sum(mask)
        else:
            num_store_tokens = len(tokens)
        monitor_req_id = self.stats_monitor.on_store_request(num_store_tokens)

        # Register the put request
        keys = []
        metadatas = []
        steds = []
        for start, end, key in self.token_database.process_tokens(
                tokens, mask):
            assert isinstance(key, CacheEngineKey)
            # Allocate the memory object
            num_tokens = end - start
            kv_shape = self.gpu_connector.get_shape(num_tokens)
            kv_dtype = self.metadata.kv_dtype
            memobj_meta = self.storage_manager.dry_allocate(kv_shape, kv_dtype)
            assert memobj_meta is not None
            keys.append(key)
            metadatas.append(memobj_meta)
            steds.append((start, end))

        self.storage_manager.prepare_put(keys, metadatas)

        offload_time = 0.
        put_time = 0.
        tot_kv_size = 0
        # Offload the KV cache and write to remote
        for key, memobj_meta, (start, end) in zip(keys,
                                                  metadatas,
                                                  steds,
                                                  strict=False):
            assert memobj_meta.dtype is not None
            kv_shape = memobj_meta.shape
            kv_dtype = memobj_meta.dtype

            # Allocate for a zero-copy buffer, trigger send if needed
            t = time.perf_counter()
            memory_obj = self.storage_manager.allocate(kv_shape, kv_dtype)
            put_time += time.perf_counter() - t
            if memory_obj is None:
                logger.warning("Failed to allocate memory for the KV cache.\n"
                               "The KV cache will not be stored.")
                break

            # Copy the KV cache to the zero-copy buffer
            t = time.perf_counter()
            self.gpu_connector.from_gpu(memory_obj, start, end, **kwargs)
            offload_time += time.perf_counter() - t

            tot_kv_size += memory_obj.get_size()

        # Flush
        t = time.perf_counter()
        self.storage_manager.commit_put()
        put_time += time.perf_counter() - t
        ed = time.perf_counter()

        logger.info(
            "Store %d tokens takes: %.4f ms, throughput: %.4f GB/s; "
            "offload_time: %.4f ms, put_time: %.4f ms", num_store_tokens,
            (ed - st) * 1000, tot_kv_size / (ed - st) / 1024**3,
            offload_time * 1000, put_time * 1000)

        self.stats_monitor.on_store_finished(monitor_req_id)

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def store(self,
              tokens: torch.Tensor,
              mask: Optional[torch.Tensor] = None,
              **kwargs) -> None:
        """Store the tokens and mask into the cache engine.

        :param torch.Tensor tokens: The tokens of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should 
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched, 
            and the Falses will ALWAYS be at the PREFIX of the tensor.

        :param **kwargs: The additional arguments for the storage backend which
            will be passed into the gpu_connector.
            Should include KV cache specific information (e.g., paged KV buffer
            and the page tables). 

        :raises: ValueError if the number of Falses in the mask is not a 
            multiple of the chunk size.
        """
        # FIXME(ApostaC): A HACK for distributed storage manager
        if self.use_distributed_storage_manager:
            self.store_distributed(tokens, mask, **kwargs)
            return

        if mask is not None:
            num_stored_tokens = torch.sum(mask).item()
        else:
            num_stored_tokens = len(tokens)
        monitor_req_id = self.stats_monitor.on_store_request(num_stored_tokens)

        for start, end, key in self.token_database.process_tokens(
                tokens, mask):
            assert isinstance(key, CacheEngineKey)
            if self.storage_manager.contains(key):
                continue
            # Allocate the memory object
            num_tokens = end - start
            kv_shape = self.gpu_connector.get_shape(num_tokens)
            kv_dtype = self.metadata.kv_dtype
            memory_obj = self.storage_manager.allocate(kv_shape, kv_dtype)
            if memory_obj is None:
                logger.warning("Failed to allocate memory for the KV cache.\n"
                               "The KV cache will not be stored.")
                break

            self.gpu_connector.from_gpu(memory_obj, start, end, **kwargs)
            self.storage_manager.put(key, memory_obj)

            # Update lookup server
            if self.lookup_server is not None:
                self.lookup_server.insert(key)

        self.stats_monitor.on_store_finished(monitor_req_id)

        logger.debug(f"Stored {num_stored_tokens} "
                     f"out of total {len(tokens)} tokens")

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def retrieve(self,
                 tokens: torch.Tensor,
                 mask: Optional[torch.Tensor] = None,
                 **kwargs) -> torch.Tensor:
        """Retrieve the KV caches from the cache engine. And put the retrieved
        KV cache to the serving engine via the GPU connector.

        :param torch.Tensor tokens: The tokens of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should 
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched, 
            and the Falses will ALWAYS be at the PREFIX of the tensor.

        :param **kwargs: The additional arguments for the storage backend which
            will be passed into the gpu_connector.
            Should include KV cache specific information (e.g., paged KV buffer
            and the page tables). 

        :return: the boolean mask indicating which tokens are retrieved. The 
            length of the mask should be the same as the tokens. On CPU.

        :raises: ValueError if the number of Falses in the mask is not a 
            multiple of the chunk size.
        """
        if mask is not None:
            num_required_tokens = torch.sum(mask).item()
        else:
            num_required_tokens = len(tokens)
        monitor_req_id = self.stats_monitor.on_retrieve_request(
            num_required_tokens)

        ret_mask = torch.zeros_like(tokens, dtype=torch.bool, device="cpu")
        for start, end, key in self.token_database.process_tokens(
                tokens, mask):

            assert isinstance(key, CacheEngineKey)

            # Get the memory object from the storage backend
            memory_obj = self.storage_manager.get(key)

            if memory_obj is None:
                if self.enable_p2p:
                    future_memory_obj = asyncio.run_coroutine_threadsafe(
                        self.distributed_server.issue_get(key),
                        self.distributed_loop)
                    memory_obj = future_memory_obj.result()
                if memory_obj is None:
                    break

            ret_mask[start:end] = True

            # NOTE(Jiayi): memory_obj doesn't have to be a pinned
            # cpu tensor for the sake of performance.
            # For example, disk->gpu is faster than disk->cpu->gpu.
            # RDMA is another example.
            self.gpu_connector.to_gpu(memory_obj, start, end, **kwargs)
            memory_obj.ref_count_down()

            # NOTE (ApostaC): This is only for the current implementation:
            # When the object is retrieved back to vLLM, the storage backend
            # will immediately remove the object from itself
            if isinstance(self.storage_manager, DistributedStorageManager):
                self.storage_manager.remove(key)

        retrieved_tokens = torch.sum(ret_mask)
        self.stats_monitor.on_retrieve_finished(monitor_req_id,
                                                retrieved_tokens)
        logger.debug(f"Retrieved {retrieved_tokens} "
                     f"out of {num_required_tokens} "
                     f"out of total {len(tokens)} tokens")
        return ret_mask

    def prefetch(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        """Launch the prefetching process in the storage manager to load the 
        KV to the local CPU memory
        """
        for start, end, key in self.token_database.process_tokens(
                tokens, mask):
            assert isinstance(key, CacheEngineKey)
            self.storage_manager.prefetch(key)

    # TODO(Jiayi): Currently, search_range is only used for testing.
    def lookup(
        self,
        tokens: Union[torch.Tensor, List[int]],
        search_range: Optional[List[str]] = None,
        pin: bool = False,
    ) -> int:
        """
        Checks the existence of KV cache of the tokens from the cache engine.

        :param tokens: the input tokens, with shape [seq_len]
        
        :param Optional[List[str]] search_range: The range of storage backends
        to search in. Should be a subset of 
        ["LocalCPUBackend", "LocalDiskBackend"] for now.
        If None, search in all backends.
        
        :param bool pin: If True, pin the KV cache in the storage.

        :return: An int indicating how many prefix tokens are cached.
        """
        end = 0
        search_local = True  # we always lookup local storage_manager first
        # secondary lookup on p2p (via lookup_server) if enabled
        search_p2p = (self.enable_p2p
                      and (search_range is None or "p2p" in search_range))

        for start, end, key in self.token_database.process_tokens(tokens):
            assert isinstance(key, CacheEngineKey)
            if search_local:
                if self.storage_manager.contains(key, search_range, pin):
                    # found in storage manager, no need to search p2p
                    continue
                else:
                    # key not found in storage_manager
                    # search only p2p from now on
                    search_local = False
            if search_p2p:
                assert self.lookup_server is not None
                if self.lookup_server.lookup(key):
                    # found in p2p
                    # continue loop to ensure a maximal prefix match
                    continue
            # not found in both storage_manager and p2p,
            # return start, which equals last iteration's end
            return start

        # all tokens where found, return the maximal end
        return end

    def clear(
        self,
        tokens: Optional[Union[torch.Tensor, List[int]]] = None,
        locations: Optional[List[str]] = None,
    ) -> int:
        assert isinstance(self.storage_manager, StorageManager)
        # Clear all caches if tokens is None
        if tokens is None or len(tokens) == 0:
            num_cleared = self.storage_manager.clear(locations)
            return num_cleared

        num_removed = 0
        # Only remove the caches for the given tokens
        for start, end, key in self.token_database.process_tokens(tokens):
            assert isinstance(key, CacheEngineKey)
            removed = self.storage_manager.remove(key, locations)
            num_removed += removed
        return num_removed

    def close(self) -> None:
        """Close the cache engine and free all the resources"""

        if self.enable_p2p:
            self.distributed_server.close()

        if self.lmcache_worker is not None:
            self.lmcache_worker.close()

        self.storage_manager.close()
        logger.info("LMCacheEngine closed.")


# TODO(Jiayi): Using a separate class here.
# Should use the same class once the code is stable.
class LayerwiseLMCacheEngine(LMCacheEngine):
    """A specialized LMCacheEngine for layerwise cache engine.
    
    This class is used to store the layerwise cache engine. It is a
    subclass of LMCacheEngine and inherits all the methods and attributes
    from it. However, it retrieves the KV cache in a layerwise manner 
    instead of chunkwise manner.
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
        memory_allocator: MemoryAllocatorInterface,
        token_database: TokenDatabase,
        layerwise_gpu_connector: GPUConnectorInterface,
        layerwise: bool = True,
        batch_size: int = 32,
    ):
        super().__init__(config, metadata, memory_allocator, token_database,
                         layerwise_gpu_connector, layerwise)
        assert isinstance(self.gpu_connector,
                          VLLMPagedMemLayerwiseGPUConnector)

        self.num_layers = metadata.kv_shape[0]
        self.batch_size = batch_size

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def store_layer(self,
                    tokens: torch.Tensor,
                    mask: Optional[torch.Tensor] = None,
                    **kwargs) -> Generator[None, None, None]:
        """
        Store the KV cache in a layerwise manner with bandwidth optimization.
        
        This implementation processes one layer at a time in the following order:
        1. Process all token chunks to get metadata for the entire layer
        2. Register metadata with storage manager
        3. Allocate memory and transfer data in batches
        4. Store in backend
        5. Move to next layer
        
        This approach ensures proper metadata registration before allocation
        while maintaining bandwidth optimization.
        
        :param torch.Tensor tokens: The tokens of the corresponding KV caches.
        
        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched.
        
        :param **kwargs: The additional arguments for the storage backend which
            will be passed into the gpu_connector.
        
        return: A generator that yields None. In the first iteration, the 
            generator allocates the memory objects for all layers and moves 
            the KV cache of the first layer from GPU to CPU. In the next 
            iterations, it moves the KV cache of layer i from GPU to the memory 
            objects (on CPU) and puts the memory objects of layer i-1 to the 
            storage backends. In the last iteration, it puts the memory objects 
            of the last layer to the storage backends.
        """
        st = time.perf_counter()
        if mask is not None:
            num_tokens = torch.sum(mask).item()
        else:
            num_tokens = len(tokens)
        monitor_req_id = self.stats_monitor.on_store_request(num_tokens)

        # Process tokens to get starts, ends, and base keys for each layer
        token_chunks = list(self.token_database.process_tokens(tokens, mask))
        if not token_chunks:
            # If no tokens to process, yield for each layer and return
            for _ in range(self.num_layers):
                yield
            yield
            return

        # Track metrics per layer
        layerwise_times = []
        total_offload_time = 0.0
        total_put_time = 0.0
        total_prepare_time = 0.0
        total_batch_put_time = 0.0
        total_kv_size = 0

        # Process one layer at a time
        for layer_id in range(self.num_layers):
            layer_st = time.perf_counter()
            # First pass: collect metadata for all chunks in this layer
            chunk_metadata = []
            chunk_keys = []
            chunk_ranges = []  # Store (start, end) pairs

            for start, end, base_key in token_chunks:
                assert isinstance(base_key, CacheEngineKey)

                # Create layer-specific key
                keys_multi_layer = base_key.split_layers(
                    self.num_layers)[layer_id]

                # Skip if already cached
                if self.storage_manager.contains(keys_multi_layer):
                    continue

                # Prepare metadata for this chunk
                num_chunk_tokens = end - start
                kv_shape_single_layer = self.gpu_connector.get_shape(
                    num_chunk_tokens)
                memobj_meta = self.storage_manager.dry_allocate(
                    kv_shape_single_layer,
                    self.metadata.kv_dtype,
                    fmt=MemoryFormat.KV_T2D)

                if memobj_meta is None:
                    logger.warning(
                        f"Failed to prepare metadata for layer {layer_id}, "
                        f"chunk {start}:{end}")
                    continue

                chunk_metadata.append(memobj_meta)
                chunk_keys.append(keys_multi_layer)
                chunk_ranges.append((start, end))

            if not chunk_ranges:
                # No chunks to process in this layer
                yield
                continue

            # Process chunks in batches
            for batch_start in range(0, len(chunk_ranges), self.batch_size):
                batch_end = min(batch_start + self.batch_size,
                                len(chunk_ranges))
                logger.debug(f"Processing batch {batch_start}:{batch_end}")

                # Get the current batch
                batch_ranges = chunk_ranges[batch_start:batch_end]
                batch_keys = chunk_keys[batch_start:batch_end]
                batch_metadata = chunk_metadata[batch_start:batch_end]

                # Register metadata for this batch
                t = time.perf_counter()
                self.storage_manager.prepare_put(batch_keys, batch_metadata)
                total_prepare_time += time.perf_counter() - t
                total_put_time += time.perf_counter() - t

                # Allocate memory for the batch
                batch_starts = []
                batch_ends = []
                batch_memory_objs = []
                batch_valid_keys = []

                for (start, end), keys_multi_layer, meta in zip(
                        batch_ranges, batch_keys, batch_metadata):
                    t = time.perf_counter()
                    mem_obj = self.storage_manager.allocate(meta.shape,
                                                            meta.dtype,
                                                            fmt=meta.fmt)
                    total_put_time += time.perf_counter() - t

                    if mem_obj is None:
                        logger.warning(
                            f"Failed to allocate memory for layer {layer_id}, "
                            f"chunk {start}:{end}")
                        continue

                    batch_starts.append(start)
                    batch_ends.append(end)
                    batch_memory_objs.append(mem_obj)
                    batch_valid_keys.append(keys_multi_layer)
                    total_kv_size += mem_obj.get_size()

                    # Update lookup server for this chunk
                    if self.lookup_server is not None:
                        self.lookup_server.insert(keys_multi_layer)

                if batch_memory_objs:
                    # Transfer data for this batch
                    assert isinstance(self.gpu_connector,
                                      VLLMPagedMemLayerwiseGPUConnector)

                    # Create batch-specific kwargs by slicing relevant data
                    batch_kwargs = kwargs.copy()
                    if "slot_mapping" in kwargs:
                        slot_mapping = kwargs["slot_mapping"]
                        min_start = min(batch_starts)
                        max_end = max(batch_ends)
                        batch_kwargs["slot_mapping"] = slot_mapping[
                            min_start:max_end]

                        # Adjust starts and ends to be relative to the sliced slot_mapping
                        relative_starts = [s - min_start for s in batch_starts]
                        relative_ends = [e - min_start for e in batch_ends]
                    else:
                        relative_starts = batch_starts
                        relative_ends = batch_ends

                    # Create a single-layer generator for this batch
                    t = time.perf_counter()
                    mem_obj_generator = self.gpu_connector.batched_from_gpu(
                        [batch_memory_objs
                         ],  # Wrap in list since we're doing one layer
                        relative_starts,
                        relative_ends,
                        **batch_kwargs)

                    # Process the generator
                    next(mem_obj_generator)  # Initial setup and transfer data
                    total_offload_time += time.perf_counter() - t

                    # Store the batch's data in backend
                    t = time.perf_counter()
                    self.storage_manager.batched_put(batch_valid_keys,
                                                     batch_memory_objs)
                    total_batch_put_time += time.perf_counter() - t
                    # Commit the put operation
                    self.storage_manager.commit_put()
                    total_put_time += time.perf_counter() - t

            # Record layer time
            layer_time = time.perf_counter() - layer_st
            layerwise_times.append(layer_time)

            # Yield after processing each layer
            yield
        # Calculate and print metrics
        ed = time.perf_counter()
        total_time = ed - st
        avg_layer_time = sum(layerwise_times) / len(
            layerwise_times) if layerwise_times else 0
        max_layer_time = max(layerwise_times) if layerwise_times else 0
        min_layer_time = min(layerwise_times) if layerwise_times else 0

        logger.info(
            "Store %d tokens takes: %.4f ms, throughput: %.4f GB/s; "
            "layerwise time (avg/max/min): %.4f/%.4f/%.4f ms; "
            "offload_time: %.4f ms, put_time: %.4f ms", num_tokens,
            total_time * 1000, total_kv_size / total_time / 1024**3,
            avg_layer_time * 1000, max_layer_time * 1000,
            min_layer_time * 1000, total_offload_time * 1000,
            total_put_time * 1000)
        logger.info(
            f"total prepare time: {total_prepare_time * 1000} ms; total put time (w/o prepare): {(total_put_time-total_prepare_time) * 1000} ms"
        )
        logger.info(f"total batch put time: {total_batch_put_time * 1000} ms")

        self.stats_monitor.on_store_finished(monitor_req_id)
        logger.debug(f"Stored {num_tokens} "
                     f"out of total {len(tokens)} tokens")
        yield

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def retrieve_layer(
            self,
            tokens: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            **kwargs) -> Generator[Optional[torch.Tensor], None, None]:
        """
        Retrieve the KV cache in a layerwise manner.
        
        :param torch.Tensor tokens: The tokens of the corresponding KV caches.
        
        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched.
        
        :param **kwargs: The additional arguments for the storage backend which
            will be passed into the gpu_connector.
        
        return: A generator that yields Optional[torch.Tensor]. The tensor will
            be the boolean mask indicating which tokens are retrieved and will
            only be returned in the last iteration. In the first iteration, 
            the generator retrieve the memory objects of the first layer from
            the storage backends. In the next iterations, it moves the KV cache 
            of layer i from the memory objects (on CPU) to GPU and retrieves
            the memory objects of layer i+1 from the storage backends. In the 
            last iteration, it moves the memory objects of the last layer to
            the GPU.
        """

        if mask is not None:
            num_tokens = torch.sum(mask).item()
        else:
            num_tokens = len(tokens)
        monitor_req_id = self.stats_monitor.on_retrieve_request(num_tokens)

        ret_mask = torch.zeros_like(tokens, dtype=torch.bool, device="cpu")

        # Process tokens to get starts, ends, and base keys
        token_chunks = list(self.token_database.process_tokens(tokens, mask))
        chunk_starts = []
        chunk_ends = []
        chunk_keys = []
        for start, end, base_key in token_chunks:
            assert isinstance(base_key, CacheEngineKey)

            keys_multi_layer = base_key.split_layers(self.num_layers)

            # NOTE: Only check the first layer
            if not self.storage_manager.contains(keys_multi_layer[0]):
                break

            chunk_starts.append(start)
            chunk_ends.append(end)
            chunk_keys.append(keys_multi_layer)

            ret_mask[start:end] = True

        if chunk_keys:
            # Process chunks in batches
            for batch_start in range(0, len(chunk_starts), self.batch_size):
                batch_end = min(batch_start + self.batch_size,
                                len(chunk_starts))
                logger.debug(f"Processing batch {batch_start}:{batch_end}")

                # Get the current batch
                batch_starts = chunk_starts[batch_start:batch_end]
                batch_ends = chunk_ends[batch_start:batch_end]
                batch_keys = chunk_keys[batch_start:batch_end]

                # Create batch-specific kwargs
                batch_kwargs = kwargs.copy()
                if "slot_mapping" in kwargs:
                    slot_mapping = kwargs["slot_mapping"]
                    min_start = min(batch_starts)
                    max_end = max(batch_ends)
                    batch_kwargs["slot_mapping"] = slot_mapping[
                        min_start:max_end]

                    # Adjust starts and ends to be relative to the sliced slot_mapping
                    relative_starts = [s - min_start for s in batch_starts]
                    relative_ends = [e - min_start for e in batch_ends]
                else:
                    relative_starts = batch_starts
                    relative_ends = batch_ends

                # Set up the consumer for this batch
                assert isinstance(self.gpu_connector,
                                  VLLMPagedMemLayerwiseGPUConnector)
                mem_obj_consumer = self.gpu_connector.batched_to_gpu(
                    relative_starts, relative_ends, **batch_kwargs)
                next(mem_obj_consumer)  # Initial setup

                # Process each layer for this batch
                batch_memory_objs = []
                for layer_id in range(self.num_layers):
                    # Extract keys for current layer from each chunk
                    layer_keys = [
                        chunk_keys[layer_id] for chunk_keys in batch_keys
                    ]

                    # Get memory objects for current layer
                    get_generator = self.storage_manager.layerwise_batched_get(
                        [layer_keys])
                    get_tasks = next(get_generator)
                    assert None not in get_tasks

                    yield None  # Allow cooperative multitasking

                    # Get results and send to consumer
                    layer_memory_objs = [
                        retrieve_task.result() for retrieve_task in get_tasks
                    ]
                    mem_obj_consumer.send(layer_memory_objs)
                    batch_memory_objs.extend(layer_memory_objs)

                    # Unpin the current layer's keys
                    self.storage_manager.batched_unpin(layer_keys)

                # Final sync for this batch
                next(mem_obj_consumer)

                # Clean up memory objects for this batch
                for mem_obj in batch_memory_objs:
                    mem_obj.ref_count_down()

                yield None

            # NOTE (Shufan): 这部分代码是和retrieve里面丢弃当前key的功能对齐的
            for batch_keys in batch_keys:
                for layer_keys in batch_keys:
                    self.storage_manager.remove(layer_keys)

        else:
            # If no cache is found, we still need to yield for each layer
            # to maintain the generator protocol
            for _ in range(self.num_layers + 1):
                yield None

        retrieved_tokens = torch.sum(ret_mask)
        self.stats_monitor.on_retrieve_finished(monitor_req_id,
                                                retrieved_tokens)
        logger.debug(f"Retrieved {retrieved_tokens} "
                     f"out of {num_tokens} "
                     f"out of total {len(tokens)} tokens")

        yield ret_mask

    def lookup(
        self,
        tokens: Union[torch.Tensor, List[int]],
        search_range: Optional[List[str]] = None,
        pin: bool = False,
    ) -> int:
        """
        Checks the existence of KV cache of the tokens from the cache engine.

        :param tokens: the input tokens, with shape [seq_len]
        
        :param Optional[List[str]] search_range: The range of storage backends
        to search in. Should be a subset of 
        ["LocalCPUBackend", "LocalDiskBackend"] for now.
        If None, search in all backends.
        
        :param bool pin: If True, pin the KV cache in the storage.

        :return: An int indicating how many prefix tokens are cached.
        """
        end = 0
        for start, end, key in self.token_database.process_tokens(tokens):
            assert isinstance(key, CacheEngineKey)

            # TODO(Jiayi): Optimize by checking only the existence of the key
            # of one layer
            key_all_layers = key.split_layers(self.num_layers)
            for key_single_layer in key_all_layers:
                if not self.storage_manager.contains(key_single_layer,
                                                     search_range, pin):
                    return start
        return end


class LayerAwareLMCacheEngine(LMCacheEngine):
    """
    A specialized cache engine that handles layer-aware caching for disaggregated inference.
    
    This engine processes layers in layer-first order (all chunks of layer-0, then all chunks
    of layer-1, etc.) and provides ultra-low latency layer readiness detection. Designed for
    scenarios where decoder needs to start processing as soon as individual layers are available
    from the prefiller.
    
    Key features:
    - Layer-first processing order for optimal pipelining
    - Ultra-low latency layer readiness tracking (~100ns)
    - Progressive layer availability signaling
    - Integration with LayerFirstTokenDatabase
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
        memory_allocator: MemoryAllocatorInterface,
        token_database: TokenDatabase,
        gpu_connector: GPUConnectorInterface,
        layerwise: bool = True,
    ):
        super().__init__(config, metadata, memory_allocator, token_database,
                         gpu_connector, layerwise)
        self.num_layers = metadata.kv_shape[0]

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def _group_keys_by_layers_first(
            self,
            tokens: torch.Tensor,
            mask: Optional[torch.Tensor] = None) -> Dict[int, List[tuple]]:
        """
        Group layer keys by layer-first order for progressive transfer.
        
        Returns:
            Dict mapping layer_id -> [(start, end, layer_key), ...]
        """
        layers_data = {}

        # Add fine-grained profiling for the generator loop
        with NVTXContext("token_processing_setup"):
            token_processor = self.token_database.process_tokens(tokens, mask)

        iteration_count = 0
        for start, end, layer_key in token_processor:
            with NVTXContext(f"process_token_chunk_{iteration_count}"):
                assert isinstance(layer_key, LayerCacheEngineKey)

                layer_id = layer_key.layer_id
                if layer_id not in layers_data:
                    layers_data[layer_id] = []

                layers_data[layer_id].append((start, end, layer_key))
                iteration_count += 1

        return layers_data

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def _get_layer_keys_for_single_layer(
            self,
            layer_id: int,
            tokens: torch.Tensor,
            mask: Optional[torch.Tensor] = None) -> List[tuple]:
        """
        Get layer keys for a specific layer only, avoiding unnecessary computation.
        
        
        Args:
            layer_id: The specific layer ID to get keys for
            tokens: Input tokens to process
            mask: Optional mask for the tokens
            
        Returns:
            List of (start, end, layer_key) tuples for the specified layer only
        """
        layer_chunks = []

        # Use the optimized layer-specific method
        with NVTXContext("token_processing_setup_single_layer"):
            token_processor = self.token_database.process_tokens_for_layer(
                tokens, layer_id, mask)

        for start, end, layer_key in token_processor:
            with NVTXContext(f"process_token_chunk_layer_{layer_id}"):
                assert isinstance(layer_key, LayerCacheEngineKey)
                assert layer_key.layer_id == layer_id
                layer_chunks.append((start, end, layer_key))

        return layer_chunks

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def store_progressive_layers(self,
                                 tokens: torch.Tensor,
                                 mask: Optional[torch.Tensor] = None,
                                 layer_ids: Optional[List[int]] = None,
                                 **kwargs) -> None:
        """
        Store KV cache data for specific layers with RDMA-aware zero-copy transfers.
        
        This method is optimized for disaggregated inference scenarios where:
        1. Prefiller sends layers progressively (layer-0, then layer-1, etc.)
        2. Decoder can start processing layer-0 + hidden_states while layer-1 is still transferring
        3. Each layer is processed independently for true pipelining
        4. Zero-copy transfers are used for optimal performance
        
        Args:
            tokens: Input tokens to process
            mask: Optional mask for selective token storage
            layer_ids: Optional list of layer IDs to store. If None, stores all layers.
            notify_readiness: Whether to signal layer readiness immediately (default: True)
            **kwargs: Additional arguments for GPU connector
        """
        st = time.perf_counter()
        if mask is not None:
            num_tokens = torch.sum(mask).item()
        else:
            num_tokens = len(tokens)
        monitor_req_id = self.stats_monitor.on_store_request(num_tokens)

        # Get layer-first grouped data
        layers_data = self._group_keys_by_layers_first(tokens, mask)
        if not layers_data:
            logger.warning("No layer data to store")
            self.stats_monitor.on_store_finished(monitor_req_id)
            return

        # Determine which layers to process
        if layer_ids is None:
            layer_ids = sorted(layers_data.keys())
        else:
            # Validate layer IDs
            for layer_id in layer_ids:
                if layer_id < 0 or layer_id >= self.num_layers:
                    raise ValueError(f"Invalid layer ID: {layer_id}")
                if layer_id not in layers_data:
                    logger.warning(f"No data for layer {layer_id}")

        # Determine memory format based on GPU connector type
        memory_format = self._get_memory_format_for_connector()

        # Check if we're using NixlBackend for RDMA
        using_nixl = isinstance(self.storage_manager,
                                DistributedStorageManager)

        total_stored = 0
        layer_timings = {}

        # Process layers progressively: layer-0, then layer-1, etc.
        # Each layer gets its own register->flush cycle for true pipelining
        for layer_id in layer_ids:
            layer_stored = self._store_single_layer_progressive(
                layer_id, layers_data[layer_id], memory_format, using_nixl,
                **kwargs)

            total_stored += layer_stored

        ed = time.perf_counter()
        total_time = ed - st

        logger.info(
            f"Progressive layer store completed: {total_stored} chunks across "
            f"{len(layer_ids)} layers in {total_time*1000:.2f}ms")
        logger.info(
            f"Using {'RDMA zero-copy' if using_nixl else 'standard'} transfer with {memory_format}"
        )

        if using_nixl:
            logger.info(
                "Layer readiness tracking via NixlObserver - layers will be marked ready when data is received"
            )

        self.stats_monitor.on_store_finished(monitor_req_id)

    def _get_memory_format_for_connector(self) -> MemoryFormat:
        """
        Determine the correct memory format based on the GPU connector type.
        
        Returns:
            MemoryFormat.KV_T2D for VLLMPagedMemLayerwiseGPUConnector (token-first)
            MemoryFormat.KV_2LTD for other connectors (layer-first)
        """
        if isinstance(self.gpu_connector, VLLMPagedMemLayerwiseGPUConnector):
            # VLLMPagedMemLayerwiseGPUConnector uses token-first format: [num_tokens, 2, hidden_dim]
            return MemoryFormat.KV_T2D
        else:
            # Other connectors use layer-first format: [2, num_layers, num_tokens, hidden_dim]
            return MemoryFormat.KV_2LTD

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def _store_single_layer_progressive(self, layer_id: int,
                                        layer_chunks: List[tuple],
                                        memory_format: MemoryFormat,
                                        using_nixl: bool, **kwargs) -> int:
        """
        Store a single layer with progressive transfer capability.
        
        This method handles each layer independently, allowing for true pipelining
        where decoder can process layer-0 while layer-1 is still being transferred.
        
        Args:
            layer_id: The layer ID to process
            layer_chunks: List of (start, end, layer_key) tuples for this layer
            memory_format: Memory format to use (KV_T2D or KV_2LTD)
            using_nixl: Whether we're using NixlBackend for RDMA
            **kwargs: Additional arguments for GPU connector
            
        Returns:
            Number of chunks successfully stored for this layer
        """
        layer_st = time.perf_counter()
        layer_stored = 0

        logger.debug(
            f"Processing layer {layer_id} with {len(layer_chunks)} chunks "
            f"using {memory_format} format")

        if using_nixl:
            # RDMA path: each layer gets its own register->flush cycle
            layer_stored = self._store_layer_rdma(layer_id, layer_chunks,
                                                  memory_format, **kwargs)
        else:
            # Standard path: direct allocation and transfer
            layer_stored = self._store_layer_standard(layer_id, layer_chunks,
                                                      memory_format, **kwargs)

        layer_time = time.perf_counter() - layer_st
        logger.debug(
            f"Layer {layer_id} completed: {layer_stored} chunks in {layer_time*1000:.2f}ms"
        )

        return layer_stored

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def _store_layer_rdma(self, layer_id: int, layer_chunks: List[tuple],
                          memory_format: MemoryFormat, **kwargs) -> int:
        """
        Store a single layer using RDMA zero-copy transfers.
        
        Each layer gets its own register->flush cycle to enable progressive transfer.
        Uses batched operations for optimal performance.
        
        Args:
            layer_id: The layer ID to store
            layer_chunks: List of (start, end, layer_key) tuples for this layer
            memory_format: Memory format to use (KV_T2D or KV_2LTD)
            **kwargs: Additional arguments for GPU connector
            
        Returns:
            Number of chunks successfully stored for this layer
        """
        # Collect keys and metadata for this layer
        layer_keys = []
        layer_metadata = []
        valid_chunks = []

        # First pass: prepare metadata for all chunks in this layer
        for start, end, layer_key in layer_chunks:
            if self.storage_manager.contains(layer_key):
                continue

            # Get chunk-specific KV cache shape
            num_chunk_tokens = end - start
            kv_shape = self.gpu_connector.get_shape(num_chunk_tokens)
            kv_dtype = self.metadata.kv_dtype

            # Prepare metadata with correct format
            metadata = self.storage_manager.dry_allocate(
                kv_shape,
                kv_dtype,
                fmt=
                memory_format  # Use KV_T2D for VLLMPagedMemLayerwiseGPUConnector
            )

            layer_keys.append(layer_key)
            layer_metadata.append(metadata)
            valid_chunks.append((start, end, layer_key))

        if not layer_keys:
            return 0

        # Register this layer's operations (each layer gets its own register->flush cycle)
        logger.debug(
            f"Registering RDMA operations for layer {layer_id}: {len(layer_keys)} chunks"
        )
        self.storage_manager.prepare_put(layer_keys, layer_metadata)

        # Second pass: allocate zero-copy memory for all chunks in this layer
        memory_objs = []
        successful_keys = []
        starts = []
        ends = []

        for i, (start, end, layer_key) in enumerate(valid_chunks):
            metadata = layer_metadata[i]

            # Allocate zero-copy memory
            memory_obj = self.storage_manager.allocate(metadata.shape,
                                                       metadata.dtype,
                                                       fmt=metadata.fmt)

            if memory_obj is None:
                logger.warning(
                    f"Failed to allocate zero-copy memory for layer {layer_id}, "
                    f"chunk {start}:{end}")
                continue

            # Collect for batch transfer
            memory_objs.append(memory_obj)
            successful_keys.append(layer_key)
            starts.append(start)
            ends.append(end)

            # Update lookup server
            if self.lookup_server is not None:
                self.lookup_server.insert(layer_key)

        # Batch transfer data FROM GPU TO memory objects
        if memory_objs:
            logger.debug(
                f"Batch transferring {len(memory_objs)} chunks FROM GPU for layer {layer_id}"
            )

            # Use the simpler single-layer method instead of complex generator
            assert isinstance(self.gpu_connector, VLLMPagedMemLayerwiseGPUConnector), \
                "from_gpu_single_layer method requires VLLMPagedMemLayerwiseGPUConnector"

            self.gpu_connector.from_gpu_single_layer(memory_objs, starts, ends,
                                                     layer_id, **kwargs)

        # Batch put all memory objects for this layer at once
        if memory_objs:
            logger.debug(
                f"Batch putting {len(memory_objs)} memory objects for layer {layer_id}"
            )
            self.storage_manager.batched_put(successful_keys, memory_objs)

        # Flush this layer's RDMA operations
        logger.debug(f"Flushing RDMA operations for layer {layer_id}")
        self.storage_manager.commit_put()

        return len(memory_objs)

    def _store_layer_standard(self, layer_id: int, layer_chunks: List[tuple],
                              memory_format: MemoryFormat, **kwargs) -> int:
        """
        Store a single layer using standard (non-RDMA) transfers.
        """
        # Collect valid chunks and allocate memory objects
        memory_objs = []
        successful_keys = []
        starts = []
        ends = []

        # Process all chunks for this layer
        for start, end, layer_key in layer_chunks:
            if self.storage_manager.contains(layer_key):
                continue

            # Get chunk-specific KV cache shape
            num_chunk_tokens = end - start
            kv_shape = self.gpu_connector.get_shape(num_chunk_tokens)
            kv_dtype = self.metadata.kv_dtype

            # Allocate memory with correct format
            memory_obj = self.storage_manager.allocate(kv_shape,
                                                       kv_dtype,
                                                       fmt=memory_format)

            if memory_obj is None:
                logger.warning(
                    f"Failed to allocate memory for layer {layer_id}, "
                    f"chunk {start}:{end}")
                continue

            # Collect for batch transfer
            memory_objs.append(memory_obj)
            successful_keys.append(layer_key)
            starts.append(start)
            ends.append(end)

            # Update lookup server
            if self.lookup_server is not None:
                self.lookup_server.insert(layer_key)

        # Batch transfer data from GPU to memory objects
        if memory_objs:
            logger.debug(
                f"Batch transferring {len(memory_objs)} chunks from GPU for layer {layer_id}"
            )

            # Format memory_objs for batched_from_gpu: List[List[MemoryObj]]
            # Since we're processing one layer, we need [memory_objs] (layer dimension)
            memory_objs_batched = [memory_objs]

            # Use batched GPU transfer
            gpu_generator = self.gpu_connector.batched_from_gpu(
                memory_objs_batched, starts, ends, **kwargs)

            # Execute the generator (first call sets up, second call transfers this layer)
            next(gpu_generator)  # Setup
            next(gpu_generator)  # Transfer layer data

            # Batch put all memory objects
            self.storage_manager.batched_put(successful_keys, memory_objs)

        return len(memory_objs)

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def retrieve_layer_when_ready(self,
                                  layer_id: int,
                                  tokens: torch.Tensor,
                                  mask: Optional[torch.Tensor] = None,
                                  timeout_us: int = 1000,
                                  **kwargs) -> Optional[torch.Tensor]:
        """
        Retrieve layer data as soon as it becomes ready.
        
        Args:
            layer_id: Specific layer to retrieve
            tokens: Input tokens to process
            mask: Optional mask for selective token retrieval
            timeout_us: Timeout in microseconds for layer readiness
            **kwargs: Additional arguments for GPU connector
            
        Returns:
            Boolean mask indicating which tokens were retrieved, or None if timeout
        """
        # Wait for layer to be ready with microsecond precision
        num_chunks = kwargs.get('num_chunks', 1) # At least 1 chunk.
        if not self.storage_manager.storage_backend.wait_for_layer_busy(
                layer_id, timeout_us, num_chunks):
            logger.debug(f"Layer {layer_id} not ready within {timeout_us}μs")
            return None

        st = time.perf_counter()
        if mask is not None:
            num_tokens = torch.sum(mask).item()
        else:
            num_tokens = len(tokens)
        monitor_req_id = self.stats_monitor.on_retrieve_request(num_tokens)

        # Check if we're using NixlBackend for RDMA
        using_nixl = isinstance(self.storage_manager,
                                DistributedStorageManager)

        # Determine memory format based on GPU connector type
        memory_format = self._get_memory_format_for_connector()

        # Get layer-specific chunks using optimized single-layer processing
        layer_chunks = self._get_layer_keys_for_single_layer(
            layer_id, tokens, mask)
        if not layer_chunks:
            logger.debug(f"No data for layer {layer_id}")
            self.stats_monitor.on_retrieve_finished(monitor_req_id, 0)
            return None

        # Retrieve data using the appropriate method
        if using_nixl:
            ret_mask = self._retrieve_layer_rdma(layer_id, layer_chunks,
                                                 memory_format, **kwargs)
        else:
            ret_mask = self._retrieve_layer_standard(layer_id, layer_chunks,
                                                     memory_format, **kwargs)

        ed = time.perf_counter()
        retrieved_tokens = torch.sum(
            ret_mask).item() if ret_mask is not None else 0

        logger.debug(
            f"Layer {layer_id} retrieved: {retrieved_tokens} tokens "
            f"in {(ed-st)*1000:.2f}ms using {'RDMA' if using_nixl else 'standard'} transfer"
        )

        self.stats_monitor.on_retrieve_finished(monitor_req_id,
                                                retrieved_tokens)
        return ret_mask

    def wait_for_early_layers(self,
                              num_layers: int = 1,
                              timeout_us: int = 5000) -> List[int]:
        """
        Wait for early layers to become ready for immediate processing.
        
        Args:
            num_layers: Number of early layers to wait for
            timeout_us: Total timeout in microseconds
            
        Returns:
            List of ready layer IDs
        """
        early_layers = list(range(min(num_layers, self.num_layers)))
        return self.storage_manager.storage_backend.wait_for_layers_batch(
            early_layers, timeout_us)

    def signal_layers_ready(self,
                            layer_ids: List[int],
                            chunk_counts: Optional[List[int]] = None) -> None:
        """
        Manually signal that specific layers are ready for consumption.
        
        This is useful when you've stored layers with notify_readiness=False
        and want to signal readiness at a later time (e.g., after validation
        or after storing multiple layers).
        
        Args:
            layer_ids: List of layer IDs to mark as ready
            chunk_counts: Optional list of chunk counts per layer. If None, uses 1 for each layer.
        """
        if chunk_counts is None:
            chunk_counts = [1] * len(layer_ids)
        elif len(chunk_counts) != len(layer_ids):
            raise ValueError("chunk_counts length must match layer_ids length")

        for layer_id, chunk_count in zip(layer_ids, chunk_counts):
            if layer_id < 0 or layer_id >= self.num_layers:
                logger.warning(f"Invalid layer_id {layer_id}, skipping")
                continue

            self.storage_manager.storage_backend.mark_layer_ready(
                layer_id, chunk_count)
            logger.debug(
                f"Manually signaled layer {layer_id} ready with {chunk_count} chunks"
            )

        logger.info(f"Signaled {len(layer_ids)} layers as ready")

    def signal_all_layers_ready(self) -> None:
        """
        Signal all layers as ready.
        
        This is a convenience method for cases where you want to signal
        all layers at once after storing them.
        """
        layer_ids = list(range(self.num_layers))
        self.signal_layers_ready(layer_ids)
        logger.info(f"Signaled all {self.num_layers} layers as ready")

    def _retrieve_layer_standard(self, layer_id: int,
                                 layer_chunks: List[tuple],
                                 memory_format: MemoryFormat,
                                 **kwargs) -> Optional[torch.Tensor]:
        """
        Retrieve a single layer using standard (non-RDMA) transfers.
        
        Args:
            layer_id: The layer ID to retrieve
            layer_chunks: List of (start, end, layer_key) tuples for this layer
            memory_format: Memory format to use (KV_T2D or KV_2LTD)
            **kwargs: Additional arguments for GPU connector
            
        Returns:
            Boolean mask indicating which tokens were retrieved, or None if no data
        """
        memory_objs = []
        successful_ranges = []
        tokens_length = None

        # Collect all memory objects for this layer
        for start, end, layer_key in layer_chunks:
            if tokens_length is None:
                # Infer tokens length from the first chunk
                tokens_length = max(end for _, end, _ in layer_chunks)

            memory_obj = self.storage_manager.get(layer_key)
            if memory_obj is not None:
                memory_objs.append(memory_obj)
                successful_ranges.append((start, end))

        if not memory_objs:
            logger.debug(f"No memory objects found for layer {layer_id}")
            return None

        # Create return mask
        ret_mask = torch.zeros(tokens_length, dtype=torch.bool, device="cpu")

        # Transfer data to GPU using single layer method if possible, otherwise use generator
        if memory_objs:
            logger.debug(
                f"Transferring {len(memory_objs)} chunks to GPU for layer {layer_id}"
            )

            # Prepare arguments for single layer GPU transfer
            starts = [start for start, end in successful_ranges]
            ends = [end for start, end in successful_ranges]

            # Use direct method if available, otherwise fall back to generator
            if isinstance(self.gpu_connector,
                          VLLMPagedMemLayerwiseGPUConnector):
                self.gpu_connector.to_gpu_single_layer(memory_objs, starts,
                                                       ends, layer_id,
                                                       **kwargs)
            else:
                # Fallback to generator pattern for other connector types
                gpu_generator = self.gpu_connector.batched_to_gpu(
                    starts, ends, **kwargs)
                next(gpu_generator)  # Setup and prepare metadata
                gpu_generator.send(
                    memory_objs)  # Send memory objects for this layer
                try:
                    next(gpu_generator)  # Complete the transfer
                except StopIteration:
                    pass  # Generator finished

            # Mark tokens as retrieved
            for start, end in successful_ranges:
                ret_mask[start:end] = True

            # Cleanup memory objects
            for memory_obj in memory_objs:
                memory_obj.ref_count_down()

        return ret_mask

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def _retrieve_layer_rdma(self, layer_id: int, layer_chunks: List[tuple],
                             memory_format: MemoryFormat,
                             **kwargs) -> Optional[torch.Tensor]:
        """
        Retrieve a single layer using RDMA zero-copy transfers.
        
        Args:
            layer_id: The layer ID to retrieve
            layer_chunks: List of (start, end, layer_key) tuples for this layer
            memory_format: Memory format to use (KV_T2D or KV_2LTD)
            **kwargs: Additional arguments for GPU connector
            
        Returns:
            Boolean mask indicating which tokens were retrieved, or None if no data
        """
        memory_objs = []
        successful_ranges = []
        tokens_length = None

        # Collect all memory objects for this layer
        for start, end, layer_key in layer_chunks:
            if tokens_length is None:
                # Infer tokens length from the first chunk
                tokens_length = max(end for _, end, _ in layer_chunks)

            memory_obj = self.storage_manager.get(layer_key)
            if memory_obj is not None:
                memory_objs.append(memory_obj)
                successful_ranges.append((start, end))

        if not memory_objs:
            logger.debug(f"No memory objects found for layer {layer_id}")
            return None

        # Create return mask
        ret_mask = torch.zeros(tokens_length, dtype=torch.bool, device="cpu")

        # Transfer data TO GPU using the simplified single-layer method
        if memory_objs:
            logger.debug(
                f"Transferring {len(memory_objs)} chunks TO GPU for layer {layer_id}"
            )

            # Prepare arguments for single layer GPU transfer
            starts = [start for start, end in successful_ranges]
            ends = [end for start, end in successful_ranges]

            # Use the new to_gpu_single_layer method for direct transfer
            assert isinstance(self.gpu_connector, VLLMPagedMemLayerwiseGPUConnector), \
                "to_gpu_single_layer method requires VLLMPagedMemLayerwiseGPUConnector"

            self.gpu_connector.to_gpu_single_layer(memory_objs, starts, ends,
                                                   layer_id, **kwargs)

            # Mark tokens as retrieved
            for start, end in successful_ranges:
                ret_mask[start:end] = True

            # Cleanup memory objects
            for memory_obj in memory_objs:
                memory_obj.ref_count_down()

        return ret_mask


class LMCacheEngineBuilder:
    _instances: Dict[str, LMCacheEngine] = {}
    _cfgs: Dict[str, LMCacheEngineConfig] = {}
    _metadatas: Dict[str, LMCacheEngineMetadata] = {}
    _stat_loggers: Dict[str, LMCacheStatsLogger] = {}

    @staticmethod
    def _Create_memory_allocator(
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
    ) -> MemoryAllocatorInterface:
        if config.enable_nixl:
            assert config.nixl_buffer_device is not None
            return AdHocMemoryAllocator(config.nixl_buffer_device)

        max_local_cpu_size = config.max_local_cpu_size
        return MixedMemoryAllocator(int(max_local_cpu_size * 1024**3))

    @staticmethod
    def _Create_token_database(
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
    ) -> TokenDatabase:
        return ChunkedTokenDatabase(config, metadata)

    @classmethod
    def get_or_create(
        cls,
        instance_id: str,
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
        gpu_connector: GPUConnectorInterface,
        use_layerwise_engine: bool = False,
    ) -> LMCacheEngine:
        """
        Builds a new LMCacheEngine instance if it doesn't already exist for the
        given ID.

        raises: ValueError if the instance already exists with a different
            configuration.
        """
        logger.info(f"Creating LMCacheEngine instance {instance_id}")
        if instance_id not in cls._instances:
            memory_allocator = cls._Create_memory_allocator(config, metadata)
            token_database = cls._Create_token_database(config, metadata)
            stat_logger = LMCacheStatsLogger(metadata, log_interval=10)

            # HACK(Jiayi): Merge two types of engine into one in the future
            engine: Union[LayerwiseLMCacheEngine, LMCacheEngine]
            if use_layerwise_engine:
                engine = LayerwiseLMCacheEngine(config, metadata,
                                                memory_allocator,
                                                token_database, gpu_connector)
            else:
                engine = LMCacheEngine(config, metadata, memory_allocator,
                                       token_database, gpu_connector)
            cls._instances[instance_id] = engine
            cls._cfgs[instance_id] = config
            cls._metadatas[instance_id] = metadata
            cls._stat_loggers[instance_id] = stat_logger
            return engine
        else:
            if (cls._cfgs[instance_id] != config
                    or cls._metadatas[instance_id] != metadata):
                raise ValueError(
                    f"Instance {instance_id} already exists with a different "
                    f"configuration or metadata.")
            return cls._instances[instance_id]

    @classmethod
    def get(cls, instance_id: str) -> Optional[LMCacheEngine]:
        """Returns the LMCacheEngine instance associated with the instance ID, 
        or None if not found."""
        return cls._instances.get(instance_id)

    @classmethod
    def destroy(cls, instance_id: str) -> None:
        """Close and delete the LMCacheEngine instance by the instance ID"""
        # TODO: unit test for this
        if instance_id in cls._instances:
            stat_logger = cls._stat_loggers[instance_id]
            stat_logger.shutdown()
            engine = cls._instances[instance_id]
            engine.close()
            cls._instances.pop(instance_id, None)
            cls._cfgs.pop(instance_id, None)
            cls._metadatas.pop(instance_id, None)
            cls._stat_loggers.pop(instance_id, None)
            LMCStatsMonitor.DestroyInstance()
