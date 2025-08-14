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
from lmcache.experimental.token_database import (ChunkedTokenDatabase, LayerFirstTokenDataBase,
                                                 TokenDatabase)
from lmcache.logging import init_logger
from lmcache.observability import LMCacheStatsLogger, LMCStatsMonitor
from lmcache.usage_context import InitializeUsageContext
from lmcache.utils import CacheEngineKey, LayerCacheEngineKey, CombinedLayerCacheEngineKey, _lmcache_nvtx_annotate, NVTXContext
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
        with NVTXContext("prepare keys and metadatas"):
            for start, end, key in self.token_database.process_tokens(
                    tokens, mask):
                assert isinstance(key, CacheEngineKey)
                # Allocate the memory object
                num_tokens = end - start
                kv_shape = self.gpu_connector.get_shape(num_tokens)
                kv_dtype = self.metadata.kv_dtype
                memobj_meta = self.storage_manager.dry_allocate(
                    kv_shape, kv_dtype)
                assert memobj_meta is not None
                keys.append(key)
                metadatas.append(memobj_meta)
                steds.append((start, end))

        self.storage_manager.prepare_put(keys, metadatas, priority=0)

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

    # ==============================
    # Special Tensor Methods (for logits and high-priority transfers)
    # ==============================

    def store_special_tensor(self, key: str, tensor: torch.Tensor) -> None:
        """
        Store small tensors (like logits) with high priority.
        
        This method should use a separate high-priority queue that preempts KV transfers.
        
        Args:
            key: Unique identifier for the tensor (should use token hash instead of req_id)
            tensor: The tensor to store (e.g., logits)
        """
        pass

    def retrieve_special_tensor(self, key: str) -> Optional[torch.Tensor]:
        """
        Retrieve special tensor from recv_obj_pool immediately.
        
        This should preempt other transfers and return immediately from recv_obj_pool.
        
        Args:
            key: Unique identifier for the tensor
            
        Returns:
            The tensor if available, None otherwise
        """
        return None

    def check_special_tensor_available(self, key: str) -> bool:
        """
        Check availability of special tensor without consuming it.
        
        This should check high-priority queue first before regular KV transfers.
        
        Args:
            key: Unique identifier for the tensor
            
        Returns:
            True if tensor is available, False otherwise
        """
        return False


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
                self.storage_manager.prepare_put(batch_keys, batch_metadata, priority=0)
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
    - Option 1 approach: CombinedLayerCacheEngineKey with embedded chunk mappings
    
    Architecture (Option 1):
    The engine uses CombinedLayerCacheEngineKey objects that contain embedded chunk mappings.
    This eliminates the need for separate chunk mapping storage and enables true disaggregated
    inference where sender and receiver machines don't share storage state. Chunk mappings
    travel automatically with the data over the NixL protocol.
    
    Storage:
    - Individual chunks are combined into single memory objects per layer
    - CombinedLayerCacheEngineKey contains embedded chunk-to-offset mappings
    - Lightweight CombinedObjectReference objects point individual keys to combined objects
    - No separate chunk mapping storage required
    
    Retrieval:
    - Individual chunk keys resolve to reference objects
    - References point to combined objects with embedded mappings
    - Chunk mappings are extracted from the combined key itself
    - Individual chunks are sliced from the combined tensor using embedded offsets
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
            if kwargs.get("use_combined_memory", True):
                layer_stored = self._store_layer_rdma(layer_id, layer_chunks,
                                                      memory_format, **kwargs)
            else:
                layer_stored = self._store_layer_single(
                    layer_id, layer_chunks, memory_format, **kwargs)
        else:
            raise NotImplementedError("Non-RDMA store is not implemented")

        layer_time = time.perf_counter() - layer_st
        logger.debug(
            f"Layer {layer_id} completed: {layer_stored} chunks in {layer_time*1000:.2f}ms"
        )

        return layer_stored

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def _store_layer_single(self, layer_id: int, layer_chunks: List[tuple],
                            memory_format: MemoryFormat, **kwargs) -> int:
        """
        Store a single layer using individual chunk processing with from_gpu_single_layer.
        
        This is an alternative to _store_layer_rdma for ablation studies. Instead of
        combining all chunks into a single memory object, this method just transfers the chunks together.
        
        Args:
            layer_id: The layer ID to store
            layer_chunks: List of (start, end, layer_key) tuples for this layer
            memory_format: Memory format to use (KV_T2D or KV_2LTD)
            **kwargs: Additional arguments for GPU connector
            
        Returns:
            Number of chunks successfully stored for this layer
        """
        if not layer_chunks:
            return 0

        # Step 1: Filter out already cached chunks
        valid_chunks = []
        starts = []
        ends = []

        with NVTXContext("filter_chunks"):
            for start, end, layer_key in layer_chunks:
                if not self.storage_manager.contains(layer_key):
                    valid_chunks.append((start, end, layer_key))
                    starts.append(start)
                    ends.append(end)

        if not valid_chunks:
            return 0

        # Step 2: Prepare metadata for all chunks
        chunk_keys = []
        chunk_metadatas = []

        with NVTXContext("prepare_metadata"):
            for start, end, layer_key in valid_chunks:
                num_tokens = end - start
                kv_shape = self.gpu_connector.get_shape(num_tokens)
                kv_dtype = self.metadata.kv_dtype

                metadata = self.storage_manager.dry_allocate(kv_shape,
                                                             kv_dtype,
                                                             fmt=memory_format)

                if metadata is None:
                    logger.warning(
                        f"Failed to prepare metadata for layer {layer_id}, "
                        f"chunk {start}:{end}")
                    continue

                chunk_keys.append(layer_key)
                chunk_metadatas.append(metadata)

        if not chunk_keys:
            logger.warning(f"No valid metadata for layer {layer_id}")
            return 0

        # Step 3: Register all metadata at once
        with NVTXContext("register_metadata"):
            self.storage_manager.prepare_put(chunk_keys, chunk_metadatas, priority=0)

        # Step 4: Transfer data from GPU to memory objects. Instead of using a combined keys and memorys, we allocate them individually, and batch them together.
        successful_chunks = 0
        memory_objs = []
        valid_keys = []

        with NVTXContext("create memory_objs"):
            # 1. generate memory_objs, starts, ends, layer_id.
            # 2. kv_caches, slot_mapping are passed in as kwargs.

            for i, ((start, end, layer_key),
                    metadata) in enumerate(zip(valid_chunks, chunk_metadatas)):
                # Allocate memory for this individual chunk
                memory_obj = self.storage_manager.allocate(metadata.shape,
                                                           metadata.dtype,
                                                           fmt=metadata.fmt)

                memory_objs.append(memory_obj)
                valid_keys.append(layer_key)
                successful_chunks += 1

                # Update lookup server for p2p discovery (if enabled)
                if self.lookup_server is not None:
                    self.lookup_server.insert(layer_key)

        with NVTXContext("gpu_to_memory_transfer"):
            # Transfer data using from_gpu_single_layer for this individual chunk
            try:
                self.gpu_connector.from_gpu_single_layer(
                    memory_objs, starts, ends, layer_id, **kwargs)
            except Exception as e:
                logger.error(f"Failed to transfer GPU data: {e}")

        # Step 5: Store all chunks in batch
        with NVTXContext("batch_store"):
            if memory_objs:
                self.storage_manager.batched_put(valid_keys, memory_objs)
                logger.debug(
                    f"Stored {len(memory_objs)} individual chunks for layer {layer_id}"
                )

        # Step 6: Flush operations
        with NVTXContext("flush"):
            self.storage_manager.commit_put()

        logger.debug(
            f"Successfully stored {successful_chunks} individual chunks for layer {layer_id}"
        )
        return successful_chunks

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def _store_layer_rdma(self, layer_id: int, layer_chunks: List[tuple],
                          memory_format: MemoryFormat, **kwargs) -> int:
        """
        Store a single layer using RDMA zero-copy transfers with optimized single allocation.
        
        CombinedLayerCacheEngineKey with embedded chunk mappings.
        This eliminates the need for separate chunk mapping storage and enables true
        disaggregated inference where sender and receiver don't share storage state.
        
        OPTIMIZATION: Single allocation for entire layer instead of multiple small allocations.
        
        Args:
            layer_id: The layer ID to store
            layer_chunks: List of (start, end, layer_key) tuples for this layer
            memory_format: Memory format to use (KV_T2D or KV_2LTD)
            **kwargs: Additional arguments for GPU connector
            
        Returns:
            Number of chunks successfully stored for this layer
        """
        if not layer_chunks:
            return 0

        # Step 1: Filter out already cached chunks and calculate total tokens
        valid_chunks = []
        total_tokens = 0

        with NVTXContext("filter"):
            for start, end, layer_key in layer_chunks:
                if not self.storage_manager.contains(layer_key):
                    valid_chunks.append((start, end, layer_key))
                    total_tokens += (end - start)

        if not valid_chunks:
            return 0

        logger.debug(
            f"RDMA store for layer {layer_id}: {len(valid_chunks)} chunks, "
            f"{total_tokens} total tokens (reduced from {len(layer_chunks)} chunks)"
        )

        # Step 2: Create combined key with embedded chunk mappings (Option 1)
        with NVTXContext("create_combined_key"):
            combined_layer_key = CombinedLayerCacheEngineKey.create_combined_key(
                valid_chunks, layer_id)

            # Calculate combined shape for all chunks in this layer
            combined_kv_shape = self.gpu_connector.get_shape(total_tokens)
            kv_dtype = self.metadata.kv_dtype

            # Single dry_allocate call for the entire layer
            combined_metadata = self.storage_manager.dry_allocate(
                combined_kv_shape, kv_dtype, fmt=memory_format)

            if combined_metadata is None:
                logger.warning(
                    f"Failed to prepare metadata for layer {layer_id}")
                return 0

            # Register the single large allocation
            self.storage_manager.prepare_put([combined_layer_key],
                                             [combined_metadata],
                                             priority=0)

            # Single allocate call for the entire layer
            combined_memory_obj = self.storage_manager.allocate(
                combined_metadata.shape,
                combined_metadata.dtype,
                fmt=combined_metadata.fmt)

            if combined_memory_obj is None:
                logger.warning(
                    f"Failed to allocate combined memory for layer {layer_id}")
                return 0

        # Step 3: Transfer data from GPU to the combined memory object
        with NVTXContext("gpu_to_memory_transfer"):
            # Create a list of chunk offsets for the GPU connector
            chunk_offsets = []
            current_offset = 0
            for start, end, layer_key in valid_chunks:
                num_chunk_tokens = end - start
                chunk_offsets.append((start, end, layer_key, current_offset,
                                      current_offset + num_chunk_tokens))
                current_offset += num_chunk_tokens

            success = self._transfer_gpu_to_combined_memory(
                combined_memory_obj, chunk_offsets, layer_id, **kwargs)

            if not success:
                logger.warning(
                    f"Failed to transfer GPU data for layer {layer_id}")
                combined_memory_obj.ref_count_down()
                return 0

        # Step 4: Store combined object (Option 1 - sender side)
        with NVTXContext("store_combined_object"):
            # Store the combined memory object - chunk mappings are embedded in the key!
            # The receiver will extract these mappings when it receives the combined key
            self.storage_manager.batched_put([combined_layer_key],
                                             [combined_memory_obj])

            # Update lookup server for p2p discovery (if enabled)
            if self.lookup_server is not None:
                self.lookup_server.insert(combined_layer_key)

            logger.debug(
                f"Stored combined object with {len(chunk_offsets)} chunks for layer {layer_id}"
            )
            logger.debug(
                f"Combined key contains {len(combined_layer_key.chunk_mappings)} embedded chunk mappings"
            )

        # Step 5: Flush RDMA operations
        with NVTXContext("flush_rdma"):
            self.storage_manager.commit_put()

        logger.debug(
            f"Successfully stored {len(valid_chunks)} chunks as single combined object for layer {layer_id}"
        )
        return len(valid_chunks)

    def _transfer_gpu_to_combined_memory(self, combined_memory_obj: MemoryObj,
                                         chunk_offsets: List[tuple],
                                         layer_id: int, **kwargs) -> bool:
        """
        Transfer GPU data to different offsets within the combined memory object.
        
        Uses the new GPU connector method for combined memory transfer.
        This is part of the Option 1 approach where individual chunks are written
        to different offsets within a single combined memory object.
        """
        try:
            # For VLLMPagedMemLayerwiseGPUConnector (KV_T2D format):
            # combined_memory_obj.tensor shape: [total_tokens, 2, hidden_dim]
            # Each chunk writes to combined_memory_obj.tensor[offset_start:offset_end]

            if isinstance(self.gpu_connector,
                          VLLMPagedMemLayerwiseGPUConnector):
                # Use the new combined memory transfer method
                self.gpu_connector.from_gpu_to_combined_memory(
                    combined_memory_obj, chunk_offsets, layer_id, **kwargs)
                return True
            else:
                # Fallback to individual transfers for other connector types
                logger.warning(
                    f"Combined memory transfer not implemented for {type(self.gpu_connector)}"
                )
                return False

        except Exception as e:
            logger.error(
                f"Failed to transfer GPU data to combined memory: {e}")
            return False

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def retrieve_layer_when_ready(self,
                                  layer_id: int,
                                  tokens: torch.Tensor,
                                  mask: Optional[torch.Tensor] = None,
                                  timeout_us: int = 1000,
                                  layers_data: Optional[Dict[
                                      int, List[tuple]]] = None,
                                  **kwargs) -> Optional[torch.Tensor]:
        """
        Retrieve layer data as soon as it becomes ready.
        
        Args:
            layer_id: Specific layer to retrieve
            tokens: Input tokens to process
            mask: Optional mask for selective token retrieval
            timeout_us: Timeout in microseconds for layer readiness
            layers_data: Optional pre-computed layer data from _group_keys_by_layers_first
            **kwargs: Additional arguments for GPU connector
            
        Returns:
            Boolean mask indicating which tokens were retrieved, or None if timeout
        """
        # Wait for layer to be ready with microsecond precision
        num_chunks = kwargs.get('num_chunks', 1)  # At least 1 chunk.
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

        # Get layer-specific chunks - use pre-computed if available, otherwise compute on-demand
        assert layers_data is not None and layer_id in layers_data, \
            f"Layer {layer_id} not found in layers_data"
        layer_chunks = layers_data[layer_id]

        if not layer_chunks:
            logger.debug(f"No data for layer {layer_id}")
            self.stats_monitor.on_retrieve_finished(monitor_req_id, 0)
            return None

        # Retrieve data using the appropriate method
        if using_nixl:
            ret_mask = self._retrieve_layer_rdma(layer_id, layer_chunks,
                                                 memory_format, **kwargs)
        else:
            raise NotImplementedError("Standard retrieval not implemented")

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

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def retrieve_layers_progressive(
            self,
            tokens: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            layer_ids: Optional[List[int]] = None,
            timeout_us: int = 1000,
            **kwargs) -> Dict[int, Optional[torch.Tensor]]:
        """
        Retrieve multiple layers progressively with pre-computed layer keys.
        
        This method pre-computes all layer keys upfront (similar to sender side)
        and then retrieves layers as they become ready, avoiding redundant
        key computation for each layer.
        
        Args:
            tokens: Input tokens to process
            mask: Optional mask for selective token retrieval
            layer_ids: Optional list of layer IDs to retrieve. If None, retrieves all layers.
            timeout_us: Timeout in microseconds for each layer readiness
            **kwargs: Additional arguments for GPU connector
            
        Returns:
            Dict mapping layer_id -> retrieval mask (or None if timeout/no data)
        """
        st = time.perf_counter()

        # Pre-compute all layer keys upfront, similar to sender side
        logger.debug("Pre-computing layer keys for all layers...")
        layers_data = self._group_keys_by_layers_first(tokens, mask)

        if not layers_data:
            logger.warning("No layer data to retrieve")
            return {}

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

        logger.debug(f"Pre-computed keys for {len(layers_data)} layers, "
                     f"will retrieve {len(layer_ids)} layers")

        # Retrieve layers progressively using pre-computed data
        results = {}
        for layer_id in layer_ids:
            if layer_id in layers_data:
                # Use pre-computed layer data - no key computation needed
                ret_mask = self.retrieve_layer_when_ready(
                    layer_id,
                    tokens,
                    mask,
                    timeout_us,
                    layers_data=layers_data,
                    **kwargs)
                results[layer_id] = ret_mask
            else:
                logger.warning(f"No pre-computed data for layer {layer_id}")
                results[layer_id] = None

        ed = time.perf_counter()
        successful_layers = sum(1 for mask in results.values()
                                if mask is not None)
        total_tokens = sum(
            torch.sum(mask).item() for mask in results.values()
            if mask is not None)

        logger.info(
            f"Progressive layer retrieval completed: {successful_layers}/{len(layer_ids)} layers, "
            f"{total_tokens} total tokens in {(ed-st)*1000:.2f}ms")

        return results

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

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def _retrieve_layer_rdma(self, layer_id: int, layer_chunks: List[tuple],
                             memory_format: MemoryFormat,
                             **kwargs) -> Optional[torch.Tensor]:
        """
        Retrieve a single layer using RDMA zero-copy transfers.
        
        Uses CombinedLayerCacheEngineKey to extract chunk mappings directly from the key.
        This enables true disaggregated inference where sender and receiver don't share
        storage state - chunk mappings travel with the data over NixL protocol.
        
        Args:
            layer_id: The layer ID to retrieve
            layer_chunks: List of (start, end, layer_key) tuples for this layer
            memory_format: Memory format to use (KV_T2D or KV_2LTD)
            **kwargs: Additional arguments for GPU connector
            
        Returns:
            Boolean mask indicating which tokens were retrieved, or None if no data
        """
        # Classify chunks by storage type (combined vs individual)
        individual_chunks: list[tuple[int, int,
                                      CombinedLayerCacheEngineKey]] = []

        for start, end, layer_key in layer_chunks:
            # Option 1: All chunks are processed as individual chunks
            # The storage backend automatically handles chunk mapping extraction
            # from combined objects when get_blocking() is called
            individual_chunks.append((start, end, layer_key))

        # Prepare return mask
        tokens_length = max(
            end for _, end, _ in layer_chunks) if layer_chunks else 0
        ret_mask = torch.zeros(tokens_length, dtype=torch.bool, device="cpu")

        # Process all chunks (Option 1: storage backend handles combined object extraction)
        if individual_chunks:
            memory_objs = []
            successful_ranges = []

            for start, end, layer_key in individual_chunks:
                # The storage backend's get_blocking() will:
                # 1. Check for direct individual memory objects, OR
                # 2. Use chunk mappings to extract from combined objects automatically
                memory_obj = self.storage_manager.get(layer_key)
                memory_objs.append(memory_obj)
                successful_ranges.append((start, end))

            if memory_objs:
                starts = [start for start, end in successful_ranges]
                ends = [end for start, end in successful_ranges]

                if isinstance(self.gpu_connector,
                              VLLMPagedMemLayerwiseGPUConnector):
                    self.gpu_connector.to_gpu_single_layer(
                        memory_objs, starts, ends, layer_id, **kwargs)
                else:
                    raise NotImplementedError("batched to gpu not implemented")

                # Mark tokens as retrieved
                for start, end in successful_ranges:
                    ret_mask[start:end] = True

                # Cleanup memory objects
                for memory_obj in memory_objs:
                    memory_obj.ref_count_down()

        return ret_mask if torch.sum(ret_mask) > 0 else None

    # ==============================
    # Special Tensor Methods Implementation (for logits and high-priority transfers)
    # ==============================

    def store_special_tensor(self, key: str, tensor: torch.Tensor) -> None:
        """
        Store small tensors (like logits) with high priority.
        
        This method uses the Nixl backend to transfer special tensors with high priority,
        preempting regular KV transfers. The actual transfer is handled by the storage
        backend to ensure proper coordination between prefiller and decoder.
        
        Args:
            key: Unique identifier for the tensor (should use token hash instead of req_id)
            tensor: The tensor to store (e.g., logits)
        """
        logger.debug(f"Storing special tensor with key: {key}")
        
        # Leverage the Nixl backend to transfer the tensor with high priority
        # Create a special key for the tensor
        special_key = CacheEngineKey("special", key)
        
        # Import required modules
        from lmcache.utils import SpecialTensorMetadata
        
        # Create special tensor metadata with high priority
        special_metadata = SpecialTensorMetadata(
            key=key,
            shape=tensor.shape,
            dtype=tensor.dtype,
            priority=100  # High priority to preempt regular KV transfers
        )
        
        # Store the special tensor with high priority using the storage manager
        try:
            self.storage_manager.store_special_tensor_with_priority(
                special_key, tensor, priority=100)
            logger.debug(f"Submitted special tensor for transfer with key: {key}")
        except Exception as e:
            logger.warning(f"Failed to submit special tensor for transfer: {e}")
            
        logger.debug(f"Stored special tensor with key: {key}, size: {tensor.shape}")

    def retrieve_special_tensor(self, key: str) -> Optional[torch.Tensor]:
        """
        Retrieve special tensor from storage backend immediately.
        
        This preempts other transfers and returns immediately from the storage backend
        which should have high-priority handling for special tensors.
        
        Args:
            key: Unique identifier for the tensor
            
        Returns:
            The tensor if available, None otherwise
        """
        logger.debug(f"Retrieving special tensor with key: {key}")
        
        # Check storage backend for the special tensor
        special_key = CacheEngineKey("special", key)
        
        try:
            memory_obj = self.storage_manager.get(special_key)
            if memory_obj is not None:
                tensor = memory_obj.tensor
                logger.debug(f"Retrieved special tensor from storage backend with key: {key}, size: {tensor.shape}")
                # Decrease reference count since we're done with this object
                memory_obj.ref_count_down()
                return tensor
        except Exception as e:
            logger.warning(f"Failed to retrieve special tensor from storage backend: {e}")
            
        logger.debug(f"Special tensor with key {key} not found")
        return None

    def check_special_tensor_available(self, key: str) -> bool:
        """
        Check availability of special tensor without consuming it.
        
        This checks the storage backend for the special tensor.
        
        Args:
            key: Unique identifier for the tensor
            
        Returns:
            True if tensor is available, False otherwise
        """
        logger.debug(f"Checking availability of special tensor with key: {key}")
        
        # Check storage backend for the special tensor
        special_key = CacheEngineKey("special", key)
        
        try:
            available = self.storage_manager.contains(special_key)
            if available:
                logger.debug(f"Special tensor with key {key} available in storage backend")
                return True
        except Exception as e:
            logger.warning(f"Failed to check special tensor in storage backend: {e}")
            
        logger.debug(f"Special tensor with key {key} not available")
        return False


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
        use_layeraware: bool = False,
    ) -> TokenDatabase:
        if use_layeraware:
            return LayerFirstTokenDataBase(config, metadata)
        return ChunkedTokenDatabase(config, metadata)

    @classmethod
    def get_or_create(
        cls,
        instance_id: str,
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
        gpu_connector: GPUConnectorInterface,
        use_layerwise_engine: bool = False,
        use_layeraware_engine: bool = False,
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
            token_database = cls._Create_token_database(config, metadata, use_layeraware_engine)
            stat_logger = LMCacheStatsLogger(metadata, log_interval=10)

            # HACK(Jiayi): Merge two types of engine into one in the future
            engine: Union[LayerwiseLMCacheEngine, LMCacheEngine]
            if use_layerwise_engine:
                engine = LayerwiseLMCacheEngine(config, metadata,
                                                memory_allocator,
                                                token_database, gpu_connector)
            elif use_layeraware_engine:
                engine = LayerAwareLMCacheEngine(config, metadata,
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
