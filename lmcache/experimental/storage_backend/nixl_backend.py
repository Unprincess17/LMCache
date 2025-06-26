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

import threading
import time
from concurrent.futures import Future
from typing import Optional, Dict, List, Callable

import torch

from lmcache.config import LMCacheEngineMetadata
from lmcache.experimental.config import LMCacheEngineConfig
from lmcache.experimental.memory_management import (MemoryAllocatorInterface,
                                                    MemoryFormat, MemoryObj,
                                                    MemoryObjMetadata,
                                                    TensorMemoryObj)
from lmcache.experimental.storage_backend.abstract_backend import \
    StorageBackendInterface
from lmcache.experimental.storage_backend.connector.nixl_connector_v2 import (
    NixlChannel, NixlObserverInterface)
from lmcache.experimental.storage_backend.connector.nixl_utils import (
    NixlConfig, NixlRole)
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey, LayerCacheEngineKey, NVTXContext, _lmcache_nvtx_annotate

logger = init_logger(__name__)


class RecvObjPool:

    def __init__(self, enable_gc: bool):
        self.lock = threading.Lock()
        self._data: dict[CacheEngineKey, MemoryObj] = {}
        self._cnt: dict[CacheEngineKey, int] = {}

        # TODO: Remove the hard-code
        # HACK: have a recycle threshold to avoid the memory leak
        self._recent_added_keys: list[CacheEngineKey] = []
        self._recent_add_threshold = 80  # Keep recent 90 keys
        self._recycle_threshold = 160

        self._enable_gc = enable_gc
        if not self._enable_gc:
            logger.warning("GC for receiver is disabled, may lead to memory "
                           "leak in non-testing environment")

        # Debug information
        self._dbg_shallow_add = 0
        self._dbg_deep_add = 0
        self._dbg_shallow_remove = 0
        self._dbg_deep_remove = 0
        self._dbg_num_get = 0
        self._dbg_num_success_get = 0
        self._dbg_num_contains = 0
        self._dbg_num_success_contains = 0
        self._dbg_num_gc = 0
        self._dbg_last_report_time = time.time()

    def dbg_report(self):
        return  # Disable debug report for now

        curr_time = time.time()
        if curr_time - self._dbg_last_report_time < 5:
            return
        self._dbg_last_report_time = curr_time

        logger.warning("RecvObjPool Debug Info:")
        logger.warning("  - New add: %d", self._dbg_deep_add)
        logger.warning("  - Redundant add: %d", self._dbg_shallow_add)
        logger.warning("  - Shallow remove: %d", self._dbg_shallow_remove)
        logger.warning("  - Deep remove: %d", self._dbg_deep_remove)
        logger.warning("  - Num get: %d", self._dbg_num_get)
        logger.warning("  - Num success get: %d", self._dbg_num_success_get)
        logger.warning("  - Num contains: %d", self._dbg_num_contains)
        logger.warning("  - Num success contains: %d",
                       self._dbg_num_success_contains)
        logger.warning("  - Current num_objs: %d", len(self._data))
        tot_size = sum([self._data[key].get_size() for key in self._data])
        logger.warning("  - Total size: %.2f GB",
                       tot_size / 1024 / 1024 / 1024)
        logger.warning("  - Number of GC: %d", self._dbg_num_gc)

    def _gc(self):
        if not self._enable_gc:
            return

        logger.warning("In GC!")
        self._dbg_num_gc += 1
        st = time.perf_counter()
        freed_size = 0
        current_keys = set(self._data.keys())
        recent_keys = set(self._recent_added_keys)
        keys_to_evict = current_keys - recent_keys
        for key in keys_to_evict:
            freed_size += self._data[key].get_size()
            self._data.pop(key)
            self._cnt.pop(key)
        ed = time.perf_counter()
        logger.warning("GC in %.4f msec, released %.2f GB memory",
                       (ed - st) * 1000, freed_size / 1024 / 1024 / 1024)

    def add(self, key: CacheEngineKey, obj: MemoryObj):
        with self.lock:
            # TODO: Get rid of this
            self._recent_added_keys.append(key)
            self._recent_added_keys = \
                    self._recent_added_keys[-self._recent_add_threshold:]

            if key in self._data:
                self._cnt[key] += 1

                # DEBUG
                self._dbg_shallow_add += 1
            else:
                self._data[key] = obj
                self._cnt[key] = 1

                # DEBUG
                self._dbg_deep_add += 1

            # DEBUG
            self.dbg_report()

    def remove(self, key: CacheEngineKey):
        with self.lock:
            if key in self._cnt:
                self._cnt[key] -= 1
                if self._cnt[key] == 0:
                    self._data.pop(key)
                    self._cnt.pop(key)

                    # DEBUG
                    self._dbg_deep_remove += 1
                else:
                    # DEBUG
                    self._dbg_shallow_remove += 1

            self.dbg_report()

    def contains(self, key: CacheEngineKey) -> bool:
        with self.lock:
            if len(self._data) >= self._recycle_threshold:
                self._gc()

            # DEBUG
            ret = key in self._data
            self._dbg_num_contains += 1
            if ret:
                self._dbg_num_success_contains += 1
            self.dbg_report()

            return ret

    def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        with self.lock:
            # DEBUG
            ret = self._data.get(key, None)
            self._dbg_num_get += 1
            if ret is not None:
                self._dbg_num_success_get += 1
            self.dbg_report()

            return ret

    def pin(self, key: CacheEngineKey) -> bool:
        raise NotImplementedError

    def unpin(self, key: CacheEngineKey) -> bool:
        raise NotImplementedError


class LayerAwareNixlObserver(NixlObserverInterface):
    """
    NixlObserver implementation that tracks layer transfer status for disaggregated inference.
    
    This observer combines data reception handling with layer readiness tracking.
    Layers are marked as ready only when the data is actually received through RDMA transfer.
    """

    def __init__(self, num_layers: int, obj_pool=None):
        self.num_layers = num_layers
        self.obj_pool = obj_pool  # Optional object pool for storing received data

        # Track layer readiness and statistics
        self._layer_ready_flags = [False] * num_layers
        self._layer_chunk_counts = [0] * num_layers
        self._layer_expected_chunks = [None] * num_layers  # None = unknown, set dynamically
        self._layer_ready_timestamps = [0.0] * num_layers

        # Callbacks for layer readiness events
        self._callbacks: Dict[int, List[Callable]] = {
            i: []
            for i in range(num_layers)
        }
        self._callback_lock = threading.RLock()

        # Statistics
        self._total_layers_transferred = 0
        self._transfer_start_time = time.perf_counter()

    def __call__(self,
                 keys: List[CacheEngineKey],
                 objs: List[MemoryObj],
                 is_view: bool = True):
        """
        Process received objects and update layer readiness status.
        
        Args:
            keys: The CacheEngineKeys
            objs: The list of MemoryObj
            is_view: Whether the memory objects are views of the transfer buffer
        """

        # Track layers that become ready in this batch
        newly_ready_layers = set()

        # Process each received object
        for key, obj in zip(keys, objs):
            # Store in object pool if provided
            if self.obj_pool is not None:
                if is_view:
                    # Clone the tensor since it's a view that will be overwritten
                    copied_obj = TensorMemoryObj(obj.tensor.clone(),
                                                 obj.metadata)
                    self.obj_pool.add(key, copied_obj)
                else:
                    self.obj_pool.add(key, obj)

            # Extract layer information if this is a layer-aware key
            if isinstance(key, LayerCacheEngineKey):
                layer_id = key.layer_id
                if layer_id < self.num_layers:
                    # Update chunk count for this layer
                    self._layer_chunk_counts[layer_id] += 1

                    # Check if this layer is now complete
                    if self._is_layer_complete(key, obj, layer_id):
                        if not self._layer_ready_flags[layer_id]:
                            self._mark_layer_ready(layer_id)
                            newly_ready_layers.add(layer_id)

        # Fire callbacks for newly ready layers
        for layer_id in newly_ready_layers:
            self._fire_callbacks_async(layer_id)

        logger.debug(f"NixlObserver processed {len(keys)} objects, "
                     f"{len(newly_ready_layers)} layers became ready")

    def _is_layer_complete(self, key: LayerCacheEngineKey, obj: MemoryObj, layer_id: int) -> bool:
        """
        Determine if a layer is complete based on received data.
        
        Uses multiple strategies to detect layer completion:
        - Key-based: Check for total_chunks/chunk_id information
        
        Args:
            key: The layer cache engine key
            obj: The memory object received
            layer_id: The layer ID being processed
            
        Returns:
            True if the layer is complete, False otherwise
        """
        # Strategy 2: Check if key contains chunk count information  
        if hasattr(key, 'total_chunks'):
            if self._layer_expected_chunks[layer_id] is None:
                self._layer_expected_chunks[layer_id] = key.total_chunks
                logger.debug(f"Layer {layer_id} expected chunks set to {key.total_chunks}")
            
            is_complete = self._layer_chunk_counts[layer_id] >= self._layer_expected_chunks[layer_id]
            if is_complete:
                logger.debug(f"Layer {layer_id} marked complete via key info ({self._layer_chunk_counts[layer_id]}/{self._layer_expected_chunks[layer_id]} chunks)")
            return is_complete
            
        return False

    def set_expected_chunks_per_layer(self, layer_id: int, expected_chunks: int) -> None:
        """
        Manually set the expected number of chunks for a specific layer.
        
        This provides backward compatibility and allows explicit control when needed.
        
        Args:
            layer_id: The layer ID
            expected_chunks: Expected number of chunks for this layer
        """
        if layer_id < self.num_layers:
            self._layer_expected_chunks[layer_id] = expected_chunks
            logger.debug(f"Layer {layer_id} expected chunks manually set to {expected_chunks}")

    def _mark_layer_ready(self, layer_id: int) -> None:
        """Mark a layer as ready (internal method)."""
        self._layer_ready_timestamps[layer_id] = time.perf_counter()
        self._layer_ready_flags[layer_id] = True
        self._total_layers_transferred += 1

        logger.debug(
            f"Layer {layer_id} marked ready with {self._layer_chunk_counts[layer_id]} chunks"
        )

    def mark_layer_ready(self, layer_id: int, num_chunks: int) -> None:
        """
        Manually mark layer as ready (compatibility method).
        
        This method provides compatibility with existing code that manually signals layer readiness.
        In normal RDMA operation, layers are marked ready automatically via the __call__ method.
        
        Args:
            layer_id: The layer that is now ready
            num_chunks: Number of chunks transferred for this layer
        """
        if layer_id >= self.num_layers:
            logger.warning(
                f"Invalid layer_id {layer_id}, max is {self.num_layers-1}")
            return

        # Update chunk count if provided
        if num_chunks > 0:
            self._layer_chunk_counts[layer_id] = num_chunks

        # Mark layer as ready if not already marked
        if not self._layer_ready_flags[layer_id]:
            self._mark_layer_ready(layer_id)
            self._fire_callbacks_async(layer_id)

    def is_layer_ready(self, layer_id: int, num_chunks: Optional[int] = 1) -> bool:
        """Check if layer is ready."""
        if layer_id >= self.num_layers:
            return False
        return self._layer_chunk_counts[layer_id] >= num_chunks

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def wait_for_layer_busy(self,
                            layer_id: int,
                            timeout_us: int = 1000,
                            num_chunks: Optional[int] = 1) -> bool:
        """
        Busy wait for layer readiness with microsecond precision.
        
        Args:
            layer_id: Layer to wait for
            timeout_us: Timeout in microseconds
            
        Returns:
            True if layer became ready within timeout, False otherwise
        """
        if layer_id >= self.num_layers:
            return False

        start = time.perf_counter()
        while not self.is_layer_ready(layer_id, num_chunks):
            elapsed_us = (time.perf_counter() - start) * 1e6
            if elapsed_us > timeout_us:
                return False
            time.sleep(0)  # Yield
        return True

    def wait_for_layers_batch(self,
                              layer_ids: List[int],
                              timeout_us: int = 5000) -> List[int]:
        """
        Wait for multiple layers and return those that become ready.
        
        Args:
            layer_ids: List of layer IDs to wait for
            timeout_us: Total timeout in microseconds
            
        Returns:
            List of layer IDs that became ready within timeout
        """
        start = time.perf_counter()
        ready_layers = []

        while len(ready_layers) < len(layer_ids):
            elapsed_us = (time.perf_counter() - start) * 1e6
            if elapsed_us > timeout_us:
                break

            for layer_id in layer_ids:
                if layer_id not in ready_layers and self.is_layer_ready(
                        layer_id):
                    ready_layers.append(layer_id)

            if len(ready_layers) < len(layer_ids):
                time.sleep(0)  # Yield

        return sorted(ready_layers)

    def reset(self) -> None:
        """Reset all tracking state for a new transfer."""
        self._layer_ready_flags = [False] * self.num_layers
        self._layer_chunk_counts = [0] * self.num_layers
        self._layer_ready_timestamps = [0.0] * self.num_layers
        self._total_layers_transferred = 0
        self._transfer_start_time = time.perf_counter()

        # Clear callbacks
        with self._callback_lock:
            for layer_callbacks in self._callbacks.values():
                layer_callbacks.clear()

    def _fire_callbacks_async(self, layer_id: int) -> None:
        """Fire callbacks in separate thread to avoid blocking critical path."""

        def fire():
            with self._callback_lock:
                for callback in self._callbacks[layer_id]:
                    try:
                        callback(layer_id)
                    except Exception as e:
                        logger.error(
                            f"Callback error for layer {layer_id}: {e}")

        threading.Thread(target=fire, daemon=True).start()


class BasicNixlObserver(NixlObserverInterface):
    """
    Basic implementation of the NixlObserverInterface to handle 
    events from NixlChannel.
    """

    def __init__(self, obj_pool: RecvObjPool):
        """
        Initialize the BasicNixlObserver.
        """
        self.obj_pool = obj_pool

    @_lmcache_nvtx_annotate
    def __call__(self,
                 keys: list[CacheEngineKey],
                 objs: list[MemoryObj],
                 is_view: bool = True):
        """Blocking function to process the received objects
        
        Args:
          keys: the CacheEngineKeys
          objs: the list of MemoryObj
          is_view: whether the memory objects are the view of the underlying 
            transfer buffer  (i.e., whether it will be overwrite by next 
            transfer)
        """
        clone_time = 0.0
        add_time = 0.0
        for key, value in zip(keys, objs, strict=False):
            assert value.tensor is not None, \
                    "The tensor in the MemoryObj is None."
            if is_view:
                #self.obj_pool.add(key, value)
                st = time.perf_counter()
                copied_obj = TensorMemoryObj(value.tensor.clone(),
                                             value.metadata)
                ed = time.perf_counter()
                self.obj_pool.add(key, copied_obj)
                ed2 = time.perf_counter()
                clone_time += (ed - st) * 1000
                add_time += (ed2 - ed) * 1000
            else:
                self.obj_pool.add(key, value)
        logger.debug(
            "Nixl Observer: clone time: %.4f msec, "
            "Add time: %.4f msec for %d objects", clone_time, add_time,
            len(keys))


class NixlBackend(StorageBackendInterface):
    """
    Implementation of the StorageBackendInterface for Nixl.

    Currently, the put is synchronized and blocking, to simplify the 
    implementation.

    At the sender side, it will never save anything but directly write the data
    to the receiver side.
    """

    def __init__(self,
                 nixl_config: NixlConfig,
                 num_layers: Optional[int] = None):
        """
        Initialize the Nixl storage backend.

        :param nixl_config: the Nixl configuration
        :param num_layers: number of layers for layer-aware tracking (required for RECEIVER role)
        """
        super().__init__(dst_device=nixl_config.buffer_device)
        self._obj_pool = RecvObjPool(nixl_config.enable_gc)
        self._num_layers = num_layers

        self._nixl_channel = NixlChannel(nixl_config)

        with NVTXContext("Create Nixl Observer"):
            if nixl_config.role == NixlRole.RECEIVER:
                if num_layers is not None:
                    # Use LayerAwareNixlObserver for layer tracking
                    self._nixl_observer = LayerAwareNixlObserver(
                        num_layers, self._obj_pool)
                    logger.info(
                        f"Created LayerAwareNixlObserver with {num_layers} layers for RDMA layer tracking"
                    )
                else:
                    # Fallback to BasicNixlObserver if num_layers not provided
                    self._nixl_observer = BasicNixlObserver(self._obj_pool)
                    logger.info(
                        "Created BasicNixlObserver (num_layers not provided)")

                self._nixl_channel.register_receive_observer(
                observer=self._nixl_observer)

        self._registered_keys: list[CacheEngineKey] = []
        self._registered_metadatas: list[MemoryObjMetadata] = []
        self._num_payload_added = 0

    # TODO(Jiayi): handle `pin` smantics
    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        """
        Check whether key is in the storage backend.
        
        :param key: The key to check
        :param pin: Whether to pin the object in the backend.
        
        :return: True if the key exists, False otherwise
        """
        return self._obj_pool.contains(key)

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        """
        Check whether key is in the ongoing submit_put_task tasks.
        
        :param key: The key to check
        :return: True if the key exists in put tasks, False otherwise
        """
        return False

    def register_put_tasks(
        self,
        keys: list[CacheEngineKey],
        metadatas: list[MemoryObjMetadata],
    ) -> None:
        """
        Register the put tasks to the backend.
        """
        if len(self._registered_keys) > 0:
            raise RuntimeError("The backend has already registered put tasks.")

        self._registered_keys = keys
        self._registered_metadatas = metadatas
        self._nixl_channel.prepare_send(keys=keys, metadatas=metadatas)

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def allocate_zero_copy_write_object(
        self,
        shape: torch.Size,
        dtype: Optional[torch.dtype],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
    ) -> MemoryObj:
        """
        Allocate a zero-copy write object for the given shape and dtype.

        This will be seen as "adding a new payload" to the backend.
        """
        assert self._registered_metadatas[self._num_payload_added].shape \
            == shape, \
            "The shape of the allocated object is not equal to the shape of " \
            "the registered metadata."

        assert self._registered_metadatas[self._num_payload_added].dtype \
            == dtype, \
            "The dtype of the allocated object is not equal to the dtype of " \
            "the registered metadata."

        assert self._registered_metadatas[self._num_payload_added].fmt == fmt, \
            "The fmt of the allocated object is not equal to the fmt of " \
            "the registered metadata."

        self._num_payload_added += 1

        ret = self._nixl_channel.allocate_for_send(shape=shape,
                                                   dtype=dtype,
                                                   fmt=fmt)
        assert ret is not None, \
            "Failed to allocate zero-copy buffer from nixl_channel"
        return ret

    def flush_put_tasks(self) -> None:
        """
        Flush the registered tasks 
        """
        assert len(self._registered_keys) > 0, \
            "The backend has not registered put tasks."
        assert self._num_payload_added == len(self._registered_keys), \
            "The number of payloads added is not equal to the number of" \
            "registered keys."

        self._nixl_channel.finish_send()
        self._registered_keys = []
        self._registered_metadatas = []
        self._num_payload_added = 0

    def submit_put_task(self, key: CacheEngineKey,
                        obj: MemoryObj) -> Optional[Future]:
        """
        Put the MemoryObj into the storage backend and send it to the receiver
        in a blocking way.

        :param key: The key of the MemoryObj.
        :param obj: The MemoryObj to be stored.
        
        :return: a future object

        :note: Right now, the 'key' is not used and it assumes that the memory 
        object has the same order as the keys passed in `register_put_tasks`.
        """
        raise NotImplementedError

    def submit_prefetch_task(self, key: CacheEngineKey) -> Optional[Future]:
        """
        An async function to get the MemoryObj from the storage backend.

        :param key: The key of the MemoryObj.

        :return: a future object. None if the key does not exist.
        """
        raise NotImplementedError

    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """
        A blocking function to get the kv cache from the storage backend.
        
        :param key: The key of the MemoryObj.
        
        :return: MemoryObj. None if the key does not exist.
        """
        return self._obj_pool.get(key)

    def get_non_blocking(
        self,
        key: CacheEngineKey,
    ) -> Optional[Future]:
        """
        Non-blocking function to get the memory object from the storage backend.
        
        :param key: The key of the MemoryObj to retrieve.
        :return: A Future object that will resolve to the MemoryObj, or None if the key doesn't exist.
        """
        if not self.contains(key):
            return None

        # Create a Future object to represent the asynchronous operation
        future = Future()

        def get_and_set_result():
            try:
                # Get the memory object from the object pool
                result = self._obj_pool.get(key)
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)

        # Start a new thread to perform the get operation
        thread = threading.Thread(target=get_and_set_result)
        thread.start()

        return future

    def remove(self, key: CacheEngineKey) -> None:
        """
        Remove the key from the storage backend.

        :param key: The key to remove.
        """
        return self._obj_pool.remove(key)

    def close(self) -> None:
        """
        Close the storage backend.
        """
        self._nixl_channel.close()

    def get_underlying_allocator(self) -> MemoryAllocatorInterface:
        """
        Get the underlying allocator from Nixl channel.
        """
        return self._nixl_channel.get_allocator()

    def pin(self, key: CacheEngineKey) -> bool:
        raise NotImplementedError

    def unpin(self, key: CacheEngineKey) -> bool:
        raise NotImplementedError

    def update_put_state(self, key: CacheEngineKey,
                         memory_obj: MemoryObj) -> None:
        """
        Update the backend's internal state after a memory object has been allocated
        and written to. This is used in the zero-copy write pattern.
        
        :param key: The key associated with the memory object
        :param memory_obj: The memory object that has been written to
        """
        # For the Nixl backend, we don't need to do anything here
        # The memory is already in the right place due to zero-copy allocation
        # and the metadata was registered during prepare_put
        pass

    # Layer status interface methods - delegate to LayerAwareNixlObserver if available
    def mark_layer_ready(self, layer_id: int, num_chunks: int = 0) -> None:
        """Mark a layer as ready for processing."""
        if hasattr(self._nixl_observer, 'mark_layer_ready'):
            self._nixl_observer.mark_layer_ready(layer_id, num_chunks)
        else:
            logger.debug(
                f"mark_layer_ready called but observer doesn't support layer tracking"
            )

    def is_layer_ready(self, layer_id: int) -> bool:
        """Check if a layer is ready for processing."""
        if hasattr(self._nixl_observer, 'is_layer_ready'):
            return self._nixl_observer.is_layer_ready(layer_id)
        return False

    def wait_for_layers_batch(self,
                              layer_ids: list[int],
                              timeout_us: int = 5000) -> list[int]:
        """Wait for multiple layers to become ready."""
        if hasattr(self._nixl_observer, 'wait_for_layers_batch'):
            return self._nixl_observer.wait_for_layers_batch(
                layer_ids, timeout_us)
        return []

    def wait_for_layer_busy(self,
                            layer_id: int,
                            timeout_us: int = 1000,
                            num_chunks: Optional[int] = 1) -> bool:
        """Busy wait for a layer to become ready."""
        if hasattr(self._nixl_observer, 'wait_for_layer_busy'):
            return self._nixl_observer.wait_for_layer_busy(
                layer_id, timeout_us, num_chunks)
        return False

    def get_layer_observer(self):
        """Get the layer-aware observer if available."""
        if hasattr(self._nixl_observer, 'num_layers'):
            return self._nixl_observer
        return None

    @staticmethod
    def CreateNixlBackend(config: LMCacheEngineConfig,
                          metadata: LMCacheEngineMetadata) -> "NixlBackend":
        """
        Create a Nixl backend with the given configuration.

        :param config: The LMCache engine configuration.
        :param metadata: The LMCache engine metadata containing layer information.
        
        :return: A NixlBackend instance.
        """
        # Create the Nixl config
        nixl_config = NixlConfig.from_cache_engine_config(config, metadata)

        # Extract num_layers from metadata for layer-aware tracking
        num_layers = metadata.kv_shape[0] if metadata.kv_shape else None

        # Create the Nixl backend with layer awareness
        backend = NixlBackend(nixl_config, num_layers)
        return backend
