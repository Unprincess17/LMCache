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
from typing import Optional, Dict, List, Callable, Tuple

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
from lmcache.utils import CacheEngineKey, LayerCacheEngineKey, CombinedLayerCacheEngineKey, NVTXContext, _lmcache_nvtx_annotate

logger = init_logger(__name__)


class RecvObjPool:

    def __init__(self, enable_gc: bool):
        self.lock = threading.Lock()
        self._data: dict[CacheEngineKey, MemoryObj] = {}
        self._cnt: dict[CacheEngineKey, int] = {}

        # OPTIMIZATION: Use deque for better performance in append/popleft operations  
        from collections import deque
        self._recent_added_keys: deque[CacheEngineKey] = deque(maxlen=80)  # Auto-limits size
        self._recent_add_threshold = 80  # Keep recent 80 keys
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
        # OPTIMIZATION: Convert deque to set for efficient difference operation
        recent_keys = set(self._recent_added_keys)
        keys_to_evict = current_keys - recent_keys
        for key in keys_to_evict:
            freed_size += self._data[key].get_size()
            self._data.pop(key)
            self._cnt.pop(key)
        ed = time.perf_counter()
        logger.warning("GC in %.4f msec, released %.2f GB memory",
                       (ed - st) * 1000, freed_size / 1024 / 1024 / 1024)

    @_lmcache_nvtx_annotate
    def add(self, key: CacheEngineKey, obj: MemoryObj):
        with self.lock:
            # OPTIMIZATION: deque.append is O(1) and auto-limits size
            self._recent_added_keys.append(key)

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

    @_lmcache_nvtx_annotate
    def batched_add(self, keys: List[CacheEngineKey], objs: List[MemoryObj]):
        """
        Batched version of add() to reduce lock acquisition overhead and
        optimize list management for better performance.
        
        Args:
            keys: List of cache engine keys
            objs: List of corresponding memory objects
        """
        if not keys:
            return
            
        with self.lock:
            # OPTIMIZATION: Batch extend is more efficient than individual appends
            # deque handles size limiting automatically
            self._recent_added_keys.extend(keys)

            # OPTIMIZATION: Reduce dictionary lookup overhead with batch operations
            data_dict = self._data
            cnt_dict = self._cnt
            
            # Batch update data and count dictionaries
            for key, obj in zip(keys, objs, strict=True):
                if key in data_dict:
                    cnt_dict[key] += 1
                    self._dbg_shallow_add += 1
                else:
                    data_dict[key] = obj
                    cnt_dict[key] = 1
                    self._dbg_deep_add += 1

            # Single debug report for the entire batch
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

    def __init__(self, num_layers: int, obj_pool=None, storage_backend=None):
        self.num_layers = num_layers
        self.obj_pool = obj_pool  # Optional object pool for storing received data
        self.storage_backend = storage_backend  # Reference to storage backend for chunk mapping storage

        # Track layer readiness and statistics
        self._layer_ready_flags = [False] * num_layers
        self._layer_chunk_counts = [0] * num_layers
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
        
        # OPTIMIZATION: Cache for parsed LayerCacheEngineKey objects to avoid repeated string parsing
        self._key_cache: Dict[str, LayerCacheEngineKey] = {}
        self._key_cache_lock = threading.RLock()

    @_lmcache_nvtx_annotate
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
            # Store in object pool if provided and determine the object to use for chunk extraction
            stored_obj = obj  # Default to original object
            if self.obj_pool is not None:
                if is_view:
                    # Clone the tensor since it's a view that will be overwritten
                    assert obj.tensor is not None, \
                            "The tensor in the MemoryObj is None."
                    copied_obj = TensorMemoryObj(obj.tensor.clone(),
                                                 obj.metadata)
                    if not isinstance(key, CombinedLayerCacheEngineKey):
                        self.obj_pool.add(key, copied_obj)
                    stored_obj = copied_obj  # Use the cloned object for chunk extraction
                else:
                    if not isinstance(key, CombinedLayerCacheEngineKey):
                        self.obj_pool.add(key, obj)
                    stored_obj = obj  # Use the original object

            # Extract layer information if this is a layer-aware key
            if isinstance(key, CombinedLayerCacheEngineKey):
                # enables individual keys to find their combined objects
                layer_id = key.layer_id
                self._store_individual_chunks_from_combined_key(key, stored_obj)

                # Mark layer as ready if expected
                if layer_id < self.num_layers:
                    if not self._layer_ready_flags[layer_id]:
                        self._mark_layer_ready(layer_id)
                        newly_ready_layers.add(layer_id)

            elif isinstance(key, LayerCacheEngineKey):
                layer_id = key.layer_id
                if layer_id < self.num_layers:
                    # Update chunk count for this layer
                    self._layer_chunk_counts[layer_id] += key.chunk_num

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

    def _get_cached_key(self, chunk_key_str: str) -> LayerCacheEngineKey:
        """
        Get a cached LayerCacheEngineKey or parse and cache it if not present.
        This reduces repeated string parsing overhead for the same chunk keys.
        
        Args:
            chunk_key_str: String representation of the chunk key
            
        Returns:
            Parsed LayerCacheEngineKey object
        """
        with self._key_cache_lock:
            if chunk_key_str not in self._key_cache:
                self._key_cache[chunk_key_str] = LayerCacheEngineKey.from_string(chunk_key_str)
                # Limit cache size to prevent memory growth
                if len(self._key_cache) > 1000:  # Reasonable limit
                    # Remove oldest 20% of entries (simple LRU approximation)
                    items_to_remove = list(self._key_cache.keys())[:200]
                    for key in items_to_remove:
                        del self._key_cache[key]
            
            return self._key_cache[chunk_key_str]

    @_lmcache_nvtx_annotate
    def _store_individual_chunks_from_combined_key(
            self, combined_key: CombinedLayerCacheEngineKey, combined_obj: MemoryObj):
        """
        Extract individual chunk keys and objects from a CombinedLayerCacheEngineKey 
        and store them directly in the obj_pool.
        
        This is called when receiving a combined key to enable individual chunk keys
        to find their objects directly. Critical for Option 1 disaggregated inference.
        
        Args:
            combined_key: The CombinedLayerCacheEngineKey with embedded chunk mappings
            combined_obj: The combined memory object containing all chunks
        """
        assert hasattr(combined_key, 'chunk_mappings') and \
                combined_key.chunk_mappings, \
                "CombinedLayerCacheEngineKey has no chunk mappings"
        assert self.obj_pool is not None
        
        # Pre-calculate common properties once for performance optimization
        with NVTXContext("Pre-calculate common properties"):
            base_format = combined_obj.get_memory_format()
            parent_allocator = getattr(combined_obj, 'parent_allocator', None)
            combined_tensor = combined_obj.tensor
        
        with NVTXContext("Batch parse chunk keys"):
            individual_keys = []
            for chunk_mapping in combined_key.chunk_mappings:
                individual_key = self._get_cached_key(chunk_mapping.chunk_key)
                individual_keys.append(individual_key)
        
        with NVTXContext("Pre-allocate batch data"):
            chunk_objects = []
            chunk_keys_for_batch = []
            
        with NVTXContext("Pre-calculate slice positions"):
            slice_positions = [(mapping.offset_start, mapping.offset_end) 
                             for mapping in combined_key.chunk_mappings]
            
        # Extract each chunk mapping and prepare objects for batched storage
        for i, (slice_pos, individual_key) in enumerate(zip(slice_positions, individual_keys)):
            try:
                with NVTXContext(f"Process chunk {i}"):
                    # Extract the specific chunk from the combined memory object
                    with NVTXContext("Extract chunk"):
                        chunk_tensor = combined_tensor[slice_pos[0]:slice_pos[1]]

                    with NVTXContext("Create metadata"):
                        chunk_metadata = MemoryObjMetadata(
                            shape=chunk_tensor.shape,
                            dtype=chunk_tensor.dtype,
                            address=chunk_tensor.data_ptr(),
                            phy_size=chunk_tensor.numel() * chunk_tensor.element_size(),
                            ref_count=1,  # Start with ref count 1
                            is_pin=False,
                            fmt=base_format)  # Use pre-calculated format

                    with NVTXContext("Create TensorMemoryObj"):
                        chunk_obj = TensorMemoryObj(
                            raw_data=chunk_tensor,
                            metadata=chunk_metadata,
                            parent_allocator=parent_allocator)

                    # Collect for batched storage instead of immediate obj_pool.add
                    chunk_objects.append(chunk_obj)
                    chunk_keys_for_batch.append(individual_key)

            except Exception as e:
                logger.error(
                    f"Failed to process chunk mapping for chunk {i}: {e}"
                )
                continue
        
        with NVTXContext("Batch add to obj_pool"):
            self.obj_pool.batched_add(chunk_keys_for_batch, chunk_objects)

    def _is_layer_complete(self, key: LayerCacheEngineKey, obj: MemoryObj,
                           layer_id: int) -> bool:
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
        # Check if key contains chunk count information
        if hasattr(key, 'total_chunks'):
            is_complete = self._layer_chunk_counts[
                layer_id] >= key.total_chunks
            if is_complete:
                logger.debug(
                    f"Layer {layer_id} marked complete via key info ({self._layer_chunk_counts[layer_id]}/{key.total_chunks} chunks)"
                )
            return is_complete

        return False

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

    def is_layer_ready(self, layer_id: int, num_chunks: int = 1) -> bool:
        """Check if layer is ready."""
        if layer_id >= self.num_layers:
            return False
        return self._layer_chunk_counts[
            layer_id] >= num_chunks or self._layer_ready_flags[layer_id]

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def wait_for_layer_busy(self,
                            layer_id: int,
                            timeout_us: int = 1000,
                            num_chunks: int = 1) -> bool:
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
        ready_layers: list[int] = []

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
        self._nixl_observer: NixlObserverInterface
        self._nixl_channel = NixlChannel(nixl_config)

        with NVTXContext("Create Nixl Observer"):
            if nixl_config.role == NixlRole.RECEIVER:
                if num_layers is not None:
                    # Use LayerAwareNixlObserver for layer tracking
                    self._nixl_observer = LayerAwareNixlObserver(
                        num_layers, self._obj_pool, storage_backend=self)
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

        # Add support for combined memory objects
        # Maps individual chunk keys to (combined_key, offset_start, offset_end)
        self._chunk_to_combined_mapping: Dict[CacheEngineKey,
                                              Tuple[CacheEngineKey, int,
                                                    int]] = {}
        self._combined_mapping_lock = threading.Lock()

    @_lmcache_nvtx_annotate
    def store_chunk_mapping(self, chunk_key: CacheEngineKey,
                            combined_key: CacheEngineKey, offset_start: int,
                            offset_end: int) -> None:
        """
        Store mapping from individual chunk key to combined memory object.
        
        Used in Option 1 approach when receiving CombinedLayerCacheEngineKey objects.
        Enables individual chunk keys to find their combined objects for retrieval.
        
        Args:
            chunk_key: Individual chunk key
            combined_key: Combined key containing this chunk
            offset_start: Start offset of chunk within combined object
            offset_end: End offset of chunk within combined object
        """
        with self._combined_mapping_lock:
            self._chunk_to_combined_mapping[chunk_key] = (combined_key,
                                                          offset_start,
                                                          offset_end)

    @_lmcache_nvtx_annotate
    def get_chunk_mapping(
        self, chunk_key: CacheEngineKey
    ) -> Optional[Tuple[CacheEngineKey, int, int]]:
        """
        Get mapping from individual chunk key to combined memory object.
        
        Args:
            chunk_key: Individual chunk key
            
        Returns:
            Tuple of (combined_key, offset_start, offset_end) if mapping exists, None otherwise
        """
        with self._combined_mapping_lock:
            return self._chunk_to_combined_mapping.get(chunk_key)

    def remove_chunk_mapping(self, chunk_key: CacheEngineKey) -> None:
        """
        Remove mapping for individual chunk key.
        
        Args:
            chunk_key: Individual chunk key to remove mapping for
        """
        with self._combined_mapping_lock:
            self._chunk_to_combined_mapping.pop(chunk_key, None)

    # TODO(Jiayi): handle `pin` smantics
    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        """
        Check whether key is in the storage backend.
        Now supports checking for keys in combined memory objects.
        
        :param key: The key to check
        :param pin: Whether to pin the object in the backend.
        
        :return: True if the key exists, False otherwise
        """
        # First check if this key maps to a combined memory object
        chunk_mapping = self.get_chunk_mapping(key)
        if chunk_mapping is not None:
            combined_key, _, _ = chunk_mapping
            # Check if the combined memory object exists
            if self._obj_pool.contains(combined_key):
                if pin:
                    self._obj_pool.pin(combined_key)
                return True

        # Check for the key directly (original logic)
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
        
        Supports both Option 1 (CombinedObjectReference) and old approach (chunk mapping storage).
        Option 1 is preferred for disaggregated inference scenarios.
        
        :param key: The key of the MemoryObj.
        
        :return: MemoryObj. None if the key does not exist.
        """
        # Option 1 approach: Check for direct key first (could be reference or actual object)
        direct_result = self._obj_pool.get(key)
        if direct_result is not None:
            # For Option 1: This could be a CombinedObjectReference or actual MemoryObj
            # The cache engine will handle CombinedObjectReference objects appropriately
            return direct_result

        # Fallback to old approach: Check if this key maps to a combined memory object
        # This is for backward compatibility with engines not using Option 1
        chunk_mapping = self.get_chunk_mapping(key)
        if chunk_mapping is not None:
            combined_key, offset_start, offset_end = chunk_mapping

            # Get the combined memory object
            combined_memory_obj = self._obj_pool.get(combined_key)
            if combined_memory_obj is not None:
                # Extract the specific chunk from the combined memory object
                return self._extract_chunk_from_combined_memory(
                    combined_memory_obj, offset_start, offset_end)

        # Key not found
        return None

    def _extract_chunk_from_combined_memory(self,
                                            combined_memory_obj: MemoryObj,
                                            offset_start: int,
                                            offset_end: int) -> MemoryObj:
        """
        Extract a specific chunk from a combined memory object.
        
        Args:
            combined_memory_obj: The combined memory object
            offset_start: Start offset for the chunk
            offset_end: End offset for the chunk
            
        Returns:
            TensorMemoryObj representing the extracted chunk
        """
        # Extract the chunk tensor from the combined memory object
        # For KV_T2D format: [total_tokens, 2, hidden_dim] -> [chunk_tokens, 2, hidden_dim]
        assert combined_memory_obj.tensor is not None

        chunk_tensor = combined_memory_obj.tensor[offset_start:offset_end]

        # Create metadata for the extracted chunk
        chunk_metadata = MemoryObjMetadata(
            shape=chunk_tensor.shape,
            dtype=chunk_tensor.dtype,
            address=chunk_tensor.data_ptr(),
            phy_size=chunk_tensor.numel() * chunk_tensor.element_size(),
            ref_count=1,  # Start with ref count 1
            is_pin=False,
            fmt=combined_memory_obj.get_memory_format())

        # Create a new TensorMemoryObj for the extracted chunk
        # Note: This creates a view of the original tensor, not a copy
        chunk_memory_obj = TensorMemoryObj(
            raw_data=chunk_tensor,
            metadata=chunk_metadata,
            parent_allocator=combined_memory_obj.parent_allocator if hasattr(
                combined_memory_obj, 'parent_allocator') else None)

        return chunk_memory_obj

    def get_non_blocking(
        self,
        key: CacheEngineKey,
    ) -> Optional[Future]:
        """
        Non-blocking function to get the memory object from the storage backend.
        Now supports retrieving from combined memory objects.
        
        :param key: The key of the MemoryObj to retrieve.
        :return: A Future object that will resolve to the MemoryObj, or None if the key doesn't exist.
        """
        # First check if this key maps to a combined memory object
        chunk_mapping = self.get_chunk_mapping(key)
        if chunk_mapping is not None:
            combined_key, offset_start, offset_end = chunk_mapping

            # Check if the combined memory object exists
            if not self._obj_pool.contains(combined_key):
                return None

            # Create a Future object for the combined memory retrieval
            future: Future[Optional[MemoryObj]] = Future()

            def get_and_extract_chunk():
                try:
                    # Get the combined memory object
                    combined_memory_obj = self._obj_pool.get(combined_key)
                    if combined_memory_obj is not None:
                        # Extract the specific chunk
                        chunk_memory_obj = self._extract_chunk_from_combined_memory(
                            combined_memory_obj, offset_start, offset_end)
                        future.set_result(chunk_memory_obj)
                    else:
                        future.set_result(None)
                except Exception as e:
                    future.set_exception(e)

            # Start a new thread to perform the get and extract operation
            thread = threading.Thread(target=get_and_extract_chunk)
            thread.start()

            return future

        # Check for the key directly (original logic)
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
