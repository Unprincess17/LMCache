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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional
import hashlib

import torch
import vllm.envs as envs
import zmq
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.utils import cdiv, make_zmq_socket
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

from lmcache.experimental.cache_engine import (LayerAwareLMCacheEngine, LayerwiseLMCacheEngine,
                                               LMCacheEngine)
from lmcache.experimental.storage_backend.storage_manager import DistributedStorageManager
from lmcache.integration.vllm.vllm_adapter import init_lmcache_engine
from lmcache.logging import init_logger
from lmcache.utils import _lmcache_nvtx_annotate

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.sched.output import CachedRequestData, NewRequestData
    from vllm.v1.request import Request

logger = init_logger(__name__)


def get_zmq_rpc_path_lmcache(
        role: KVConnectorRole,
        is_tp: bool = False,
        vllm_config: Optional["VllmConfig"] = None) -> str:
    base_url = envs.VLLM_RPC_BASE_PATH
    # Default to 0 if not configured
    rpc_port = 0
    if vllm_config is not None:
        rpc_port = vllm_config.kv_transfer_config.get_from_extra_config(
            "lmcache_rpc_port", 0)
    logger.debug("Base URL: %s, RPC Port: %s", base_url, rpc_port)
    return f"ipc://{base_url}/lmcache_rpc_port_{rpc_port}"


# TODO: move this to LMCache so that we can gracefully close it
class LMCacheLookupClient:

    def __init__(self, role: KVConnectorRole, is_tp: bool,
                 vllm_config: "VllmConfig"):
        self.encoder = MsgpackEncoder()
        self.ctx = zmq.Context()  # type: ignore[attr-defined]
        socket_path = get_zmq_rpc_path_lmcache(role, is_tp, vllm_config)
        self.socket = make_zmq_socket(
            self.ctx,
            socket_path,
            zmq.REQ,  # type: ignore[attr-defined]
            bind=False)

    def lookup(self, token_ids: torch.Tensor) -> int:
        request = self.encoder.encode(token_ids)
        self.socket.send_multipart(request, copy=False)
        resp = self.socket.recv()
        result = int.from_bytes(resp, "big")
        return result

    def check_special_tensor_available(self, key: str) -> bool:
        """
        Check if a special tensor (like first decode logits) is available in the cache.
        
        Args:
            key: The cache key for the special tensor
            
        Returns:
            True if the special tensor is available, False otherwise
        """
        # Send a special marker to indicate this is a special tensor check
        request = [b"SPECIAL_TENSOR_CHECK", key.encode('utf-8')]
        self.socket.send_multipart(request, copy=False)
        resp = self.socket.recv()
        result = bool(int.from_bytes(resp, "big"))
        return result

    def close(self):
        self.socket.close(linger=0)


class LMCacheLookupServer:

    def __init__(self, lmcache_engine: LMCacheEngine, role: KVConnectorRole,
                 is_tp: bool, vllm_config: "VllmConfig"):
        self.decoder = MsgpackDecoder(torch.Tensor)
        self.ctx = zmq.Context()  # type: ignore[attr-defined]
        socket_path = get_zmq_rpc_path_lmcache(role, is_tp, vllm_config)
        self.socket = make_zmq_socket(
            self.ctx,
            socket_path,
            zmq.REP,  # type: ignore[attr-defined]
            bind=True)

        self.lmcache_engine = lmcache_engine
        self.running = True

        def process_request():
            while self.running:
                #try:
                #request = self.socket.recv()
                frames = self.socket.recv_multipart(copy=False)
                
                # Check if this is a special tensor availability check
                if len(frames) == 2 and bytes(frames[0].buffer) == b"SPECIAL_TENSOR_CHECK":
                    key = bytes(frames[1].buffer).decode('utf-8')
                    result = self.lmcache_engine.check_special_tensor_available(key)
                    response = int(result).to_bytes(4, "big")
                    self.socket.send(response)
                else:
                    # Regular token lookup
                    token_ids = self.decoder.decode(frames)
                    result = self.lmcache_engine.lookup(token_ids)
                    response = result.to_bytes(4, "big")
                    self.socket.send(response)
                #except Exception as e:
                #    logger.error("Error in LMCache lookup server: %s", e)
                #    break
                #continue

        self.thread = threading.Thread(target=process_request, daemon=True)
        self.thread.start()

    def close(self):
        self.socket.close(linger=0)
        # TODO: close the thread!


@dataclass
class LoadSpec:
    # Number of tokens cached in vLLM
    vllm_cached_tokens: int
    # Number of tokens that are cached in LMCache
    lmcache_cached_tokens: int
    # Whether the scheduler allow us to load the tokens
    can_load: bool


@dataclass
class SaveSpec:
    # Skip already saved tokens
    skip_leading_tokens: int
    # Whether the scheduler allow us to save the tokens
    can_save: bool


@dataclass
class RequestTracker:
    # Request id
    req_id: str

    # The token ids that has been scheduled so far
    token_ids: list[int]

    # The block ids that has been allocated so far
    # NOTE: allocated blocks could be more than the number of tokens
    # FIXME: need to check whether the block ids will be changed after
    #        preemption
    allocated_block_ids: list[int]

    # The number of tokens that has been savd
    num_saved_tokens: int = 0

    @staticmethod
    def from_new_request(
        new_request: "NewRequestData",
        num_tokens_to_compute: int,
    ) -> "RequestTracker":
        """Create the request tracker from a new request.

        Args:
            new_request (NewRequestData): the new request data.
            num_tokens_to_compute (int): the number of tokens that will 
                be 'computed', including the `num_computed_tokens` (vLLM's
                local cache hit) and new tokens that will be scheduled.

        """
        return RequestTracker(
            req_id=new_request.req_id,
            token_ids=new_request.prompt_token_ids[:num_tokens_to_compute].
            copy(),
            allocated_block_ids=new_request.block_ids.copy(),
            num_saved_tokens=0,
        )

    def update(
        self,
        cached_request: "CachedRequestData",
    ) -> None:
        """Update the request tracker when a running request is 
        scheduled again
        """
        self.token_ids.extend(cached_request.new_token_ids)
        self.allocated_block_ids.extend(cached_request.new_block_ids)


@dataclass
class ReqMeta:
    # Request id
    req_id: str
    # Request tokens
    token_ids: torch.Tensor
    # Slot mapping
    slot_mapping: torch.Tensor
    # Skip save or not
    save_spec: Optional[SaveSpec] = None
    # load_spec
    load_spec: Optional[LoadSpec] = None

    @staticmethod
    def from_request_tracker(
        tracker: RequestTracker,
        block_size: int,
        lmcache_chunk_size: int = 256,
        load_spec: Optional[LoadSpec] = None,
        skip_save: bool = False,
        discard_partial_chunks: bool = True,
    ) -> Optional["ReqMeta"]:
        """Create the request metadata from a request tracker.

        Args:
            tracker (RequestTracker): the request tracker.
            block_size (int): the block size in vLLM.
            lmcache_chunk_size (int): the chunk size for LMCache.
            load_spec (Optional[LoadSpec]): the load spec for KV cache loading.
            skip_save (bool): whether to skip the save operation.
            discard_partial_chunks (bool): whether to discard partial chunks.

        Returns:
            the request metadata if we need to perform load/save 
            operations, None otherwise.
        """
        input_token_ids = tracker.token_ids
        input_token_len = len(input_token_ids)

        # For save operation: do not save if the following condition is met
        # 1. has already been saved before (num_saved_tokens > 0)
        # 2. number of unsaved tokens is not reached the chunk boundary
        skip_leading_tokens = tracker.num_saved_tokens
        chunk_boundary = cdiv(tracker.num_saved_tokens, lmcache_chunk_size) * \
                lmcache_chunk_size
        skip_save = skip_save or (tracker.num_saved_tokens > 0 and \
                input_token_len < chunk_boundary)

        if skip_save and load_spec is None:
            return None

        # Calculate number of tokens to save based on discard_partial_chunks
        # setting
        num_tokens_to_save = (
            input_token_len // lmcache_chunk_size *
            lmcache_chunk_size) if discard_partial_chunks else input_token_len

        # If we need to save, update the number of saved tokens
        if not skip_save:
            tracker.num_saved_tokens = num_tokens_to_save
        save_spec = SaveSpec(skip_leading_tokens, not skip_save)

        # Calculate the token ids and slot mappings for load and save
        # OPTIMIZATION: pre-allocate the buffer for token ids and block
        # ids
        token_ids = torch.tensor(input_token_ids)[:num_tokens_to_save]
        num_blocks = len(tracker.allocated_block_ids)
        block_ids = torch.tensor(tracker.allocated_block_ids, dtype=torch.long)

        if len(token_ids) > num_blocks * block_size:
            logger.error(
                "The number of tokens is more than the number of blocks."
                "Something might be wrong in scheduling logic!")
            logger.error("Num tokens: %d, num blocks: %d, block size: %d",
                         len(token_ids), num_blocks, block_size)

        block_offsets = torch.arange(0, block_size, dtype=torch.long)
        slot_mapping = block_offsets.reshape((1, block_size)) + \
                block_ids.reshape((num_blocks, 1)) * block_size

        slot_mapping = slot_mapping.flatten()[:len(token_ids)]
        assert slot_mapping.dtype == torch.long  # TODO: this could be removed

        # For load operation: check whether the request is scheduled to load
        if load_spec is not None and load_spec.can_load:
            logger.debug("Scheduled to load %d tokens for request %s",
                         load_spec.lmcache_cached_tokens, tracker.req_id)
        else:
            # Do not load if not in `can_load` state
            load_spec = None

        return ReqMeta(
            req_id=tracker.req_id,
            token_ids=token_ids,
            slot_mapping=slot_mapping,
            save_spec=save_spec,
            load_spec=load_spec,
        )


@dataclass
class LMCacheConnectorMetadata(KVConnectorMetadata):
    requests: list[ReqMeta]

    def __init__(self):
        self.requests = []

    def add_request(self, req_meta: ReqMeta) -> None:
        """Add a request to the metadata.

        Args:
            req_meta (ReqMeta): the request metadata.
        """
        self.requests.append(req_meta)


class LMCacheConnectorV1Impl:

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole,
                 parent: KVConnectorBase_V1):
        self._parent = parent
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self.layers_data = None
        is_tp = vllm_config.parallel_config.tensor_parallel_size > 1
        self.pd_disaggregated = vllm_config.kv_transfer_config.get_from_extra_config(
            "pd_disaggregated", False)
        if role == KVConnectorRole.SCHEDULER:
            self.lookup_client = LMCacheLookupClient(role, is_tp, vllm_config)
        else:
            self.lmcache_engine = init_lmcache_engine(
                vllm_config.model_config, vllm_config.parallel_config,
                vllm_config.cache_config)
            self.use_layerwise = isinstance(self.lmcache_engine, LayerwiseLMCacheEngine)
            self.use_layeraware = isinstance(self.lmcache_engine, LayerAwareLMCacheEngine)

            # NOTE: Only create the KV lookup API server on worker rank 0
            # when there are multiple workers
            assert self.lmcache_engine is not None
            if vllm_config.parallel_config.rank == 0:
                self.lookup_server = LMCacheLookupServer(
                    self.lmcache_engine, role, is_tp, vllm_config)

        self.kv_caches: dict[str, torch.Tensor] = {}

        self._block_size = vllm_config.cache_config.block_size

        # request_id -> (vllm cached tokes, lmcache cached tokens)
        self.load_specs: dict[str, LoadSpec] = {}

        self.kv_cache_manager: Optional[KVCacheManager] = None

        # request_id -> full_token_ids
        self._request_trackers: dict[str, RequestTracker] = {}

        # Whether to discard partial chunks
        self._discard_partial_chunks = vllm_config\
                .kv_transfer_config.get_from_extra_config(
                    "discard_partial_chunks", False)

        # FIXME(Jiayi): need to align this chunk size with lmcache
        self._lmcache_chunk_size = 256

        self.skip_last_n_tokens = \
            vllm_config.kv_transfer_config.get_from_extra_config(
                "skip_last_n_tokens", 0)

        self.num_layers = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config)
        self.current_layer = 0

    def _init_kv_caches_from_forward_context(
            self, forward_context: "ForwardContext"):
        for layer_name in forward_context.no_compile_layers:
            attn_layer = forward_context.no_compile_layers[layer_name]
            if not hasattr(attn_layer, "kv_cache"):
                logger.debug("The layer %s does not have kv_cache, skip it",
                             layer_name)
                continue

            if layer_name not in self.kv_caches:
                self.kv_caches[layer_name] = attn_layer.kv_cache[\
                        forward_context.virtual_engine]

    ####################
    # Worker side APIs
    ####################
    @_lmcache_nvtx_annotate
    def start_load_kv(self, forward_context: "ForwardContext",
                      **kwargs) -> None:
        """Start loading the KV cache from the connector buffer to vLLM's 
        paged KV buffer.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation

        Note:
            The number of elements in kv_caches and layer_names should be 
            the same.
        """
        self.current_layer = 0

        if len(self.kv_caches) == 0:
            self._init_kv_caches_from_forward_context(forward_context)

        metadata = self._parent._get_connector_metadata()
        assert isinstance(metadata, LMCacheConnectorMetadata)

        assert len(self.kv_caches) > 0
        kvcaches = list(self.kv_caches.values())

        assert self.lmcache_engine is not None

        self.layerwise_retrievers = []
        
        self.layeraware_request_data = {}
        
        for request in metadata.requests:
            tokens = request.token_ids
            
            if request.load_spec is None:
                # used for publish first decode logits.
                self.layeraware_request_data[request.req_id] = {
                    'token_ids': tokens,
                }
                continue

            # TODO: have a pre-allocated buffer to hold the slot_mappings
            slot_mapping = request.slot_mapping.cuda()
            assert len(tokens) == len(slot_mapping)

            token_mask = torch.ones_like(tokens, dtype=torch.bool)
            masked_token_count = request.load_spec.vllm_cached_tokens // \
                self._lmcache_chunk_size * self._lmcache_chunk_size
            token_mask[:masked_token_count] = False

            if self.skip_last_n_tokens > 0:
                tokens = tokens[:-self.skip_last_n_tokens]
                token_mask = token_mask[:-self.skip_last_n_tokens]

            if self.use_layerwise:
                assert isinstance(self.lmcache_engine, LayerwiseLMCacheEngine)
                layerwise_retriever = self.lmcache_engine.retrieve_layer(
                    tokens,
                    token_mask,
                    kvcaches=kvcaches,
                    slot_mapping=slot_mapping,
                )
                # NOTE: retrieve for two layers at the first layer
                next(layerwise_retriever)
                next(layerwise_retriever)
                self.layerwise_retrievers.append(layerwise_retriever)
            elif self.use_layeraware:
                layers_data = self.lmcache_engine._group_keys_by_layers_first(
                    tokens,
                    token_mask,
                )

                self.layeraware_request_data[request.req_id] = {
                    'layers_data': layers_data,
                    'token_ids': tokens,
                    'token_mask': token_mask,
                    'slot_mapping': slot_mapping,
                    'kvcaches': kvcaches,
                    'is_load': True,
                }
                logger.debug(
                    f"Prepared layer-aware data for request {request.req_id}: "
                    f"{len(layers_data)} layers with data")
            else:
                ret_token_mask = self.lmcache_engine.retrieve(
                    tokens,
                    token_mask,
                    kvcaches=kvcaches,
                    slot_mapping=slot_mapping,
                )

                # Check the result
                num_retrieved_tokens = ret_token_mask.sum().item()
                num_expected_tokens = \
                    request.load_spec.lmcache_cached_tokens - \
                        request.load_spec.vllm_cached_tokens - \
                        self.skip_last_n_tokens
                if num_retrieved_tokens < num_expected_tokens:
                    logger.error(
                        "The number of retrieved tokens is less than the "
                        "expected number of tokens! This should not happen!")
                    logger.error(
                        "Num retrieved tokens: %d, num expected tokens: %d",
                        num_retrieved_tokens, num_expected_tokens)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Blocking until the KV for a specific layer is loaded into vLLM's
        paged buffer. 
        
        Enhanced for LayerAwareLMCacheEngine: This method now supports progressive
        layer loading where individual layers become available one by one.
        Instead of waiting for the entire KV cache, we wait only for the specific
        layer needed by the current attention computation.

        Args:
            layer_name: the name of that layer (e.g. "layers.0", "layers.1")
        """
        # Parse layer ID from layer name (e.g. "layers.0" -> 0)
        layer_id = self._parse_layer_id(layer_name)
        
        if self.use_layeraware:
            # Get the metadata to access request information
            metadata = self._parent._get_connector_metadata()
            assert isinstance(metadata, LMCacheConnectorMetadata)
            
            # Process each request that needs layer loading
            for request in metadata.requests:
                if request.load_spec is None or not request.load_spec.can_load:
                    continue

                # Get the stored data for this request
                if request.req_id not in self.layeraware_request_data:
                    logger.warning(f"No layer data found for request {request.req_id}")
                    continue
                
                request_data = self.layeraware_request_data[request.req_id]
                # Check if this request has load data
                if 'is_load' not in request_data or not request_data['is_load']:
                    logger.warning(f"No load data found for request {request.req_id}")
                    continue
                    
                tokens = request_data['token_ids']
                token_mask = request_data['token_mask']
                slot_mapping = request_data['slot_mapping']
                kvcaches = request_data['kvcaches']
                layers_data = request_data['layers_data']
                
                # Calculate number of chunks based on token length
                num_chunks = len(tokens) // self._lmcache_chunk_size
                if len(tokens) % self._lmcache_chunk_size != 0:
                    num_chunks += 1
                
                ret_mask = self.lmcache_engine.retrieve_layer_when_ready(
                    layer_id,
                    tokens,
                    token_mask,
                    timeout_us=1000000,
                    layers_data=layers_data,
                    num_chunks=num_chunks,
                    slot_mapping=slot_mapping,
                    kvcaches=kvcaches
                )
                
                # Log the result for debugging
                if ret_mask is not None:
                    num_retrieved_tokens = ret_mask.sum().item()
                    logger.debug(f"Retrieved {num_retrieved_tokens} tokens for layer {layer_id} in request {request.req_id}")
                else:
                    logger.debug(f"No tokens retrieved for layer {layer_id} in request {request.req_id}")
        elif self.layerwise_retrievers:
            self._wait_using_retrievers(layer_id)

    def _parse_layer_id(self, layer_name: str) -> int:
        """
        Parse layer ID from layer name.
        
        Args:
            layer_name: Layer name like "layers.0", "layers.1", etc.
            
        Returns:
            Layer ID as integer
        """
        try:
            import re
            numbers = re.findall(r'\d+', layer_name)
            return int(numbers[-1]) if numbers else 0
        except (ValueError, IndexError):
            logger.warning(f"Could not parse layer ID from {layer_name}, defaulting to 0")
            return 0

    def _wait_using_retrievers(self, target_layer_id: int) -> None:
        """
        Wait for layer using the layerwise retrievers (original method).
        
        Args:
            target_layer_id: The layer ID to wait for
        """
        # Wait for the layer to be loaded using retrievers
        for layerwise_retriever in self.layerwise_retrievers:
            # We've already processed the first setup yields in start_load_kv

            ret_token_mask = next(layerwise_retriever)
            
            # Check if this is the final result (non-None mask)
            if ret_token_mask is not None:
                num_retrieved_tokens = ret_token_mask.sum().item()
                logger.debug(f"Retrieved {num_retrieved_tokens} tokens for target {target_layer_id+1} layer.")

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata: "AttentionMetadata", **kwargs) -> None:
        """Start saving the a layer of KV cache from vLLM's paged buffer 
        to the connector.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current 
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """

        if not self.use_layerwise and not self.use_layeraware:
            return

        if self.kv_role == "kv_consumer":
            # Don't do save if the role is kv_consumer
            return

        connector_metadata = self._parent._get_connector_metadata()
        assert isinstance(connector_metadata, LMCacheConnectorMetadata)

        assert len(self.kv_caches) > 0

        assert isinstance(self.lmcache_engine, (LayerwiseLMCacheEngine, LayerAwareLMCacheEngine))

        kvcaches = list(self.kv_caches.values())
        
        if self.use_layeraware:
            # LayerAwareLMCacheEngine uses direct method calls for single layer storage
            if self.current_layer == 0:
                # Initialize layer-aware storage data for all requests
                self.layeraware_request_data = {}
                for request in connector_metadata.requests:
                    save_spec = request.save_spec
                    if save_spec is None or not save_spec.can_save:
                        continue

                    token_ids = request.token_ids
                    assert isinstance(token_ids, torch.Tensor)
                    assert token_ids.is_cpu

                    slot_mapping = request.slot_mapping
                    assert isinstance(slot_mapping, torch.Tensor)
                    assert len(slot_mapping) == len(token_ids)

                    # TODO: have a pre-allocated buffer to hold the slot_mappings
                    slot_mapping = slot_mapping.cuda()
                    # NOTE: In PD setting, lmcache_engine.lookup() will always
                    # return 0 if there is no local storage configured.
                    # In this case, we should rely on the skip_leading_tokens in
                    # save_spec to avoid transmit the already saved tokens again.
                    skip_leading_tokens = max(
                        self.lmcache_engine.lookup(token_ids),
                        save_spec.skip_leading_tokens)
                    if skip_leading_tokens == len(token_ids):
                        continue  # skip this request
                    # Align to lmcache chunk size
                    skip_leading_tokens = skip_leading_tokens // \
                            self._lmcache_chunk_size * self._lmcache_chunk_size

                    store_mask = torch.ones_like(token_ids, dtype=torch.bool)
                    store_mask[:skip_leading_tokens] = False

                    logger.info(
                        "Storing KV cache for %d out of %d tokens for request %s",
                        len(token_ids) - skip_leading_tokens, len(token_ids),
                        request.req_id)
                    
                    # Group keys by layers first for LayerAwareLMCacheEngine
                    layers_data = self.lmcache_engine._group_keys_by_layers_first(
                        token_ids,
                        store_mask,
                    )
                    
                    # Store the data for each request
                    self.layeraware_request_data[request.req_id] = {
                        'layers_data': layers_data,
                        'token_ids': token_ids,
                        'store_mask': store_mask,
                        'slot_mapping': slot_mapping,
                        'kvcaches': kvcaches,
                        'skip_leading_tokens': skip_leading_tokens,
                    }
            
            # Process the current layer for all requests
            current_layer_id = self.current_layer
            for req_id, request_data in self.layeraware_request_data.items():
                layers_data = request_data['layers_data']
                token_ids = request_data['token_ids']
                store_mask = request_data['store_mask']
                slot_mapping = request_data['slot_mapping']
                kvcaches = request_data['kvcaches']
                skip_leading_tokens = request_data['skip_leading_tokens']
                
                # Check if this layer has data to store
                if current_layer_id in layers_data:
                    layer_chunks = layers_data[current_layer_id]
                    
                    # Determine memory format and RDMA usage
                    memory_format = self.lmcache_engine._get_memory_format_for_connector()
                    using_nixl = isinstance(self.lmcache_engine.storage_manager,
                                            DistributedStorageManager)
                    
                    # Store this single layer
                    layer_stored = self.lmcache_engine._store_single_layer_progressive(
                        current_layer_id,
                        layer_chunks,
                        memory_format,
                        using_nixl,
                        kvcaches=kvcaches,
                        slot_mapping=slot_mapping,
                        offset=skip_leading_tokens,
                        **kwargs
                    )
                    
                    logger.debug(
                        f"Stored {layer_stored} chunks for layer {current_layer_id} "
                        f"in request {req_id}")
                else:
                    logger.debug(
                        f"No data to store for layer {current_layer_id} "
                        f"in request {req_id}")
            
            self.current_layer += 1
            
        else: # use layerwise; generator pattern
            if self.current_layer == 0:
                self.layerwise_storers = []
                for request in connector_metadata.requests:
                    save_spec = request.save_spec
                    if save_spec is None or not save_spec.can_save:
                        continue

                    token_ids = request.token_ids
                    assert isinstance(token_ids, torch.Tensor)
                    assert token_ids.is_cpu

                    slot_mapping = request.slot_mapping
                    assert isinstance(slot_mapping, torch.Tensor)
                    assert len(slot_mapping) == len(token_ids)

                    # TODO: have a pre-allocated buffer to hold the slot_mappings
                    slot_mapping = slot_mapping.cuda()
                    # NOTE: In PD setting, lmcache_engine.lookup() will always
                    # return 0 if there is no local storage configured.
                    # In this case, we should rely on the skip_leading_tokens in
                    # save_spec to avoid transmit the already saved tokens again.
                    skip_leading_tokens = max(
                        self.lmcache_engine.lookup(token_ids),
                        save_spec.skip_leading_tokens)
                    if skip_leading_tokens == len(token_ids):
                        continue  # skip this request
                    # Align to lmcache chunk size
                    skip_leading_tokens = skip_leading_tokens // \
                            self._lmcache_chunk_size * self._lmcache_chunk_size

                    store_mask = torch.ones_like(token_ids, dtype=torch.bool)
                    store_mask[:skip_leading_tokens] = False

                    logger.info(
                        "Storing KV cache for %d out of %d tokens for request %s",
                        len(token_ids) - skip_leading_tokens, len(token_ids),
                        request.req_id)
                    layerwise_storer = self.lmcache_engine.store_layer(
                        token_ids,
                        mask=store_mask,
                        kvcaches=kvcaches,
                        slot_mapping=slot_mapping,
                        offset=skip_leading_tokens)
                    self.layerwise_storers.append(layerwise_storer)

            for layerwise_storer in self.layerwise_storers:
                next(layerwise_storer)
                if self.current_layer == self.num_layers - 1:
                    next(layerwise_storer)
            self.current_layer += 1

    @_lmcache_nvtx_annotate
    def wait_for_save(self):
        """Blocking until the KV cache is saved to the connector buffer."""
        if self.kv_role == "kv_consumer":
            # Don't do save if the role is kv_consumer
            return
        if self.use_layerwise or self.use_layeraware:
            return
        connector_metadata = self._parent._get_connector_metadata()
        assert isinstance(connector_metadata, LMCacheConnectorMetadata)
        assert len(self.kv_caches) > 0
        kvcaches = list(self.kv_caches.values())
        assert self.lmcache_engine is not None
        for request in connector_metadata.requests:
            save_spec = request.save_spec
            if save_spec is None or not save_spec.can_save:
                continue
            token_ids = request.token_ids
            assert isinstance(token_ids, torch.Tensor)
            assert token_ids.is_cpu
            slot_mapping = request.slot_mapping
            assert isinstance(slot_mapping, torch.Tensor)
            assert len(slot_mapping) == len(token_ids)
            # TODO: have a pre-allocated buffer to hold the slot_mappings
            slot_mapping = slot_mapping.cuda()
            # NOTE: In PD setting, lmcache_engine.lookup() will always return
            # 0 if there is no local storage configured. In this case, we
            # should rely on the skip_leading_tokens in save_spec to avoid
            # transmit the already saved tokens again.
            skip_leading_tokens = max(self.lmcache_engine.lookup(token_ids),
                                      save_spec.skip_leading_tokens)
            if skip_leading_tokens == len(token_ids):
                continue  # skip this request
            # Align to lmcache chunk size
            skip_leading_tokens = skip_leading_tokens // \
                    self._lmcache_chunk_size * self._lmcache_chunk_size
            store_mask = torch.ones_like(token_ids, dtype=torch.bool)
            store_mask[:skip_leading_tokens] = False
            logger.info(
                "Storing KV cache for %d out of %d tokens for request %s",
                len(token_ids) - skip_leading_tokens, len(token_ids),
                request.req_id)
            self.lmcache_engine.store(token_ids,
                                      mask=store_mask,
                                      kvcaches=kvcaches,
                                      slot_mapping=slot_mapping,
                                      offset=skip_leading_tokens)

    def _generate_logits_key(self, token_ids: list[int]) -> str:
        """
        Generate a consistent logits key based on the hash of token_ids.
        This ensures that both prefiller and decoder use the same key
        regardless of potential req_id differences.
        
        Args:
            token_ids: List of token IDs for the prompt
            
        Returns:
            Consistent string key for storing/retrieving logits
        """
        # Convert token_ids to a consistent string representation
        token_str = ",".join(map(str, token_ids))
        # Create a hash for a shorter, consistent key
        token_hash = hashlib.sha256(token_str.encode()).hexdigest()[:16]
        return f"first_decode_logits:{token_hash}"

    def publish_first_decode_logits(self, req_id: str, logits: torch.Tensor) -> None:
        """
        Publish the last prompt token logits for a request to enable
        zero-compute first decode on the decoder worker.
        
        Args:
            req_id: The request ID
            logits: Last prompt token logits tensor [vocab_size]
        """
        if self.kv_role == "kv_consumer":
            # Only prefiller publishes logits
            return

        if not isinstance(self.lmcache_engine, LayerAwareLMCacheEngine):
            return
            
        # Get token_ids from the request tracker to generate consistent key
        if req_id not in self.layeraware_request_data:
            logger.warning(f"No request tracker found for req_id {req_id}, cannot publish logits")
            return
            
        request_tracker = self.layeraware_request_data[req_id]
        logits_key = self._generate_logits_key(request_tracker['token_ids'].tolist())
        
        logger.info(f"Publishing first decode logits for request {req_id} with key {logits_key}")
        
        # Store logits directly in LMCache backend with high priority
        # This should preempt other transfers for immediate availability
        self.lmcache_engine.store_special_tensor(logits_key, logits.detach())
        logger.debug(f"Stored first decode logits in LMCache for request {req_id}")
            
    def try_consume_first_decode_logits(self, token_ids: list[int]) -> Optional[torch.Tensor]:
        """
        Try to retrieve the first decode logits for a request.
        
        Args:
            req_id: The request ID
            
        Returns:
            Logits tensor [vocab_size] if available, None otherwise
        """
        if self.kv_role == "kv_producer":
            # Only decoder consumes logits
            return None
        
        if not isinstance(self.lmcache_engine, LayerAwareLMCacheEngine):
            return None

        logits_key = self._generate_logits_key(token_ids)
        
        # Retrieve from LMCache backend with high priority
        # This should preempt other transfers and return immediately from recv_obj_pool
        # The backend should check high-priority queue first before regular KV transfers
        logits = self.lmcache_engine.retrieve_special_tensor(logits_key)
        if logits is not None:
            return logits.cuda() if logits.device.type == 'cpu' else logits
        # consumer's scheduler, return directly
        return None

    ###################
    # Scheduler side APIs
    ####################

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
        **kwargs,
    ) -> tuple[int, bool]:
        """
        Check for external KV cache hit.
        
        Implements policy:
        - First decode: KV from prefiller (don't rely on scheduler waiting)
        - Following decodes: KV from local storage
        
        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            Tuple of (num_tokens_available, should_load_async):
            - num_tokens_available: number of tokens that can be loaded from external cache
            - should_load_async: True to force async layer-wise loading
        """
        if self.kv_role == "kv_producer":
            return 0, False

        if num_computed_tokens == 0:
            prompt_length = len(request.prompt_token_ids)
            if self.skip_last_n_tokens > 0:
                prompt_length -= self.skip_last_n_tokens

            if kwargs.get("skip_logits") and self.has_first_decode_logits(request):
                # With logits available, we can use all prompt tokens from external cache
                num_external_hit_tokens = prompt_length
                logger.info(
                    "First decode: Reqid: %s, External hit tokens: %d, zero-compute decode with logits",
                    request.request_id, num_external_hit_tokens)
                    
                self.load_specs[request.request_id] = LoadSpec(
                    vllm_cached_tokens=num_computed_tokens,
                    lmcache_cached_tokens=num_computed_tokens + num_external_hit_tokens,
                    can_load=False)
                return num_external_hit_tokens, False
            elif self.kv_role == "kv_consumer": 
                # use prompt_length - 1 to ensure num_new_tokens = 1
                num_external_hit_tokens = max(prompt_length - 1, 0) # TODO: it's strange. What is the difference between len=1's and len>1's? 
                
                if num_external_hit_tokens > 0:
                    logger.info(
                        "First decode: Reqid: %s, External hit tokens: %d, will compute 1 new token",
                        request.request_id, num_external_hit_tokens)
                    
                    self.load_specs[request.request_id] = LoadSpec(
                        vllm_cached_tokens=num_computed_tokens,
                        lmcache_cached_tokens=num_computed_tokens + num_external_hit_tokens,
                        can_load=False)
                    return num_external_hit_tokens, False
            else: # non PD-disagg scenario
                logger.info(
                    "First decode: Reqid: %s, No external hits, will compute all %d tokens locally",
                    request.request_id, prompt_length)
                return 0, False
                
        # Following decodes: Use local storage only
        token_ids = torch.tensor(request.prompt_token_ids)
        if self.skip_last_n_tokens > 0:
            num_external_hit_tokens = self.lookup_client.lookup(
                token_ids[:-self.skip_last_n_tokens])
        else:
            num_external_hit_tokens = self.lookup_client.lookup(token_ids)
        
        if kwargs.get("skip_logits"): # Default is skip_logits = True
            num_external_hit_tokens = min(num_external_hit_tokens, request.num_tokens-1)

        need_to_allocate = num_external_hit_tokens - num_computed_tokens
        
        if need_to_allocate <= 0:
            logger.debug(
                "Following decode: Reqid: %s, No new external hits needed, "
                "computed: %d, External hit tokens: %d", request.request_id,
                num_computed_tokens, num_external_hit_tokens)
            return 0, False
        
        logger.info(
            "Following decode: Reqid: %s, External hit tokens: %d, need to load: %d",
            request.request_id, num_external_hit_tokens, need_to_allocate)
        
        self.load_specs[request.request_id] = LoadSpec(
            vllm_cached_tokens=num_computed_tokens,
            lmcache_cached_tokens=num_external_hit_tokens,
            can_load=False)
        
        return need_to_allocate, False

    def has_first_decode_logits(self, request: "Request") -> bool:
        """
        Check if first decode logits are available for a request.
        This is called from the scheduler side to determine if zero-compute is possible.
        
        Args:
            request: The request object
            
        Returns:
            True if first decode logits are available
        """
        if self.kv_role == "kv_producer":
            # Only decoder needs to check for logits
            return False
        
        # Generate consistent logits key using request's prompt token_ids
        logits_key = self._generate_logits_key(request.prompt_token_ids)
        
        # Use lookup_client to check if logits are available in the cache
        # This goes through the proper client/server protocol
        try:
            has_logits = self.lookup_client.check_special_tensor_available(logits_key)
            if has_logits:
                request.has_first_decode_logits = True
                logger.debug(f"First decode logits available for request {request.request_id}")
                return True
        except Exception as e:
            logger.debug(f"Failed to check logits availability in LMCache for {request.request_id}: {e}")
            
        # No logits
        return False

    def update_state_after_alloc(self, request: "Request",
                                 num_external_tokens: int):
        """
        Update KVConnector state after temporary buffer alloc.

        For SharedStorageConnector, update _request_needs_load
        if the CacheManager this allocated blocks for us.
        """
        if request.request_id not in self.load_specs:
            # No KV tokens from external KV cache, return
            return

        if num_external_tokens == 0:
            # No need to load anything
            self.load_specs[request.request_id].can_load = False
            return

        assert num_external_tokens > 0 and num_external_tokens == \
            self.load_specs[request.request_id].lmcache_cached_tokens - \
            self.load_specs[request.request_id].vllm_cached_tokens, \
            f"Mismatch in number of tokens: {num_external_tokens} vs " \
            f"{self.load_specs[request.request_id].lmcache_cached_tokens} - " \
            f"{self.load_specs[request.request_id].vllm_cached_tokens}" \
            f" for request {request.request_id}"

        self.load_specs[request.request_id].can_load = True

    def build_connector_meta(
            self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        """Attach the connector metadata to the request object.

        This function should NOT modify other fields in the scheduler_output 
        except the `kv_connector_metadata` field.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """

        force_skip_save = self.kv_role == "kv_consumer"

        meta = LMCacheConnectorMetadata()

        for finished_req_id in scheduler_output.finished_req_ids:
            self._request_trackers.pop(finished_req_id, None)

        for request in scheduler_output.scheduled_new_reqs:
            # Right now, we only load KV for new requests
            load_spec = self.load_specs.pop(request.req_id, None)
            num_tokens_to_compute = request.num_computed_tokens + \
                    scheduler_output.num_scheduled_tokens[request.req_id]
            request_tracker = RequestTracker.from_new_request(
                request, num_tokens_to_compute)
            self._request_trackers[request.req_id] = request_tracker

            req_meta = ReqMeta.from_request_tracker(
                request_tracker,
                self._block_size,
                self._lmcache_chunk_size,
                load_spec=load_spec,
                skip_save=force_skip_save,
                discard_partial_chunks=self._discard_partial_chunks)
            if req_meta is not None:
                meta.add_request(req_meta)

        for request in scheduler_output.scheduled_cached_reqs:
            request_tracker = self._request_trackers[request.req_id]
            request_tracker.update(request)

            req_meta = ReqMeta.from_request_tracker(
                request_tracker,
                self._block_size,
                self._lmcache_chunk_size,
                load_spec=None,
                skip_save=force_skip_save,
                discard_partial_chunks=self._discard_partial_chunks)
            if req_meta is not None:
                meta.add_request(req_meta)

        return meta

