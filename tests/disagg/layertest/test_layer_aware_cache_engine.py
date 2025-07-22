#!/usr/bin/env python3
"""
Test LayerAware Cache Engine for Disaggregated Inference

This test demonstrates layer-aware caching where:
- Sender transmits layers progressively (layer-0, layer-1, etc.)
- Receiver starts processing as soon as early layers are available
- Performance is measured for both layer transmission and early processing latency
"""

import argparse
import random
import time
import threading
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from lmcache.config import LMCacheEngineMetadata
from lmcache.experimental.cache_engine import LMCacheEngineBuilder, LayerAwareLMCacheEngine
from lmcache.experimental.config import LMCacheEngineConfig
from lmcache.experimental.gpu_connector import VLLMPagedMemGPUConnectorV2, VLLMPagedMemLayerwiseGPUConnector
from lmcache.experimental.token_database import LayerFirstTokenDataBase
from lmcache.experimental.memory_management import MixedMemoryAllocator
from lmcache.logging import init_logger

logger = init_logger(__name__)

num_layers = 32


def generate_test_tokens(num_chunks: int, chunk_size: int) -> torch.Tensor:
    """Generate test tokens for testing."""
    return torch.arange(0,
                        num_chunks * chunk_size,
                        dtype=torch.long,
                        device="cuda")


def generate_kv_cache_paged_list_tensors(num_blocks,
                                         device,
                                         block_size=16,
                                         dtype=torch.bfloat16):
    """Generate paged KV cache tensors with deterministic initialization for testing."""
    ret = []
    num_heads = 8
    head_size = 128
    shape = [2, num_blocks, block_size, num_heads, head_size]

    for i in range(num_layers):
        # torch.manual_seed(42 + i)
        # kv = torch.rand(shape, dtype=dtype, device=device)

        # Use deterministic initialization instead of random values
        # Each layer gets a unique base value for easier debugging/verification
        base_value = 0.1 + (i * 0.01)  # Layer 0: 0.1, Layer 1: 0.11, etc.
        kv = torch.full(shape, base_value, dtype=dtype, device=device)
        ret.append(kv)

    return ret


def fill_kv_cache_with_layer_pattern(kv_cache, slot_mapping, base_pattern=0.8):
    """Fill KV cache with layer-specific patterns for verification."""
    for layer_idx, layer_tensor in tqdm(enumerate(kv_cache),
                                        total=len(kv_cache),
                                        desc="Filling KV cache"):
        # Each layer gets a unique pattern value
        pattern_value = base_pattern + (layer_idx * 0.01)

        num_blocks = layer_tensor.shape[1]
        block_size = layer_tensor.shape[2]
        new_shape = (2, num_blocks * block_size, 8, 128)
        layer_tensor.reshape(new_shape)[:, slot_mapping, :, :] = pattern_value

    return kv_cache


def verify_layer_pattern(kv_cache,
                         slot_mapping,
                         layer_id,
                         base_pattern=0.8,
                         tolerance=0.01):
    """Verify that a specific layer contains the expected pattern."""
    expected_pattern = base_pattern + (layer_id * 0.01)
    layer_tensor = kv_cache[layer_id]

    num_blocks = layer_tensor.shape[1]
    block_size = layer_tensor.shape[2]
    new_shape = (2, num_blocks * block_size, 8, 128)
    actual_values = layer_tensor.reshape(new_shape)[:, slot_mapping, :, :]

    mean_value = actual_values.mean().item()
    is_correct = abs(mean_value - expected_pattern) <= tolerance

    logger.info(
        f"Layer {layer_id} verification: expected={expected_pattern:.3f}, "
        f"actual={mean_value:.3f}, correct={is_correct}")

    return is_correct


def calculate_throughput(total_bytes: int, elapsed_time: float) -> float:
    """Calculate throughput in GB/s."""
    if elapsed_time == 0:
        return float('inf')
    gb = total_bytes / (1024 * 1024 * 1024)
    return gb / elapsed_time


def create_config(role: str, host: str, port: int) -> LMCacheEngineConfig:
    """Create LayerAware-compatible configuration."""
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        local_cpu=False,
        max_local_cpu_size=0,
        local_disk=None,
        max_local_disk_size=0,
        remote_url=None,
        remote_serde=None,
        save_decode_cache=False,
        enable_p2p=False,
        enable_nixl=True,
        nixl_role=role,
        nixl_receiver_host=host,
        nixl_receiver_port=port,
        nixl_buffer_size=2**30,  # 1GB
        nixl_buffer_device='cuda',
    )
    return config


def create_metadata() -> LMCacheEngineMetadata:
    """Create metadata with layer information."""
    chunk_size = 256
    num_heads = 32
    head_dim = 128
    kv_shape = (num_layers, 2, chunk_size, num_heads, head_dim)

    return LMCacheEngineMetadata(
        model_name="test_layer_aware_model",
        world_size=1,
        worker_id=0,
        fmt="vllm",
        kv_dtype=torch.bfloat16,
        kv_shape=kv_shape,
    )


def create_layer_aware_engine(config: LMCacheEngineConfig,
                              metadata: LMCacheEngineMetadata,
                              num_chunks: int) -> LayerAwareLMCacheEngine:
    """Create LayerAware Cache Engine with proper components."""
    # Create LayerFirst token database
    token_database = LayerFirstTokenDataBase(config, metadata)

    # Create memory allocator
    memory_allocator = MixedMemoryAllocator(int(10 * 1024**3))  # 2GB

    # Create GPU connector
    hidden_dim = 1024
    gpu_connector = VLLMPagedMemLayerwiseGPUConnector(
        hidden_dim,
        metadata.kv_shape[0],
        use_gpu=True,
        chunk_size=config.chunk_size,
        dtype=metadata.kv_dtype,
        device="cuda",
        max_tokens=num_chunks * config.chunk_size,
    )

    # Create LayerAware cache engine
    engine = LayerAwareLMCacheEngine(config=config,
                                     metadata=metadata,
                                     memory_allocator=memory_allocator,
                                     token_database=token_database,
                                     gpu_connector=gpu_connector)

    return engine


class LayerPerformanceTracker:
    """Track layer-specific performance metrics with detailed debugging."""

    def __init__(self, num_layers: int):
        self.num_layers: int = num_layers
        self.layer_times: Dict[int, float] = {}
        self.layer_ready_times: Dict[int, float] = {}
        self.layer_wait_times: Dict[int, float] = {
        }  # Time spent waiting for each layer
        self.layer_retrieve_times: Dict[int, float] = {
        }  # Time spent retrieving each layer
        self.layer_verify_times: Dict[int, float] = {
        }  # Time spent verifying each layer
        self.first_layer_latency: Optional[float] = None
        self.total_completion_time: Optional[float] = None

        # Detailed phase tracking
        self.phase_times = {
            'total_wait_time': 0.0,
            'total_retrieve_time': 0.0,
            'total_verify_time': 0.0,
            'total_gpu_transfer_time': 0.0
        }

    def record_layer_wait_start(self, layer_id: int) -> float:
        """Record when we start waiting for a layer. Returns timestamp."""
        timestamp = time.perf_counter()
        return timestamp

    def record_layer_wait_end(self, layer_id: int, start_timestamp: float):
        """Record when we finish waiting for a layer."""
        wait_time = time.perf_counter() - start_timestamp
        self.layer_wait_times[layer_id] = wait_time
        self.phase_times['total_wait_time'] += wait_time

    def record_layer_retrieve_start(self, layer_id: int) -> float:
        """Record when we start retrieving a layer. Returns timestamp."""
        return time.perf_counter()

    def record_layer_retrieve_end(self, layer_id: int, start_timestamp: float):
        """Record when we finish retrieving a layer."""
        retrieve_time = time.perf_counter() - start_timestamp
        self.layer_retrieve_times[layer_id] = retrieve_time
        self.phase_times['total_retrieve_time'] += retrieve_time

    def record_layer_verify_start(self, layer_id: int) -> float:
        """Record when we start verifying a layer. Returns timestamp."""
        return time.perf_counter()

    def record_layer_verify_end(self, layer_id: int, start_timestamp: float):
        """Record when we finish verifying a layer."""
        verify_time = time.perf_counter() - start_timestamp
        self.layer_verify_times[layer_id] = verify_time
        self.phase_times['total_verify_time'] += verify_time

    def record_layer_ready(self, layer_id: int, timestamp: float):
        """Record when a layer becomes ready."""
        self.layer_ready_times[layer_id] = timestamp

    def record_layer_processed(self, layer_id: int, process_time: float):
        """Record layer processing time."""
        self.layer_times[layer_id] = process_time

    def get_detailed_breakdown(self) -> Dict:
        """Get detailed timing breakdown for debugging."""
        if not self.layer_ready_times:
            return {"status": "no_data"}

        first_ready = min(self.layer_ready_times.values())
        last_ready = max(self.layer_ready_times.values())

        # Calculate first layer latency (time to first layer ready)
        self.first_layer_latency = first_ready

        # Calculate total completion time
        self.total_completion_time = last_ready - first_ready

        # Calculate processing efficiency
        total_process_time = sum(
            self.layer_times.values()) if self.layer_times else 0

        # Phase analysis
        total_accounted_time = sum(self.phase_times.values())

        return {
            "status":
            "success",
            "first_layer_latency_ms":
            self.first_layer_latency * 1000,
            "total_completion_time_ms":
            self.total_completion_time * 1000,
            "layers_ready":
            len(self.layer_ready_times),
            "layers_processed":
            len(self.layer_times),
            "total_process_time_ms":
            total_process_time * 1000,
            "avg_time_per_layer_ms":
            total_process_time / len(self.layer_times) *
            1000 if self.layer_times else 0,

            # Detailed phase breakdown
            "phase_breakdown": {
                "total_wait_time_ms":
                self.phase_times['total_wait_time'] * 1000,
                "total_retrieve_time_ms":
                self.phase_times['total_retrieve_time'] * 1000,
                "total_verify_time_ms":
                self.phase_times['total_verify_time'] * 1000,
                "total_gpu_transfer_time_ms":
                self.phase_times['total_gpu_transfer_time'] * 1000,
                "total_accounted_time_ms":
                total_accounted_time * 1000,
            },

            # Per-layer breakdown
            "per_layer_wait_times": {
                k: v * 1000
                for k, v in self.layer_wait_times.items()
            },
            "per_layer_retrieve_times": {
                k: v * 1000
                for k, v in self.layer_retrieve_times.items()
            },
            "per_layer_verify_times": {
                k: v * 1000
                for k, v in self.layer_verify_times.items()
            },
            "layer_ready_times": {
                k: v * 1000
                for k, v in self.layer_ready_times.items()
            },
            "layer_process_times": {
                k: v * 1000
                for k, v in self.layer_times.items()
            }
        }

    def get_metrics(self) -> Dict:
        """Get basic metrics for backward compatibility."""
        breakdown = self.get_detailed_breakdown()
        if breakdown["status"] == "no_data":
            return breakdown

        return {
            "first_layer_latency_ms": breakdown["first_layer_latency_ms"],
            "total_completion_time_ms": breakdown["total_completion_time_ms"],
            "layers_ready": breakdown["layers_ready"],
            "layers_processed": breakdown["layers_processed"],
            "total_process_time_ms": breakdown["total_process_time_ms"],
            "avg_time_per_layer_ms": breakdown["avg_time_per_layer_ms"],
            "layer_ready_times": breakdown["layer_ready_times"],
            "layer_process_times": breakdown["layer_process_times"]
        }


def run_sender(args, config, metadata, tokens, kv_cache, slot_mapping, engine):
    """Run sender logic with progressive layer transmission."""

    # Calculate data size for throughput measurement
    kv_shape = engine.gpu_connector.get_shape(config.chunk_size)
    element_size = torch.tensor([], dtype=metadata.kv_dtype).element_size()
    chunk_size_bytes = torch.prod(torch.tensor(kv_shape)).item() * element_size
    total_size = chunk_size_bytes * args.num_chunks * metadata.kv_shape[0]

    logger.info(
        f"📡 SENDER: Starting progressive layer transmission for "
        f"{len(tokens)} tokens ({args.num_chunks} chunks, {metadata.kv_shape[0]} layers)"
    )

    start_time = time.time()

    # Use progressive layer storage - this will send layers in layer-first order
    engine.store_progressive_layers(tokens=tokens,
                                    notify_readiness=True,
                                    kvcaches=kv_cache,
                                    slot_mapping=slot_mapping)

    end_time = time.time()
    elapsed_time = end_time - start_time

    logger.info(
        f"✅ SENDER: Completed progressive transmission in {elapsed_time:.3f}s")
    throughput = calculate_throughput(total_size, elapsed_time)
    logger.info(f"📊 SENDER: Throughput: {throughput:.2f} GB/s")

    # Cleanup
    engine.close()
    return throughput


def run_receiver(args, config, metadata, tokens, retrieved_cache, slot_mapping,
                 engine):
    """Run receiver logic with early layer processing."""

    perf_tracker = LayerPerformanceTracker(metadata.kv_shape[0])
    verification_results = {}

    logger.info("⏳ RECEIVER: Waiting for early layers...")

    # Process layers progressively as they become ready
    processed_layers = []
    start_time = time.time()

    if args.retrieval_method == 'bulk':
        # Option 1: Use the new bulk retrieval method
        logger.info("🚀 RECEIVER: Using bulk progressive retrieval...")
        
        # Track bulk retrieval time
        bulk_start = time.perf_counter()
        results = engine.retrieve_layers_progressive(
            tokens=tokens,
            mask=None,
            layer_ids=list(range(metadata.kv_shape[0])),
            timeout_us=10000000,  # 10s timeout per layer
            kvcaches=retrieved_cache,
            slot_mapping=slot_mapping,
            num_chunks=args.num_chunks
        )
        bulk_time = time.perf_counter() - bulk_start
        
        # Process results from bulk retrieval
        for layer_id in sorted(results.keys()):
            layer_start = time.time()
            ret_mask = results.get(layer_id)
            
            if ret_mask is not None:
                layer_ready_time = time.time() - start_time
                perf_tracker.record_layer_ready(layer_id, layer_ready_time)
                
                # Special handling for first layer
                if layer_id == 0:
                    logger.info(
                        f"⚡ RECEIVER: First layer {layer_id} ready via bulk retrieval in {layer_ready_time*1000:.2f}ms!"
                    )
                
                retrieved_tokens = torch.sum(ret_mask).item()
                
                # Track verification time
                verify_start = perf_tracker.record_layer_verify_start(layer_id)
                
                # Verify the layer data
                if verify_layer_pattern(retrieved_cache, slot_mapping, layer_id, tolerance=0.01):
                    verification_results[layer_id] = "✅ PASS"
                else:
                    verification_results[layer_id] = "❌ FAIL"
                
                perf_tracker.record_layer_verify_end(layer_id, verify_start)
                
                layer_process_time = time.time() - layer_start
                perf_tracker.record_layer_processed(layer_id, layer_process_time)
                processed_layers.append(layer_id)
                
                # For bulk retrieval, we approximate timing breakdown
                # Since bulk method combines waiting and retrieval
                verify_time = perf_tracker.layer_verify_times.get(layer_id, 0) * 1000
                
                logger.info(
                    f"   ✨ Layer {layer_id}: {retrieved_tokens} tokens | "
                    f"Total: {layer_process_time*1000:.1f}ms | "
                    f"Verify: {verify_time:.1f}ms | "
                    f"{verification_results[layer_id]}")
            else:
                logger.warning(f"   ⚠️ Layer {layer_id}: No data from bulk retrieval")
        
        logger.info(f"🔍 Bulk retrieval completed in {bulk_time*1000:.2f}ms for {len(results)} layers")
        
    else:  # individual retrieval
        # Option 2: Use individual retrieval with pre-computed layer data (current approach)
        logger.info("🔧 RECEIVER: Using individual retrieval with pre-computed keys...")
        # Pre-compute all layer keys upfront before any waiting (like sender side)
        logger.info("🔧 RECEIVER: Pre-computing layer keys for all layers...")
        pre_compute_start = time.perf_counter()
        layers_data = engine._group_keys_by_layers_first(tokens, None)  # mask=None since we want all tokens
        pre_compute_time = time.perf_counter() - pre_compute_start
        logger.info(f"✅ RECEIVER: Pre-computed keys for {len(layers_data)} layers in {pre_compute_time*1000:.2f}ms")

        for target_layer in range(metadata.kv_shape[0]):
            layer_start = time.time()

            # Track waiting time
            wait_start = perf_tracker.record_layer_wait_start(target_layer)
            logger.debug(f"   🔄 Layer {target_layer}: Starting wait...")

            # Wait for this specific layer to be ready
            if engine.storage_manager.storage_backend.wait_for_layer_busy(
                    target_layer, timeout_us=10000000,
                    num_chunks=args.num_chunks):  # 10s timeout per layer
                perf_tracker.record_layer_wait_end(target_layer, wait_start)
                layer_ready_time = time.time() - start_time
                perf_tracker.record_layer_ready(target_layer, layer_ready_time)

                # Special handling for first layer - log the early processing message
                if target_layer == 0:
                    logger.info(
                        f"⚡ RECEIVER: First layer {target_layer} ready in {layer_ready_time*1000:.2f}ms! Starting early processing..."
                    )

                logger.debug(
                    f"   ⚡ Layer {target_layer}: Ready after {(time.time() - layer_start)*1000:.2f}ms wait"
                )

                # Track retrieval time
                retrieve_start = perf_tracker.record_layer_retrieve_start(
                    target_layer)

                # NEW: Use pre-computed layer data - no key computation needed during retrieval!
                ret_mask = engine.retrieve_layer_when_ready(
                    layer_id=target_layer,
                    tokens=tokens,
                    timeout_us=500000,  # 500ms timeout since we know it's ready
                    layers_data=layers_data,  # Pass pre-computed layer data
                    kvcaches=retrieved_cache,
                    slot_mapping=slot_mapping,
                    num_chunks=args.num_chunks)

                perf_tracker.record_layer_retrieve_end(target_layer,
                                                       retrieve_start)

                if ret_mask is not None:
                    retrieved_tokens = torch.sum(ret_mask).item()

                    # Track verification time
                    verify_start = perf_tracker.record_layer_verify_start(
                        target_layer)

                    # Verify the layer data
                    if verify_layer_pattern(retrieved_cache,
                                            slot_mapping,
                                            target_layer,
                                            tolerance=0.01):
                        verification_results[target_layer] = "✅ PASS"
                    else:
                        verification_results[target_layer] = "❌ FAIL"

                    perf_tracker.record_layer_verify_end(target_layer,
                                                         verify_start)

                    layer_process_time = time.time() - layer_start
                    perf_tracker.record_layer_processed(target_layer,
                                                        layer_process_time)

                    processed_layers.append(target_layer)

                    # Detailed per-layer timing
                    wait_time = perf_tracker.layer_wait_times.get(target_layer,
                                                                  0) * 1000
                    retrieve_time = perf_tracker.layer_retrieve_times.get(
                        target_layer, 0) * 1000
                    verify_time = perf_tracker.layer_verify_times.get(
                        target_layer, 0) * 1000

                    logger.info(
                        f"   ✨ Layer {target_layer}: {retrieved_tokens} tokens | "
                        f"Total: {layer_process_time*1000:.1f}ms | "
                        f"Wait: {wait_time:.1f}ms | "
                        f"Retrieve: {retrieve_time:.1f}ms | "
                        f"Verify: {verify_time:.1f}ms | "
                        f"{verification_results[target_layer]}")
                else:
                    perf_tracker.record_layer_wait_end(target_layer, wait_start)
                    logger.warning(
                        f"   ⚠️ Layer {target_layer}: Failed to retrieve data")
                    break
            else:
                perf_tracker.record_layer_wait_end(target_layer, wait_start)
                wait_time = (time.time() - layer_start) * 1000
                logger.warning(
                    f"   ⏰ Layer {target_layer}: Timeout after {wait_time:.1f}ms wait"
                )
                break

    # Check if we processed any layers at all
    if not processed_layers:
        logger.error("❌ RECEIVER: No layers were processed (timeout or error)")
        engine.close()
        return None

    total_time = time.time() - start_time

    # Performance analysis
    detailed_metrics = perf_tracker.get_detailed_breakdown()
    metrics = perf_tracker.get_metrics()  # For backward compatibility

    logger.info(f"\n📊 RECEIVER: Performance Analysis")
    logger.info(f"   Pre-compute time: {pre_compute_time*1000:.2f}ms")
    logger.info(
        f"   Processed layers: {len(processed_layers)}/{metadata.kv_shape[0]}")
    logger.info(
        f"   First layer latency: {metrics['first_layer_latency_ms']:.2f}ms")
    logger.info(f"   Total completion time: {total_time*1000:.2f}ms")
    logger.info(
        f"   Average time per layer: {metrics['avg_time_per_layer_ms']:.2f}ms")

    # Detailed phase breakdown
    if detailed_metrics["status"] != "no_data":
        phase_breakdown = detailed_metrics["phase_breakdown"]
        logger.info(f"\n🔍 DETAILED TIMING BREAKDOWN:")
        logger.info(
            f"   Pre-compute time: {pre_compute_time*1000:.2f}ms")
        logger.info(
            f"   Total wait time: {phase_breakdown['total_wait_time_ms']:.2f}ms"
        )
        logger.info(
            f"   Total retrieve time: {phase_breakdown['total_retrieve_time_ms']:.2f}ms"
        )
        logger.info(
            f"   Total verify time: {phase_breakdown['total_verify_time_ms']:.2f}ms"
        )
        logger.info(
            f"   Total accounted time: {phase_breakdown['total_accounted_time_ms']:.2f}ms"
        )
        logger.info(
            f"   Unaccounted time: {(total_time*1000) - phase_breakdown['total_accounted_time_ms']:.2f}ms"
        )

        # Phase percentages
        total_time_ms = total_time * 1000
        if total_time_ms > 0:
            pre_compute_pct = (pre_compute_time * 1000 / total_time_ms) * 100
            wait_pct = (phase_breakdown['total_wait_time_ms'] /
                        total_time_ms) * 100
            retrieve_pct = (phase_breakdown['total_retrieve_time_ms'] /
                            total_time_ms) * 100
            verify_pct = (phase_breakdown['total_verify_time_ms'] /
                          total_time_ms) * 100

            logger.info(f"\n📈 TIME DISTRIBUTION:")
            logger.info(f"   Pre-compute time: {pre_compute_pct:.1f}% of total")
            logger.info(f"   Wait time: {wait_pct:.1f}% of total")
            logger.info(f"   Retrieve time: {retrieve_pct:.1f}% of total")
            logger.info(f"   Verify time: {verify_pct:.1f}% of total")

            # Identify the bottleneck
            max_phase = max(
                [("Pre-compute", pre_compute_time * 1000),
                 ("Wait", phase_breakdown['total_wait_time_ms']),
                 ("Retrieve", phase_breakdown['total_retrieve_time_ms']),
                 ("Verify", phase_breakdown['total_verify_time_ms'])],
                key=lambda x: x[1])

            logger.info(
                f"   🚨 BOTTLENECK: {max_phase[0]} phase ({max_phase[1]:.2f}ms)"
            )

        # Key computation efficiency analysis
        estimated_per_layer_key_compute = pre_compute_time / len(layers_data) if layers_data else 0
        total_saved_time = estimated_per_layer_key_compute * len(processed_layers) * 1000
        logger.info(f"\n🎯 KEY COMPUTATION EFFICIENCY:")
        logger.info(f"   Estimated per-layer key computation time: {estimated_per_layer_key_compute*1000:.2f}ms")
        logger.info(f"   Total time saved by pre-computing: {total_saved_time:.2f}ms")
        logger.info(f"   Efficiency improvement: {total_saved_time / (total_time*1000) * 100:.1f}% of total time")

        # Per-layer analysis - show slowest layers
        wait_times = detailed_metrics["per_layer_wait_times"]
        retrieve_times = detailed_metrics["per_layer_retrieve_times"]

        if wait_times:
            # record all >1s
            slowest_wait_layers = [
                (layer_id, wait_time)
                for layer_id, wait_time in wait_times.items()
                if wait_time > 1000
            ]
            slowest_wait_layers.sort(key=lambda x: x[1], reverse=True)
            logger.info(f"\n⏱️  SLOWEST WAIT TIMES:")
            for layer_id, wait_time in slowest_wait_layers:
                logger.info(f"   Layer {layer_id}: {wait_time:.2f}ms wait")
            total_wait_time = sum(wait_times.values())
            logger.info(f"   Total wait time: {total_wait_time:.2f}ms")

        if retrieve_times:
            slowest_retrieve_layers = [
                (layer_id, retrieve_time)
                for layer_id, retrieve_time in retrieve_times.items()
                if retrieve_time > 1000
            ]
            slowest_retrieve_layers.sort(key=lambda x: x[1], reverse=True)
            logger.info(f"\n📥 SLOWEST RETRIEVE TIMES:")
            for layer_id, retrieve_time in slowest_retrieve_layers:
                logger.info(
                    f"   Layer {layer_id}: {retrieve_time:.2f}ms retrieve")
            total_retrieve_time = sum(retrieve_times.values())
            logger.info(f"   Total retrieve time: {total_retrieve_time:.2f}ms")

    # Verification summary
    passed = sum(1 for r in verification_results.values() if "PASS" in r)
    logger.info(
        f"   Data verification: {passed}/{len(verification_results)} layers passed"
    )

    # Calculate receiver throughput
    if total_time > 0 and len(tokens) > 0:
        # Calculate data size based on KV cache dimensions
        kv_shape = engine.gpu_connector.get_shape(config.chunk_size)
        element_size = torch.tensor([], dtype=metadata.kv_dtype).element_size()
        chunk_size_bytes = torch.prod(
            torch.tensor(kv_shape)).item() * element_size

        # Total data received = chunks × layers × bytes_per_chunk
        total_chunks_received = len(processed_layers) * (len(tokens) //
                                                         config.chunk_size)
        total_bytes_received = total_chunks_received * chunk_size_bytes

        receiver_throughput = calculate_throughput(total_bytes_received,
                                                   total_time)
        logger.info(f"   === Receiver Throughput Analysis ===")
        logger.info(f"   Total tokens processed: {len(tokens)}")
        logger.info(f"   Total chunks received: {total_chunks_received}")
        logger.info(
            f"   Total data received: {total_bytes_received / (1024**3):.3f} GB"
        )
        logger.info(f"   Total time: {total_time:.3f}s")
        logger.info(f"   Receiver throughput: {receiver_throughput:.2f} GB/s")

    # Calculate latency benefit vs sequential processing
    if len(processed_layers) > 1:
        sequential_time = metrics['avg_time_per_layer_ms'] * len(
            processed_layers)
        actual_time_ms = total_time * 1000
        parallel_benefit = (
            (sequential_time - actual_time_ms) / sequential_time) * 100

        # Detailed calculation breakdown
        logger.info(f"   === Latency Benefit Calculation ===")
        logger.info(
            f"   Avg time per layer: {metrics['avg_time_per_layer_ms']:.2f}ms")
        logger.info(f"   Number of processed layers: {len(processed_layers)}")
        logger.info(
            f"   Estimated sequential time: {metrics['avg_time_per_layer_ms']:.2f} × {len(processed_layers)} = {sequential_time:.2f}ms"
        )
        logger.info(f"   Actual parallel time: {actual_time_ms:.2f}ms")
        logger.info(
            f"   Time difference: {sequential_time:.2f} - {actual_time_ms:.2f} = {sequential_time - actual_time_ms:.2f}ms"
        )
        logger.info(
            f"   Benefit calculation: ({sequential_time - actual_time_ms:.2f} / {sequential_time:.2f}) × 100 = {parallel_benefit:.1f}%"
        )
        if parallel_benefit < 0:
            slowdown_factor = actual_time_ms / sequential_time
            logger.info(
                f"   🚨 PERFORMANCE WARNING: Parallel processing is {slowdown_factor:.1f}x SLOWER than sequential!"
            )

        # Diagnose the issue
        if detailed_metrics["status"] != "no_data":
            phase_breakdown = detailed_metrics["phase_breakdown"]
            wait_time_ms = phase_breakdown['total_wait_time_ms']
            total_time_ms = actual_time_ms

            logger.info(f"\n🩺 PERFORMANCE DIAGNOSIS:")
            if wait_time_ms > total_time_ms * 0.8:
                logger.info(
                    f"   Issue: Excessive waiting time ({wait_time_ms:.1f}ms / {total_time_ms:.1f}ms = {wait_time_ms/total_time_ms*100:.1f}%)"
                )
                logger.info(
                    f"   Likely cause: Network latency, sender not transmitting layers fast enough, or timeout issues"
                )
                logger.info(
                    f"   Fix: Check sender throughput, reduce timeouts, or optimize RDMA transfer"
                )
            elif phase_breakdown[
                    'total_retrieve_time_ms'] > total_time_ms * 0.5:
                logger.info(f"   Issue: Slow data retrieval")
                logger.info(
                    f"   Likely cause: GPU memory bottleneck or storage backend latency"
                )
                logger.info(
                    f"   Fix: Optimize GPU connector or storage backend")
            elif len(processed_layers) < metadata.kv_shape[0] * 0.5:
                logger.info(
                    f"   Issue: Many layers failed to process ({len(processed_layers)}/{metadata.kv_shape[0]})"
                )
                logger.info(f"   Likely cause: Timeouts or data corruption")
                logger.info(
                    f"   Fix: Increase timeout values or check data integrity")
            else:
                logger.info(
                    f"   Issue: General overhead - layers processed sequentially instead of in parallel"
                )
                logger.info(
                    f"   Likely cause: Lock contention or blocking operations")
                logger.info(
                    f"   Fix: Profile thread contention and optimize synchronization"
                )

        logger.info(
            f"   Latency benefit vs sequential: {parallel_benefit:.1f}%")

    # Cleanup
    engine.close()

    return {
        "processed_layers": len(processed_layers),
        "first_layer_latency_ms": metrics['first_layer_latency_ms'],
        "total_time_ms": total_time * 1000,
        "avg_time_per_layer_ms": metrics['avg_time_per_layer_ms'],
        "verification_passed": passed,
        "verification_total": len(verification_results),
        "pre_compute_time_ms": pre_compute_time * 1000
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Test LayerAware Cache Engine for Disaggregated Inference')
    parser.add_argument('--role',
                        type=str,
                        required=True,
                        choices=['sender', 'receiver'],
                        help='Role of this instance')
    parser.add_argument('--host',
                        type=str,
                        default='localhost',
                        help='Host name/IP for connection')
    parser.add_argument('--port',
                        type=int,
                        default=5555,
                        help='Port number for connection')
    parser.add_argument('--num-chunks',
                        type=int,
                        default=8,
                        help='Number of chunks to send')
    parser.add_argument('--num-rounds',
                        type=int,
                        default=1,
                        help='Number of rounds to run')
    parser.add_argument('--debug',
                        action='store_true',
                        help='Enable detailed debugging output')
    parser.add_argument('--retrieval-method',
                        type=str,
                        choices=['individual', 'bulk'],
                        default='individual',
                        help='Retrieval method for receiver: individual (with pre-computed keys) or bulk progressive')

    args = parser.parse_args()

    # Set random seeds for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    # Create configuration and metadata
    config = create_config(args.role, args.host, args.port)
    config.nixl_enable_gc = True
    metadata = create_metadata()

    # Setup test data
    num_blocks = 20000
    block_size = 16
    dtype = torch.bfloat16
    device = "cuda"

    max_chunks = num_blocks * block_size // config.chunk_size
    assert args.num_chunks <= max_chunks, f"Max chunks: {max_chunks}"

    # Generate test data (reused across rounds)
    tokens = generate_test_tokens(args.num_chunks, config.chunk_size)
    slot_indices = list(range(0, num_blocks * block_size))
    random.shuffle(slot_indices)
    slot_mapping = torch.tensor(slot_indices[:len(tokens)], device=device)

    # Create KV cache data once (reused across rounds)
    if args.role == "sender":
        kv_cache = generate_kv_cache_paged_list_tensors(
            num_blocks, device, block_size, dtype)
        kv_cache = fill_kv_cache_with_layer_pattern(kv_cache, slot_mapping)
    else:
        retrieved_cache = generate_kv_cache_paged_list_tensors(
            num_blocks, device, block_size, dtype)

    # Run test rounds
    results = []

    for round_num in range(args.num_rounds):
        logger.info(f"\n{'='*80}")
        logger.info(f"🏁 ROUND {round_num + 1}/{args.num_rounds}")
        logger.info(f"{'='*80}")

        # Create a fresh LayerAwareLMCacheEngine for each round
        # This is necessary because engine.close() closes ZMQ sockets and other resources
        # that cannot be reused
        engine = create_layer_aware_engine(config, metadata, args.num_chunks)

        if args.role == "sender":
            result = run_sender(args, config, metadata, tokens, kv_cache,
                                slot_mapping, engine)
            results.append({"throughput_gb_s": result})

        else:  # receiver
            result = run_receiver(args, config, metadata, tokens,
                                  retrieved_cache, slot_mapping, engine)
            if result:
                results.append(result)

        # Wait between rounds
        if round_num < args.num_rounds - 1:
            time.sleep(2)

    # Print summary
    if results:
        logger.info(f"\n{'='*80}")
        logger.info("📈 SUMMARY STATISTICS")
        logger.info(f"{'='*80}")

        if args.role == "sender":
            throughputs = [r["throughput_gb_s"] for r in results]
            logger.info(
                f"Mean throughput: {np.mean(throughputs):.2f} ± {np.std(throughputs):.2f} GB/s"
            )
            logger.info(
                f"Min/Max throughput: {min(throughputs):.2f} / {max(throughputs):.2f} GB/s"
            )

        else:  # receiver
            first_latencies = [r["first_layer_latency_ms"] for r in results]
            total_times = [r["total_time_ms"] for r in results]

            logger.info(
                f"Mean first layer latency: {np.mean(first_latencies):.2f} ± {np.std(first_latencies):.2f} ms"
            )
            logger.info(
                f"Mean total completion time: {np.mean(total_times):.2f} ± {np.std(total_times):.2f} ms"
            )
            logger.info(
                f"Mean verification success rate: {np.mean([r['verification_passed']/r['verification_total'] for r in results])*100:.1f}%"
            )

    logger.info("🎉 Test completed!")
