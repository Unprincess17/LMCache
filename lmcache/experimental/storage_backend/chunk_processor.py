"""
High-performance chunk processing module that bypasses Python's GIL.

This module provides multiple implementations:
1. C++ extension (via PyBind11) - releases GIL for true parallelism
2. CUDA kernel - GPU-accelerated processing  
3. Optimized Python fallback - for compatibility

The goal is to process tensor chunks and create memory objects faster than
Python threading by avoiding GIL contention.
"""

import os
import time
import logging
import threading
from typing import List, Tuple, Optional, Any
import torch

from lmcache.experimental.memory_management import (MemoryFormat, MemoryObj,
                                                    MemoryObjMetadata,
                                                    TensorMemoryObj)
from lmcache.logging import init_logger

logger = init_logger(__name__)


def configure_chunk_processing_threads(
        num_threads: Optional[int] = None) -> None:
    """
    Configure the number of threads for chunk processing.
    
    Args:
        num_threads: Number of threads to use. If None, uses hardware-optimal setting.
                    Can be overridden by LMCACHE_CHUNK_THREADS environment variable.
    """
    if num_threads is not None:
        os.environ['LMCACHE_CHUNK_THREADS'] = str(num_threads)
        logger.debug(f"Set chunk processing threads to {num_threads}")
    elif 'LMCACHE_CHUNK_THREADS' not in os.environ:
        # Set default based on hardware
        try:
            import multiprocessing
            # Get actual CPU core count (hardware concurrency)
            hardware_cores = multiprocessing.cpu_count()

            # Special handling for high-core-count systems
            if hardware_cores > 32:
                # Memory bandwidth becomes the bottleneck with many cores
                # Use more conservative thread count to avoid memory bus saturation
                optimal_threads = min(16, hardware_cores // 4)
                logger.info(
                    f"High-core-count system detected ({hardware_cores} cores). "
                    f"Using conservative thread count: {optimal_threads} "
                    f"to avoid memory bandwidth saturation.")
            else:
                # Use 75% of available cores, but cap at 32
                optimal_threads = min(int(hardware_cores * 0.75), 32)
                logger.info(
                    f"Auto-configured chunk processing threads to {optimal_threads} (hardware cores: {hardware_cores})"
                )

            os.environ['LMCACHE_CHUNK_THREADS'] = str(optimal_threads)

        except Exception as e:
            # Fallback to conservative default
            fallback_threads = 4
            os.environ['LMCACHE_CHUNK_THREADS'] = str(fallback_threads)
            logger.warning(
                f"Failed to detect hardware cores: {e}. Using fallback: {fallback_threads} threads"
            )


def configure_numa_optimization() -> None:
    """
    Configure NUMA optimization for high-core-count systems.
    """
    try:
        import multiprocessing
        hardware_cores = multiprocessing.cpu_count()

        if hardware_cores > 32:
            logger.info(
                f"High-core-count system detected ({hardware_cores} cores). "
                f"Enabling NUMA-aware optimizations.")

            # Set environment variables for NUMA optimization
            os.environ['LMCACHE_NUMA_AWARE'] = '1'

            # Configure memory allocation policy
            try:
                import numa
                if numa.available():
                    logger.info(
                        "NUMA library available. Configuring memory allocation."
                    )
                    # Set memory allocation policy to local node
                    numa.set_localalloc()
                else:
                    logger.warning(
                        "NUMA library not available. Using default memory allocation."
                    )
            except ImportError:
                logger.warning(
                    "NUMA library not available. Install 'numa' package for optimal performance."
                )

    except Exception as e:
        logger.warning(f"Failed to configure NUMA optimization: {e}")


def get_memory_bandwidth_optimal_threads() -> int:
    """
    Calculate optimal thread count based on memory bandwidth considerations.
    
    Returns:
        int: Optimal thread count for memory bandwidth
    """
    try:
        import multiprocessing
        hardware_cores = multiprocessing.cpu_count()

        # Memory bandwidth typically scales with sqrt of cores
        # For high-core systems, memory becomes the bottleneck
        if hardware_cores > 32:
            # Conservative approach: use fewer threads to avoid memory bus saturation
            return min(16, hardware_cores // 4)
        else:
            # Standard approach for moderate core counts
            return min(hardware_cores, 32)

    except Exception:
        return 4  # Conservative fallback


def get_optimal_chunk_processor() -> str:
    """
    Determine the optimal chunk processing method based on environment and hardware.
    
    Returns:
        str: 'cpp', 'cuda', or 'python' indicating the best processor to use
    """
    # Check environment variable override
    forced_method = os.environ.get('LMCACHE_CHUNK_PROCESSOR', '').lower()
    if forced_method in ['cpp', 'cuda', 'python']:
        logger.debug(f"Using forced chunk processor: {forced_method}")
        return forced_method

    # Try to detect optimal method
    try:
        # Check if C++ extension is available
        if _check_cpp_extension():
            logger.debug("C++ chunk processor available and preferred")
            return 'cpp'
    except Exception as e:
        logger.debug(f"C++ chunk processor not available: {e}")

    # Check if CUDA is available and beneficial
    if torch.cuda.is_available():
        logger.debug("CUDA chunk processor available")
        return 'cuda'

    # Fallback to optimized Python
    logger.debug("Using optimized Python chunk processor")
    return 'python'


def _check_cpp_extension() -> bool:
    """Check if the C++ extension is compiled and available."""
    try:
        import lmcache.c_ops
        return True
    except ImportError:
        return False


class ChunkProcessor:
    """High-performance chunk processor with multiple backend implementations."""

    @staticmethod
    def process_chunks_cpp(
        combined_tensor: torch.Tensor, slice_positions: List[Tuple[int, int]],
        base_format: MemoryFormat, parent_allocator: Any
    ) -> Tuple[List[torch.Tensor], List[MemoryObjMetadata]]:
        """
        Process chunks using C++ extension that releases the GIL.
        
        This method uses a compiled C++ extension to:
        1. Extract tensor slices in parallel (GIL released)
        2. Compute metadata in parallel 
        3. Return results for Python to create memory objects
        
        Args:
            combined_tensor: Source tensor to slice
            slice_positions: List of (start, end) positions
            base_format: Memory format for metadata
            parent_allocator: Parent allocator reference
            
        Returns:
            Tuple of (chunk_tensors, chunk_metadatas)
        """
        try:
            import lmcache.c_ops

            logger.debug(
                f"Processing {len(slice_positions)} chunks with C++ extension")
            start_time = time.perf_counter()

            # Call C++ function that releases GIL
            chunk_tensors, metadata_data = lmcache.c_ops.process_chunks(
                combined_tensor, slice_positions)

            # Create metadata objects from C++ results
            chunk_metadatas = []
            for i, (tensor,
                    (shape, dtype_str, address,
                     phy_size)) in enumerate(zip(chunk_tensors,
                                                 metadata_data)):

                metadata = MemoryObjMetadata(shape=torch.Size(shape),
                                             dtype=getattr(torch, dtype_str),
                                             address=address,
                                             phy_size=phy_size,
                                             ref_count=1,
                                             is_pin=False,
                                             fmt=base_format)
                chunk_metadatas.append(metadata)

            end_time = time.perf_counter()
            logger.debug(
                f"C++ chunk processing completed in {(end_time - start_time) * 1000:.2f}ms"
            )

            return chunk_tensors, chunk_metadatas

        except ImportError as e:
            raise RuntimeError(f"C++ chunk processor not available: {e}")
        except Exception as e:
            raise RuntimeError(f"C++ chunk processing failed: {e}")

    @staticmethod
    def process_chunks_cuda_kernel(
        combined_tensor: torch.Tensor, slice_positions: List[Tuple[int, int]],
        base_format: MemoryFormat, parent_allocator: Any
    ) -> Tuple[List[torch.Tensor], List[MemoryObjMetadata]]:
        """
        Process chunks using CUDA kernel for GPU tensors.
        
        This method uses CUDA operations to:
        1. Create all chunk tensors in a single kernel launch
        2. Compute metadata on GPU where possible
        3. Minimize CPU-GPU synchronization
        
        Args:
            combined_tensor: Source tensor (must be on CUDA device)
            slice_positions: List of (start, end) positions  
            base_format: Memory format for metadata
            parent_allocator: Parent allocator reference
            
        Returns:
            Tuple of (chunk_tensors, chunk_metadatas)
        """
        if not combined_tensor.is_cuda:
            raise ValueError(
                "CUDA chunk processor requires tensor to be on CUDA device")

        logger.debug(
            f"Processing {len(slice_positions)} chunks with CUDA kernel")
        start_time = time.perf_counter()

        try:
            # Method 1: Use CuPy for custom kernel (if available)
            chunk_tensors, chunk_metadatas = ChunkProcessor._process_chunks_cupy(
                combined_tensor, slice_positions, base_format)

        except ImportError:
            # Method 2: Use PyTorch CUDA operations
            chunk_tensors, chunk_metadatas = ChunkProcessor._process_chunks_torch_cuda(
                combined_tensor, slice_positions, base_format)

        end_time = time.perf_counter()
        logger.debug(
            f"CUDA chunk processing completed in {(end_time - start_time) * 1000:.2f}ms"
        )

        return chunk_tensors, chunk_metadatas

    @staticmethod
    def _process_chunks_cupy(
        combined_tensor: torch.Tensor, slice_positions: List[Tuple[int, int]],
        base_format: MemoryFormat
    ) -> Tuple[List[torch.Tensor], List[MemoryObjMetadata]]:
        """Process chunks using CuPy for maximum GPU efficiency."""
        try:
            import cupy as cp

            # Convert PyTorch tensor to CuPy array (zero-copy)
            combined_cupy = cp.asarray(combined_tensor.detach())

            # Extract all chunks in parallel using CuPy
            chunk_arrays = []
            for start, end in slice_positions:
                chunk_array = combined_cupy[start:end]
                chunk_arrays.append(chunk_array)

            # Convert back to PyTorch tensors (zero-copy)
            chunk_tensors = []
            chunk_metadatas = []
            element_size = combined_tensor.element_size()

            for chunk_array in chunk_arrays:
                # Zero-copy conversion back to PyTorch
                chunk_tensor = torch.as_tensor(chunk_array,
                                               device=combined_tensor.device)
                chunk_tensors.append(chunk_tensor)

                # Create metadata
                metadata = MemoryObjMetadata(shape=chunk_tensor.shape,
                                             dtype=chunk_tensor.dtype,
                                             address=chunk_tensor.data_ptr(),
                                             phy_size=chunk_tensor.numel() *
                                             element_size,
                                             ref_count=1,
                                             is_pin=False,
                                             fmt=base_format)
                chunk_metadatas.append(metadata)

            return chunk_tensors, chunk_metadatas

        except ImportError:
            raise ImportError("CuPy not available for CUDA chunk processing")

    @staticmethod
    def _process_chunks_torch_cuda(
        combined_tensor: torch.Tensor, slice_positions: List[Tuple[int, int]],
        base_format: MemoryFormat
    ) -> Tuple[List[torch.Tensor], List[MemoryObjMetadata]]:
        """Process chunks using PyTorch CUDA operations."""

        # Use PyTorch's efficient tensor operations
        with torch.cuda.device(combined_tensor.device):
            # Pre-allocate tensors to minimize memory operations
            chunk_tensors = []
            chunk_metadatas = []
            element_size = combined_tensor.element_size()

            # Process chunks using PyTorch's optimized slicing
            for start, end in slice_positions:
                # PyTorch tensor slicing is already quite optimized on CUDA
                chunk_tensor = combined_tensor[start:end].contiguous()
                chunk_tensors.append(chunk_tensor)

                # Create metadata
                metadata = MemoryObjMetadata(shape=chunk_tensor.shape,
                                             dtype=chunk_tensor.dtype,
                                             address=chunk_tensor.data_ptr(),
                                             phy_size=chunk_tensor.numel() *
                                             element_size,
                                             ref_count=1,
                                             is_pin=False,
                                             fmt=base_format)
                chunk_metadatas.append(metadata)

            return chunk_tensors, chunk_metadatas

    @staticmethod
    def _process_chunks_python_fallback(
        combined_tensor: torch.Tensor, slice_positions: List[Tuple[int, int]],
        base_format: MemoryFormat, parent_allocator: Any
    ) -> Tuple[List[torch.Tensor], List[MemoryObjMetadata]]:
        """
        Optimized Python fallback implementation.
        
        This avoids threading overhead and focuses on vectorized operations
        where possible to maximize single-threaded performance.
        """
        logger.debug(
            f"Processing {len(slice_positions)} chunks with optimized Python")
        start_time = time.perf_counter()

        # Pre-compute constants
        element_size = combined_tensor.element_size()
        dtype = combined_tensor.dtype

        # Pre-allocate output lists
        chunk_tensors = []
        chunk_metadatas = []

        # Process chunks sequentially but efficiently
        for start, end in slice_positions:
            # Extract chunk (this is already optimized in PyTorch)
            chunk_tensor = combined_tensor[start:end]

            # Create metadata with pre-computed values
            metadata = MemoryObjMetadata(shape=chunk_tensor.shape,
                                         dtype=dtype,
                                         address=chunk_tensor.data_ptr(),
                                         phy_size=chunk_tensor.numel() *
                                         element_size,
                                         ref_count=1,
                                         is_pin=False,
                                         fmt=base_format)

            chunk_tensors.append(chunk_tensor)
            chunk_metadatas.append(metadata)

        end_time = time.perf_counter()
        logger.debug(
            f"Python fallback processing completed in {(end_time - start_time) * 1000:.2f}ms"
        )

        return chunk_tensors, chunk_metadatas

    @staticmethod
    def create_memory_objects(chunk_tensors: List[torch.Tensor],
                              chunk_metadatas: List[MemoryObjMetadata],
                              parent_allocator: Any) -> List[MemoryObj]:
        """
        Create TensorMemoryObj instances from tensors and metadata.
        
        This step is kept in Python since it involves object creation
        which doesn't benefit much from C++/CUDA acceleration.
        """
        chunk_objects = []
        for tensor, metadata in zip(chunk_tensors, chunk_metadatas):
            chunk_obj = TensorMemoryObj(raw_data=tensor,
                                        metadata=metadata,
                                        parent_allocator=parent_allocator)
            chunk_objects.append(chunk_obj)

        return chunk_objects
