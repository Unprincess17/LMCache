/*
 * C++ extension for high-performance tensor chunk processing in LMCache.
 * 
 * This module bypasses Python's GIL limitations by:
 * 1. Releasing the GIL during computation-heavy operations
 * 2. Using C++ threading for true parallelism  
 * 3. Minimizing Python object creation overhead
 * 
 * The key insight is that tensor slicing and metadata computation
 * are CPU-bound tasks that can benefit significantly from parallel processing
 * when not constrained by the GIL.
 */
 #include "chunk_processor.h"
 #include <torch/extension.h>
 #include <pybind11/pybind11.h>
 #include <pybind11/stl.h>
 #include <vector>
 #include <tuple>
 #include <thread>
 #include <future>
 #include <algorithm>
 
 namespace py = pybind11;
 
 /*
 * Process multiple tensor chunks in parallel with GIL released.
 * 
 * This function:
 * 1. Releases the Python GIL using py::gil_scoped_release
 * 2. Creates tensor slices in parallel using C++ threads
 * 3. Computes metadata for each chunk
 * 4. Returns results for Python to create memory objects
 */
std::tuple<std::vector<torch::Tensor>, std::vector<py::object>>
process_chunks(const torch::Tensor& combined_tensor, 
               const std::vector<std::pair<int64_t, int64_t>>& slice_positions) {
     
     const size_t num_chunks = slice_positions.size();
     const size_t num_threads = std::min(static_cast<size_t>(16), num_chunks);
     
     // Pre-allocate result vectors
     std::vector<torch::Tensor> chunk_tensors(num_chunks);
     std::vector<std::tuple<std::vector<int64_t>, std::string, uintptr_t, size_t>> metadata_data(num_chunks);
     
     // Pre-compute common values
     const size_t element_size = combined_tensor.element_size();
     const auto dtype = combined_tensor.dtype();
     std::string dtype_str;
     
     // Convert PyTorch dtype to string
     if (dtype == torch::kFloat32) {
         dtype_str = "float32";
     } else if (dtype == torch::kFloat16) {
         dtype_str = "float16";
     } else if (dtype == torch::kBFloat16) {
         dtype_str = "bfloat16";
     } else if (dtype == torch::kInt32) {
         dtype_str = "int32";
     } else if (dtype == torch::kInt64) {
         dtype_str = "int64";
     } else {
         dtype_str = "float32";  // Default fallback
     }
     
     // Release GIL for parallel processing
     {
        pybind11::gil_scoped_release release;
         
         // Worker function for processing chunks
         auto process_chunk_range = [&](size_t start_idx, size_t end_idx) {
             for (size_t i = start_idx; i < end_idx; ++i) {
                 const auto& [slice_start, slice_end] = slice_positions[i];
                 
                 // Create tensor slice (this is the expensive operation)
                 torch::Tensor chunk_tensor = combined_tensor.slice(0, slice_start, slice_end);
                 chunk_tensors[i] = chunk_tensor;
                 
                 // Compute metadata
                 auto sizes = chunk_tensor.sizes();
                 std::vector<int64_t> shape(sizes.begin(), sizes.end());
                 uintptr_t address = reinterpret_cast<uintptr_t>(chunk_tensor.data_ptr());
                 size_t phy_size = chunk_tensor.numel() * element_size;
                 
                 metadata_data[i] = std::make_tuple(shape, dtype_str, address, phy_size);
             }
         };
         
         // Launch parallel workers
         if (num_chunks > 16 && num_threads > 1) {
             // Use threading for large chunk counts
             std::vector<std::future<void>> futures;
             const size_t chunks_per_thread = (num_chunks + num_threads - 1) / num_threads;
             
             for (size_t t = 0; t < num_threads; ++t) {
                 size_t start_idx = t * chunks_per_thread;
                 size_t end_idx = std::min(start_idx + chunks_per_thread, num_chunks);
                 
                 if (start_idx < end_idx) {
                     futures.emplace_back(
                         std::async(std::launch::async, process_chunk_range, start_idx, end_idx)
                     );
                 }
             }
             
             // Wait for all threads to complete
             for (auto& future : futures) {
                 future.wait();
             }
         } else {
             // Process sequentially for small chunk counts to avoid thread overhead
             process_chunk_range(0, num_chunks);
         }
     }
     // GIL is automatically reacquired here
     
     // Convert metadata to py::object format
     std::vector<py::object> metadata_objects;
     metadata_objects.reserve(num_chunks);
     for (size_t i = 0; i < num_chunks; ++i) {
         const auto& [shape, dtype_str, address, phy_size] = metadata_data[i];
         // Create tuple of (shape, dtype_str, address, phy_size) as expected by Python
         metadata_objects.emplace_back(py::make_tuple(shape, dtype_str, address, phy_size));
     }
     
     return std::make_tuple(chunk_tensors, metadata_objects);
 }
 
 /*
 * Optimized version for contiguous memory processing.
 * Uses memory-mapped operations where possible.
 */
std::tuple<std::vector<torch::Tensor>, std::vector<py::object>>
process_chunks_contiguous(const torch::Tensor& combined_tensor,
                         const std::vector<std::pair<int64_t, int64_t>>& slice_positions) {
     
     // Ensure tensor is contiguous for optimal memory access
     torch::Tensor contiguous_tensor = combined_tensor.contiguous();
     
     const size_t num_chunks = slice_positions.size();
     std::vector<torch::Tensor> chunk_tensors;
     std::vector<py::object> metadata_objects;
     
     chunk_tensors.reserve(num_chunks);
     metadata_objects.reserve(num_chunks);
     
     // Pre-compute common values
     const size_t element_size = contiguous_tensor.element_size();
     const auto dtype = contiguous_tensor.dtype();
     std::string dtype_str;
     
     // Convert PyTorch dtype to string
     if (dtype == torch::kFloat32) {
         dtype_str = "float32";
     } else if (dtype == torch::kFloat16) {
         dtype_str = "float16";
     } else if (dtype == torch::kBFloat16) {
         dtype_str = "bfloat16";
     } else if (dtype == torch::kInt32) {
         dtype_str = "int32";
     } else if (dtype == torch::kInt64) {
         dtype_str = "int64";
     } else {
         dtype_str = "float32";  // Default fallback
     }
     
     // Release GIL for the bulk operations
     {
        pybind11::gil_scoped_release release;
         
         // Process all chunks with minimal overhead
         for (const auto& [slice_start, slice_end] : slice_positions) {
             // Use narrow for better performance with contiguous tensors
             torch::Tensor chunk_tensor = contiguous_tensor.narrow(0, slice_start, slice_end - slice_start);
             
             chunk_tensors.emplace_back(std::move(chunk_tensor));
             
             // Compute metadata for this chunk
             auto sizes = chunk_tensor.sizes();
             std::vector<int64_t> shape(sizes.begin(), sizes.end());
             uintptr_t address = reinterpret_cast<uintptr_t>(chunk_tensor.data_ptr());
             size_t phy_size = chunk_tensor.numel() * element_size;
             
             // Create tuple of (shape, dtype_str, address, phy_size) as expected by Python
             metadata_objects.emplace_back(py::make_tuple(shape, dtype_str, address, phy_size));
         }
     }
     
     return std::make_tuple(chunk_tensors, metadata_objects);
 }
 
 /*
  * Batch tensor slicing with memory pre-allocation for maximum efficiency.
  */
 std::vector<torch::Tensor> batch_slice_tensors(const torch::Tensor& source,
                                               const std::vector<std::pair<int64_t, int64_t>>& slices) {
     std::vector<torch::Tensor> results;
     results.reserve(slices.size());
     
     // Release GIL for batch operations
     {
        pybind11::gil_scoped_release release;
         
         for (const auto& [start, end] : slices) {
             results.emplace_back(source.slice(0, start, end));
         }
     }
     
     return results;
 }