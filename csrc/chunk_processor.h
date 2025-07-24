// csrc/chunk_processor.h
#pragma once

#include <pybind11/pybind11.h>
#include <torch/torch.h>
#include <vector>
#include <tuple>

namespace py = pybind11;

// Function declarations for chunk processing
std::tuple<std::vector<torch::Tensor>, std::vector<py::object>> 
process_chunks(const torch::Tensor& combined_tensor,
                   const std::vector<std::pair<int64_t, int64_t>>& slice_positions);

std::tuple<std::vector<torch::Tensor>, std::vector<py::object>> 
process_chunks_contiguous(const torch::Tensor& combined_tensor,
                         const std::vector<std::pair<int64_t, int64_t>>& slice_positions);

std::vector<torch::Tensor> 
batch_slice_tensors(const torch::Tensor& combined_tensor,
                   const std::vector<std::pair<int64_t, int64_t>>& slice_positions);