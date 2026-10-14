#pragma once
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cstdint>

__global__ void set_if_any(cudaGraphConditionalHandle handle,
                          const int32_t* values, int count, bool positive) {
  bool active = false;
  for (int i = threadIdx.x; i < count; i += blockDim.x)
    active |= (positive ? values[i] > 0 : values[i] != 0);
  const int any = __syncthreads_or(active);
  if (threadIdx.x == 0) cudaGraphSetConditional(handle, any != 0);
}

__global__ void set_if_any_not_in(cudaGraphConditionalHandle handle,
                                 const int32_t* values, int count,
                                 int inactive0, int inactive1) {
  bool active = false;
  for (int i = threadIdx.x; i < count; i += blockDim.x) {
    const int value = values[i];
    active |= value != inactive0 && value != inactive1;
  }
  const int any = __syncthreads_or(active);
  if (threadIdx.x == 0) cudaGraphSetConditional(handle, any != 0);
}

// The supplied child graph is immutable and its tensor allocations must remain
// alive. cudaGraphAddChildGraphNode copies its topology into the owning graph.
void attach_conditional(torch::Tensor values, bool positive, uint64_t body_address,
                        bool exclude_inactive, int inactive0, int inactive1) {
  TORCH_CHECK(values.is_cuda() && values.is_contiguous() &&
              values.scalar_type() == torch::kInt && values.numel() > 0,
              "conditional predicate must be contiguous CUDA int32");
  TORCH_CHECK(values.numel() <= INT32_MAX && body_address != 0,
              "invalid conditional body or predicate length");
  c10::cuda::CUDAGuard guard(values.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  cudaStreamCaptureStatus status;
  cudaGraph_t graph;
  C10_CUDA_CHECK(cudaStreamGetCaptureInfo(stream, &status, nullptr, &graph,
                                         nullptr, nullptr, nullptr));
  TORCH_CHECK(status == cudaStreamCaptureStatusActive,
              "conditional attachment requires active stream capture");
  cudaGraphConditionalHandle handle;
  C10_CUDA_CHECK(cudaGraphConditionalHandleCreate(
      &handle, graph, 0, cudaGraphCondAssignDefault));
  if (exclude_inactive)
    set_if_any_not_in<<<1, 256, 0, stream>>>(
        handle, values.data_ptr<int32_t>(), int(values.numel()), inactive0, inactive1);
  else
    set_if_any<<<1, 256, 0, stream>>>(handle, values.data_ptr<int32_t>(),
                                     int(values.numel()), positive);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const cudaGraphNode_t* dependencies;
  const cudaGraphEdgeData* edges;
  size_t count;
  C10_CUDA_CHECK(cudaStreamGetCaptureInfo(stream, &status, nullptr, &graph,
                                         &dependencies, &edges, &count));
  cudaGraphNodeParams params{};
  params.type = cudaGraphNodeTypeConditional;
  params.conditional.handle = handle;
  params.conditional.type = cudaGraphCondTypeIf;
  params.conditional.size = 1;
  cudaGraphNode_t conditional, child;
  C10_CUDA_CHECK(cudaGraphAddNode(&conditional, graph, dependencies, edges,
                                 count, &params));
  C10_CUDA_CHECK(cudaGraphAddChildGraphNode(
      &child, params.conditional.phGraph_out[0], nullptr, 0,
      reinterpret_cast<cudaGraph_t>(body_address)));
  C10_CUDA_CHECK(cudaStreamUpdateCaptureDependencies(
      stream, &conditional, nullptr, 1, cudaStreamSetCaptureDependencies));
}

void attach_if_any(torch::Tensor values, bool positive, uint64_t body_address) {
  attach_conditional(values, positive, body_address, false, 0, 0);
}

void attach_if_any_not_in(torch::Tensor values, int inactive0, int inactive1,
                         uint64_t body_address) {
  // Every value outside the explicitly inactive set executes the original body,
  // including malformed/unknown budgets. Original error handling stays inside it.
  attach_conditional(values, false, body_address, true, inactive0, inactive1);
}
