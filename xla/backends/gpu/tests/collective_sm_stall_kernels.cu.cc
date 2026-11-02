/* Copyright 2026 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/backends/gpu/tests/collective_sm_stall_kernels.h"

#include <cstddef>
#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/cuda/cuda_status.h"

namespace xla::gpu {
namespace {

constexpr int kThreadsPerBlock = 1024;

// A block that waits longer gives up, so that a stall ends and shows in
// traces.
constexpr uint64_t kTimeoutNs = 10'000'000'000;

__device__ uint64_t GlobalTimer() {
  uint64_t time;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(time));
  return time;
}

__global__ void GridBarrier(uint32_t* counters, int rank, int num_ranks,
                            const uint32_t* src, uint32_t* dst, size_t count) {
  if (threadIdx.x == 0) {
    // Counters only grow: the barrier of execution `n` waits for `n + 1`
    // grids on every rank.
    uint32_t arrived = atomicAdd_system(&counters[rank], 1);
    uint32_t target = (arrived / gridDim.x + 1) * gridDim.x;
    uint64_t start = GlobalTimer();
    for (int peer = 0; peer < num_ranks; ++peer) {
      while (atomicAdd_system(&counters[peer], 0) < target) {
        if (GlobalTimer() - start > kTimeoutNs) {
          atomicAdd_system(&counters[num_ranks + rank], 1);
          peer = num_ranks;
          break;
        }
        __nanosleep(256);
      }
    }
  }
  __syncthreads();
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += gridDim.x * blockDim.x) {
    dst[i] = src[i];
  }
}

__global__ void Delay(uint64_t nanoseconds, const uint32_t* src, uint32_t* dst,
                      size_t count) {
  for (uint64_t start = GlobalTimer(); GlobalTimer() - start < nanoseconds;) {
    __nanosleep(1000);
  }
  for (size_t i = threadIdx.x; i < count; i += blockDim.x) {
    dst[i] = src[i];
  }
}

}  // namespace

absl::StatusOr<uint32_t*> AllocateArrivalCounters(int num_ranks) {
  void* counters = nullptr;
  ABSL_RETURN_IF_ERROR(stream_executor::cuda::ToStatus(
      cudaHostAlloc(&counters, 2 * num_ranks * sizeof(uint32_t),
                    cudaHostAllocMapped | cudaHostAllocPortable)));
  for (int i = 0; i < 2 * num_ranks; ++i) {
    static_cast<uint32_t*>(counters)[i] = 0;
  }
  return static_cast<uint32_t*>(counters);
}

absl::StatusOr<int> LaunchGridBarrier(void* stream, int sms, uint32_t* counters,
                                      int rank, int num_ranks,
                                      const uint32_t* src, uint32_t* dst,
                                      size_t count) {
  int blocks_per_sm;
  ABSL_RETURN_IF_ERROR(stream_executor::cuda::ToStatus(
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, GridBarrier,
                                                    kThreadsPerBlock, 0)));
  int blocks = sms * blocks_per_sm;
  GridBarrier<<<blocks, kThreadsPerBlock, 0,
                static_cast<cudaStream_t>(stream)>>>(counters, rank, num_ranks,
                                                     src, dst, count);
  ABSL_RETURN_IF_ERROR(stream_executor::cuda::ToStatus(cudaGetLastError()));
  return blocks;
}

absl::Status LaunchDelay(void* stream, uint64_t nanoseconds,
                         const uint32_t* src, uint32_t* dst, size_t count) {
  Delay<<<1, 256, 0, static_cast<cudaStream_t>(stream)>>>(nanoseconds, src, dst,
                                                          count);
  return stream_executor::cuda::ToStatus(cudaGetLastError());
}

}  // namespace xla::gpu
