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

#ifndef XLA_BACKENDS_GPU_TESTS_COLLECTIVE_SM_STALL_KERNELS_H_
#define XLA_BACKENDS_GPU_TESTS_COLLECTIVE_SM_STALL_KERNELS_H_

#include <cstddef>
#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace xla::gpu {

// Allocates zeroed counters in host memory mapped into every device of the
// process: `num_ranks` arrival counters, then `num_ranks` counters of blocks
// that gave up.
absl::StatusOr<uint32_t*> AllocateArrivalCounters(int num_ranks);

// Launches on `stream` (a CUstream) a persistent kernel with as many blocks as
// the `sms` SMs of the device can hold at once, and returns that number. Every
// block arrives on `counters[rank]` and waits until the blocks of every rank
// arrived: like any grid-wide barrier, it only completes if all its blocks are
// resident at the same time on every rank. A block that waits more than 10
// seconds gives up and counts itself in `counters[num_ranks + rank]`. The
// kernel then copies `count` elements from `src` to `dst`.
absl::StatusOr<int> LaunchGridBarrier(void* stream, int sms, uint32_t* counters,
                                      int rank, int num_ranks,
                                      const uint32_t* src, uint32_t* dst,
                                      size_t count);

// Launches on `stream` a single block that spins for `nanoseconds` and then
// copies `count` elements from `src` to `dst`.
absl::Status LaunchDelay(void* stream, uint64_t nanoseconds,
                         const uint32_t* src, uint32_t* dst, size_t count);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_TESTS_COLLECTIVE_SM_STALL_KERNELS_H_
