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

// Reproduces a stall between a persistent collective kernel launched by an FFI
// custom call and an NCCL all-reduce that XLA runs concurrently with it in a
// command buffer.
//
// The custom call launches a grid-wide cross-device barrier with as many blocks
// as the GPU can hold, the pattern of persistent collective kernels (one-shot
// all-reduce, expert-parallel dispatch, fused MoE...). Such a kernel completes
// only if all its blocks are resident on every rank at the same time. The
// all-reduce does not depend on it, so the CONCURRENT command buffer runs both
// at the same time. Delay kernels make each rank start a different one first:
//
//   rank 0: barrier (takes every SM)  ->  all-reduce cannot start
//   rank 1: all-reduce (takes SMs)    ->  barrier cannot get all its blocks
//
// The barrier of rank 0 waits for the blocks of rank 1, which wait for SMs held
// by the all-reduce of rank 1, which waits for the all-reduce of rank 0, which
// waits for SMs held by the barrier of rank 0. Nothing progresses until the
// barrier gives up after 10 seconds.
//
// XLA has no way to know the custom call needs every SM of every rank at once,
// so it cannot order it against the all-reduce.

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <ostream>
#include <string>
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/backends/gpu/ffi.h"
#include "xla/backends/gpu/tests/collective_ops_e2e_test_base.h"
#include "xla/backends/gpu/tests/collective_sm_stall_kernels.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/hlo_module_config.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla.pb.h"

namespace xla::gpu {
namespace {

constexpr int kNumRanks = 2;
constexpr int64_t kCount = 1024;
constexpr uint64_t kDelayNs = 200'000'000;

// Counters of the barrier (see `AllocateArrivalCounters`), in host memory: the
// test reads them while the devices are stalled.
uint32_t* counters = nullptr;
std::atomic<int> barrier_blocks[kNumRanks];

// Runs the barrier on every SM but `spare_sms`.
absl::Status GridBarrier(se::Stream* stream, ffi::BufferR1<U32> src,
                         ffi::Result<ffi::BufferR1<U32>> dst,
                         int64_t spare_sms) {
  int rank = stream->parent()->device_ordinal();
  int sms = stream->parent()->GetDeviceDescription().core_count() - spare_sms;
  ABSL_ASSIGN_OR_RETURN(
      int blocks,
      LaunchGridBarrier(stream->platform_specific_handle().stream, sms,
                        counters, rank, kNumRanks, src.typed_data(),
                        dst->typed_data(), src.element_count()));
  barrier_blocks[rank] = blocks;
  return absl::OkStatus();
}

XLA_FFI_DEFINE_HANDLER(kGridBarrier, GridBarrier,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::Stream>()
                           .Arg<ffi::BufferR1<U32>>()
                           .Ret<ffi::BufferR1<U32>>()
                           .Attr<int64_t>("spare_sms"),
                       {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_test$$grid_barrier", "gpu",
                         kGridBarrier);

// Copies `src` to `dst`, after `kDelayNs` on rank `rank` only.
absl::Status Delay(se::Stream* stream, ffi::BufferR1<U32> src,
                   ffi::Result<ffi::BufferR1<U32>> dst, int64_t rank) {
  uint64_t delay = stream->parent()->device_ordinal() == rank ? kDelayNs : 0;
  return LaunchDelay(stream->platform_specific_handle().stream, delay,
                     src.typed_data(), dst->typed_data(), src.element_count());
}

XLA_FFI_DEFINE_HANDLER(kDelay, Delay,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::Stream>()
                           .Arg<ffi::BufferR1<U32>>()
                           .Ret<ffi::BufferR1<U32>>()
                           .Attr<int64_t>("rank"),
                       {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_test$$delay", "gpu",
                         kDelay);

constexpr absl::string_view kHlo = R"(
  HloModule stall, replica_count=2

  add {
    a = u32[] parameter(0)
    b = u32[] parameter(1)
    ROOT sum = u32[] add(a, b)
  }

  ENTRY main {
    p = u32[1024] parameter(0)

    // Rank 1 starts the barrier late.
    late_on_1 = u32[1024] custom-call(p),
      custom_call_target="__xla_test$$delay",
      api_version=API_VERSION_TYPED_FFI, backend_config={rank = 1 : i64}
    barrier = u32[1024] custom-call(late_on_1),
      custom_call_target="__xla_test$$grid_barrier",
      api_version=API_VERSION_TYPED_FFI, backend_config={spare_sms = SPARE_SMS : i64}

    // Rank 0 starts the all-reduce late.
    late_on_0 = u32[1024] custom-call(p),
      custom_call_target="__xla_test$$delay",
      api_version=API_VERSION_TYPED_FFI, backend_config={rank = 0 : i64}
    all-reduce = u32[1024] all-reduce(late_on_0), replica_groups={{0,1}},
      to_apply=add

    ROOT result = (u32[1024], u32[1024]) tuple(barrier, all-reduce)
  }
)";

// Blocks of the barrier of execution `run` that arrived, per rank.
std::string Arrived(int run, int offset = 0) {
  std::string out;
  for (int rank = 0; rank < kNumRanks; ++rank) {
    uint32_t counter =
        reinterpret_cast<volatile uint32_t*>(counters)[offset + rank];
    absl::StrAppend(&out, "  rank ", rank, ": ",
                    counter - run * barrier_blocks[rank], "/",
                    barrier_blocks[rank].load());
  }
  return out;
}

struct Config {
  std::string name;
  DebugOptions::CommandBufferSchedulingMode mode;
  int64_t spare_sms;

  friend void PrintTo(const Config& config, std::ostream* os) {
    *os << config.name;
  }
};

class CollectiveSmStallTest : public CollectiveOpsE2ETestBase,
                              public ::testing::WithParamInterface<Config> {
 public:
  CollectiveSmStallTest()
      : CollectiveOpsE2ETestBase(/*memory_size=*/32 * kMB,
                                 /*collectives_memory_size=*/32 * kMB) {}

  static void SetUpTestSuite() {
    if (counters == nullptr) {
      absl::StatusOr<uint32_t*> allocated = AllocateArrivalCounters(kNumRanks);
      CHECK_OK(allocated);
      counters = *allocated;
    }
  }

  void SetUp() override {
    CollectiveOpsE2ETestBase::SetUp();
    for (int i = 0; i < 2 * kNumRanks; ++i) {
      counters[i] = 0;
    }
  }
};

TEST_P(CollectiveSmStallTest, BarrierAndAllReduce) {
  if (device_count() < kNumRanks) {
    GTEST_SKIP() << "Requires " << kNumRanks << " GPUs";
  }
  HloModuleConfig config = GetModuleConfigForTest(/*replica_count=*/kNumRanks);
  DebugOptions& options = config.mutable_debug_options();
  options.clear_xla_gpu_enable_command_buffer();
  options.add_xla_gpu_enable_command_buffer(DebugOptions::CUSTOM_CALL);
  options.add_xla_gpu_enable_command_buffer(DebugOptions::COLLECTIVES);
  options.set_xla_gpu_graph_min_graph_size(1);
  options.set_xla_gpu_command_buffer_scheduling_mode(GetParam().mode);
  ASSERT_OK_AND_ASSIGN(
      auto module,
      ParseAndReturnVerifiedModule(
          absl::StrReplaceAll(
              kHlo, {{"SPARE_SMS", absl::StrCat(GetParam().spare_sms)}}),
          config));
  ASSERT_OK_AND_ASSIGN(auto executable,
                       CreateExecutable(std::move(module), true));

  Literal input = LiteralUtil::CreateFull<uint32_t>({kCount}, 1);
  Literal barrier = LiteralUtil::CreateFull<uint32_t>({kCount}, 1);
  Literal all_reduce = LiteralUtil::CreateFull<uint32_t>({kCount}, kNumRanks);

  for (int run = 0; run < 3; ++run) {
    absl::Notification done;
    absl::StatusOr<std::vector<Literal>> results;
    std::thread execute([&] {
      results = ExecuteReplicated(executable.get(), {{&input}, {&input}});
      done.Notify();
    });
    if (!done.WaitForNotificationWithTimeout(absl::Seconds(5))) {
      std::string before = Arrived(run);
      absl::SleepFor(absl::Seconds(3));
      fprintf(stderr,
              "run %d not done after 5s, barrier blocks arrived:%s\n"
              "3 seconds later:                       %s\n",
              run, before.c_str(), Arrived(run).c_str());
    }
    execute.join();
    ASSERT_OK(results);
    for (int rank = 0; rank < kNumRanks; ++rank) {
      if (counters[kNumRanks + rank] > 0) {
        FAIL() << "Stall in run " << run
               << ", barrier blocks that gave up after 10s:"
               << Arrived(0, kNumRanks);
      }
    }
    for (const Literal& result : *results) {
      EXPECT_TRUE(LiteralTestUtil::Equal(barrier, LiteralSlice(result, {0})));
      EXPECT_TRUE(
          LiteralTestUtil::Equal(all_reduce, LiteralSlice(result, {1})));
    }
    fprintf(stderr, "run %d done, barrier blocks arrived:%s\n", run,
            Arrived(run).c_str());
  }
}

INSTANTIATE_TEST_SUITE_P(
    SchedulingModes, CollectiveSmStallTest,
    ::testing::Values(Config{"Serialize", DebugOptions::SERIALIZE, 0},
                      Config{"ConcurrentSparingNcclSms",
                             DebugOptions::CONCURRENT, 32},
                      Config{"Concurrent", DebugOptions::CONCURRENT, 0}),
    [](const auto& info) { return info.param.name; });

}  // namespace
}  // namespace xla::gpu
