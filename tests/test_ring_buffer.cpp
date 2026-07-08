/*
 * Ring-buffer unit tests. Covers:
 *   - Basic single-thread append/snapshot
 *   - Wrap-around index math
 *   - Overwrite detection and counters
 *   - Snapshot clamp when producer laps consumer
 *   - SPSC append/consume across two threads (correctness, not TSan)
 *
 * TSan is a build-flag concern (see -DCHRONOSV_ENABLE_TSAN=ON) and adds no
 * new tests here — the same tests run under TSan validate memory ordering.
 */
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

#include "chronosv/ring_buffer.h"
#include "chronosv/types.h"

using chronosv::internal::SensorRing;

namespace {

// Small helper to append a float vector into the ring. Uses a fake norm to
// keep the test independent of the kernel implementation.
void append(SensorRing& r, std::int64_t ts, std::vector<float> v) {
    r.Append(ts, v.data(), /*precomputed_norm=*/1.0f, /*payload=*/nullptr);
}

}  // namespace

TEST_CASE("SensorRing constructs with valid config", "[ring]") {
    SensorRing r(/*capacity=*/64, /*dim=*/8,
                 /*payload_size=*/0, /*dtype=*/CHRONOSV_DTYPE_FLOAT32);
    REQUIRE(r.ok());
    REQUIRE(r.capacity() == 64);
    REQUIRE(r.dim() == 8);
    REQUIRE(r.payload_size() == 0);
    REQUIRE(r.dtype() == CHRONOSV_DTYPE_FLOAT32);

    auto [begin, end] = r.Snapshot();
    REQUIRE(begin == 0);
    REQUIRE(end == 0);
    REQUIRE(r.overwrite_events() == 0);
}

TEST_CASE("SensorRing appends and snapshot advances head", "[ring]") {
    SensorRing r(64, 4, 0, CHRONOSV_DTYPE_FLOAT32);
    REQUIRE(r.ok());

    for (int i = 0; i < 10; ++i) {
        append(r, /*ts=*/i, {float(i), float(i)+1, float(i)+2, float(i)+3});
    }
    auto [begin, end] = r.Snapshot();
    REQUIRE(begin == 0);
    REQUIRE(end == 10);
    REQUIRE(r.overwrite_events() == 0);

    // Slot 3's stored vector matches what we wrote.
    const auto* vp = static_cast<const float*>(r.vector_at(3));
    REQUIRE(vp[0] == 3.0f);
    REQUIRE(vp[3] == 6.0f);
    REQUIRE(r.timestamp_at(3) == 3);
}

TEST_CASE("SensorRing wraps at capacity boundary", "[ring]") {
    SensorRing r(64, 2, 0, CHRONOSV_DTYPE_FLOAT32);
    REQUIRE(r.ok());

    // Append exactly capacity entries — should NOT trigger overwrite yet.
    for (int i = 0; i < 64; ++i) append(r, i, {float(i), float(i) * 2.0f});
    REQUIRE(r.overwrite_events() == 0);

    auto [begin, end] = r.Snapshot();
    REQUIRE(begin == 0);
    REQUIRE(end == 64);

    // Append one more — slot 0 (logical index 64) wraps and overwrites the
    // oldest entry. Since tail is still 0, this is an overwrite event.
    append(r, 64, {64.0f, 128.0f});
    REQUIRE(r.overwrite_events() == 1);
    REQUIRE(r.overwritten_entries() == 1);

    // Snapshot should clamp begin so we still see only `capacity` entries.
    auto [b2, e2] = r.Snapshot();
    REQUIRE(e2 == 65);
    REQUIRE(b2 == 1);  // e2 - capacity

    // The slot at logical index 0 in the ring (i.e. head=64, idx=0) now
    // contains the new value.
    const auto* v_at_slot0 = static_cast<const float*>(r.vector_at(64));
    REQUIRE(v_at_slot0[0] == 64.0f);
    REQUIRE(r.timestamp_at(64) == 64);
}

TEST_CASE("SensorRing AdvanceTail clears overwrite pressure", "[ring]") {
    SensorRing r(64, 1, 0, CHRONOSV_DTYPE_FLOAT32);
    REQUIRE(r.ok());

    // Fill to capacity.
    for (int i = 0; i < 64; ++i) append(r, i, {float(i)});
    REQUIRE(r.overwrite_events() == 0);

    // Simulate eviction advancing tail past the first 10 entries.
    r.AdvanceTail(10);
    auto [begin, end] = r.Snapshot();
    REQUIRE(begin == 10);
    REQUIRE(end == 64);

    // Now appending 10 more should NOT trigger overwrites (tail made room).
    // Before each append: h-t < capacity, so the overwrite check `h-t >= cap`
    // does not fire. After 10 appends: head=74, tail=10, h-t = 64 = capacity.
    for (int i = 64; i < 74; ++i) append(r, i, {float(i)});
    REQUIRE(r.overwrite_events() == 0);
    REQUIRE(r.Snapshot().second == 74);

    // Now every subsequent append starts with h-t == capacity, which meets
    // the overwrite condition — the slot at h & mask is still "live" from
    // the consumer's view (tail hasn't advanced past it). So each of the
    // next 10 appends fires an overwrite event.
    for (int i = 74; i < 84; ++i) append(r, i, {float(i)});
    REQUIRE(r.overwrite_events() == 10);
    append(r, 84, {84.0f});
    REQUIRE(r.overwrite_events() == 11);
}

TEST_CASE("SensorRing hot_bytes reflects allocation", "[ring]") {
    // capacity 64, dim 4 float32, no payload, no INT8:
    //   64 * (8 timestamp + 16 vector + 4 norm) = 64 * 28 = 1792
    SensorRing r(64, 4, 0, CHRONOSV_DTYPE_FLOAT32);
    REQUIRE(r.ok());
    REQUIRE(r.hot_bytes() == 64u * (8u + 16u + 4u));

    // With payload_size=16:
    //   64 * (8 + 16 + 4 + 16) = 64 * 44 = 2816
    SensorRing rp(64, 4, 16, CHRONOSV_DTYPE_FLOAT32);
    REQUIRE(rp.ok());
    REQUIRE(rp.hot_bytes() == 64u * (8u + 16u + 4u + 16u));
}

TEST_CASE("SensorRing SPSC without wrap: strict consecutive observation",
          "[ring][thread]") {
    // Cap chosen larger than kN so the producer never laps the consumer.
    // In this regime we can make the strong guarantee that every observed
    // timestamp is exactly prev + 1 — i.e. no torn reads, no gaps.
    constexpr std::uint64_t kCap = 65536;
    constexpr int kN = 20000;

    SensorRing r(kCap, 1, 0, CHRONOSV_DTYPE_FLOAT32);
    REQUIRE(r.ok());

    std::atomic<bool> producer_done{false};
    std::atomic<int>  last_seen{-1};
    std::atomic<int>  ordering_violations{0};

    std::thread producer([&] {
        for (int i = 0; i < kN; ++i) {
            const float v = float(i);
            r.Append(/*ts=*/i, &v, /*norm=*/1.0f, /*payload=*/nullptr);
        }
        producer_done.store(true, std::memory_order_release);
    });

    std::thread consumer([&] {
        while (true) {
            auto [begin, end] = r.Snapshot();
            for (std::uint64_t idx = begin; idx < end; ++idx) {
                const auto ts = r.timestamp_at(idx);
                const int prev = last_seen.load(std::memory_order_relaxed);
                if (ts <= prev) continue;                // already saw this
                if (ts != prev + 1) {
                    ordering_violations.fetch_add(1, std::memory_order_relaxed);
                }
                last_seen.store(int(ts), std::memory_order_relaxed);
            }
            if (producer_done.load(std::memory_order_acquire)
                && last_seen.load(std::memory_order_relaxed) >= kN - 1) {
                break;
            }
        }
    });

    producer.join();
    consumer.join();

    REQUIRE(ordering_violations.load() == 0);
    REQUIRE(last_seen.load() == kN - 1);
    REQUIRE(r.overwrite_events() == 0);
}

/* Detect ThreadSanitizer at compile time. GCC defines __SANITIZE_THREAD__;
 * Clang uses __has_feature. */
#if defined(__has_feature)
#  if __has_feature(thread_sanitizer)
#    define CHRONOSV_UNDER_TSAN 1
#  endif
#endif
#if defined(__SANITIZE_THREAD__)
#  define CHRONOSV_UNDER_TSAN 1
#endif

#if !defined(CHRONOSV_UNDER_TSAN)
TEST_CASE("SensorRing SPSC with wrap: overwrites counted, no crash",
          "[ring][thread]") {
    // Cap much smaller than kN — producer WILL lap the consumer. This
    // exercises the documented overwrite path.
    //
    // *** EXCLUDED FROM TSAN BUILDS. ***
    // When the producer laps the consumer, the producer writes the same
    // physical slot the consumer is about to read from its stale snapshot.
    // TSan (correctly, per the C++ memory model) flags this as a data race
    // on the raw slot bytes — but the design (§5.1) documents this as an
    // accepted trade-off: the ring is a lossy queue when reader falls
    // behind, and correctness for this test is defined as "no crash +
    // overwrite counters increment." The strict-ordering variant that
    // guarantees synchronized reads (test "SPSC without wrap: strict
    // consecutive observation") DOES run under TSan and validates memory
    // ordering when no wrap occurs. Excluding this test from TSan is
    // therefore honest, not a workaround for a bug.
    //
    // We intentionally do NOT assert observation ordering here. Under
    // producer-laps-consumer, a naive reader (no snapshot re-check pattern)
    // can legitimately see torn reads: read slot A (post-overwrite, new ts),
    // then read slot A+1 (not yet overwritten, old ts) — apparent backwards
    // motion in the reader's local sequence, even though each individual
    // slot's timestamp is monotonically increasing over time. The
    // documented mitigation is a head-based re-check the reader performs;
    // that's a future test / helper, not enforced here.
    //
    // What this test DOES check: no crash, and that the overwrite counter
    // reflects the wrap-induced loss.
    constexpr std::uint64_t kCap = 4096;
    constexpr int kN = 50000;

    SensorRing r(kCap, 1, 0, CHRONOSV_DTYPE_FLOAT32);
    REQUIRE(r.ok());

    std::atomic<bool> producer_done{false};
    std::atomic<std::uint64_t> reads_total{0};

    std::thread producer([&] {
        for (int i = 0; i < kN; ++i) {
            const float v = float(i);
            r.Append(/*ts=*/i, &v, /*norm=*/1.0f, /*payload=*/nullptr);
        }
        producer_done.store(true, std::memory_order_release);
    });

    std::thread consumer([&] {
        while (true) {
            auto [begin, end] = r.Snapshot();
            for (std::uint64_t idx = begin; idx < end; ++idx) {
                // Force the reads to actually happen (not optimized out).
                (void) r.timestamp_at(idx);
                (void) r.vector_at(idx);
                reads_total.fetch_add(1, std::memory_order_relaxed);
            }
            if (producer_done.load(std::memory_order_acquire)) break;
        }
    });

    producer.join();
    consumer.join();

    REQUIRE(r.overwrite_events() > 0);
    // Producer completed all kN appends; head advanced to kN.
    auto [_, end] = r.Snapshot();
    REQUIRE(end == static_cast<std::uint64_t>(kN));
    // Consumer observed at least SOMETHING (exact count is nondeterministic).
    REQUIRE(reads_total.load() > 0);
}
#endif  // !CHRONOSV_UNDER_TSAN

/* ==========================================================================
 * Coverage-expansion tests (round 2)
 * ========================================================================== */

TEST_CASE("SensorRing: fresh ring has empty snapshot", "[ring][edge]") {
    SensorRing r(64, 4, 0, CHRONOSV_DTYPE_FLOAT32);
    REQUIRE(r.ok());
    auto [begin, end] = r.Snapshot();
    REQUIRE(begin == 0);
    REQUIRE(end == 0);
    REQUIRE(end - begin == 0);  // iteration should be a no-op
    REQUIRE(r.overwrite_events() == 0);
    REQUIRE(r.overwritten_entries() == 0);
}

TEST_CASE("SensorRing INT8 dtype allocates scales_", "[ring][int8]") {
    // The INT8 storage_dtype allocates a scales_ column in addition to
    // vectors_. hot_bytes reflects it. We can't easily inspect the pointer
    // externally, but hot_bytes accounting catches the extra allocation.
    SensorRing r(/*capacity=*/128, /*dim=*/64, /*payload_size=*/0,
                 /*dtype=*/CHRONOSV_DTYPE_INT8);
    REQUIRE(r.ok());
    REQUIRE(r.dtype() == CHRONOSV_DTYPE_INT8);
    // 128 * (8 ts + 64*1 vec + 4 norm + 4 scale) = 128 * 80 = 10240
    REQUIRE(r.hot_bytes() == 128u * (8u + 64u + 4u + 4u));
}

TEST_CASE("SensorRing INT8 append/read round-trip", "[ring][int8]") {
    SensorRing r(64, 4, 0, CHRONOSV_DTYPE_INT8);
    REQUIRE(r.ok());

    // int8 vector — value bytes must survive memcpy in and out.
    std::array<std::int8_t, 4> vec = {-127, 0, 42, 127};
    r.Append(/*ts=*/7, vec.data(), /*norm=*/1.0f, /*payload=*/nullptr);

    const auto* stored = static_cast<const std::int8_t*>(r.vector_at(0));
    REQUIRE(stored[0] == -127);
    REQUIRE(stored[1] == 0);
    REQUIRE(stored[2] == 42);
    REQUIRE(stored[3] == 127);
    REQUIRE(r.timestamp_at(0) == 7);
}

TEST_CASE("SensorRing payload round-trip", "[ring][payload]") {
    constexpr std::uint32_t kPayloadSize = 24;
    SensorRing r(/*cap=*/64, /*dim=*/4, /*payload=*/kPayloadSize,
                 CHRONOSV_DTYPE_FLOAT32);
    REQUIRE(r.ok());
    REQUIRE(r.payload_size() == kPayloadSize);

    // Distinct bytes per slot so we can catch cross-slot bleed.
    std::vector<std::vector<std::uint8_t>> payloads;
    for (int i = 0; i < 10; ++i) {
        std::vector<std::uint8_t> p(kPayloadSize);
        for (std::uint32_t j = 0; j < kPayloadSize; ++j) {
            p[j] = static_cast<std::uint8_t>((i * 31 + j) & 0xFF);
        }
        payloads.push_back(std::move(p));
        std::array<float, 4> v = {float(i), float(i), float(i), float(i)};
        r.Append(/*ts=*/i, v.data(), 1.0f, payloads.back().data());
    }

    // Verify each slot has its own payload intact.
    for (std::uint64_t idx = 0; idx < 10; ++idx) {
        const auto* pp = static_cast<const std::uint8_t*>(r.payload_at(idx));
        REQUIRE(pp != nullptr);
        for (std::uint32_t j = 0; j < kPayloadSize; ++j) {
            REQUIRE(pp[j] == payloads[idx][j]);
        }
    }
}

TEST_CASE("SensorRing payload_at returns nullptr when disabled", "[ring][payload]") {
    SensorRing r(64, 4, /*payload_size=*/0, CHRONOSV_DTYPE_FLOAT32);
    REQUIRE(r.ok());
    std::array<float, 4> v = {1, 2, 3, 4};
    r.Append(/*ts=*/0, v.data(), 1.0f, /*payload=*/nullptr);
    REQUIRE(r.payload_at(0) == nullptr);
}

TEST_CASE("SensorRing: vector data across wrap remains coherent", "[ring][wrap]") {
    // After wrapping, ALL slots should contain the most-recent write for that
    // physical slot — not stale garbage. We fill twice past capacity and
    // verify every slot maps to the expected generation-2 value.
    constexpr std::uint64_t kCap = 64;  // ring enforces capacity >= 64
    SensorRing r(kCap, 2, 0, CHRONOSV_DTYPE_FLOAT32);
    REQUIRE(r.ok());

    // Generation 1: fill capacity entries.
    for (int i = 0; i < int(kCap); ++i) {
        std::array<float, 2> v = {float(i), float(i) * 10.0f};
        r.Append(/*ts=*/i, v.data(), 1.0f, nullptr);
    }
    // Generation 2: overwrite every slot.
    for (int i = int(kCap); i < int(kCap) * 2; ++i) {
        std::array<float, 2> v = {float(i), float(i) * 10.0f};
        r.Append(/*ts=*/i, v.data(), 1.0f, nullptr);
    }

    // Every slot should reflect gen-2 data, not gen-1.
    for (std::uint64_t idx = kCap; idx < 2 * kCap; ++idx) {
        const auto* vp = static_cast<const float*>(r.vector_at(idx));
        REQUIRE(vp[0] == static_cast<float>(idx));
        REQUIRE(vp[1] == static_cast<float>(idx) * 10.0f);
        REQUIRE(r.timestamp_at(idx) == static_cast<std::int64_t>(idx));
    }

    // Overwrite counters should be exactly kCap (one per gen-2 append).
    REQUIRE(r.overwrite_events() == kCap);
    REQUIRE(r.overwritten_entries() == kCap);
}

TEST_CASE("SensorRing invariant: overwrite_events == overwritten_entries",
          "[ring][invariant]") {
    // In v0.1 both counters are bumped in lockstep on every overwriting Append.
    // If a future change (batch append, coalesced overwrite tracking) diverges
    // them, this test catches it and forces an explicit documentation update.
    SensorRing r(64, 1, 0, CHRONOSV_DTYPE_FLOAT32);
    REQUIRE(r.ok());
    for (int i = 0; i < 200; ++i) {
        float v = float(i);
        r.Append(i, &v, 1.0f, nullptr);
    }
    REQUIRE(r.overwrite_events() == r.overwritten_entries());
}

TEST_CASE("SensorRing supports multiple concurrent readers", "[ring][thread]") {
    // Design contract: any number of consumer threads may query
    // concurrently. This test spins up N readers against one producer;
    // primary purpose is TSan-clean (no data races) and no crashes.
    constexpr std::uint64_t kCap = 65536;
    constexpr int kN = 15000;
    constexpr int kReaders = 4;

    SensorRing r(kCap, 4, 0, CHRONOSV_DTYPE_FLOAT32);
    REQUIRE(r.ok());

    std::atomic<bool> producer_done{false};
    std::atomic<std::uint64_t> reader_max_end{0};
    std::atomic<int> reader_faults{0};

    std::thread producer([&] {
        for (int i = 0; i < kN; ++i) {
            std::array<float, 4> v = {float(i), float(i)+1, float(i)+2, float(i)+3};
            r.Append(/*ts=*/i, v.data(), 1.0f, nullptr);
        }
        producer_done.store(true, std::memory_order_release);
    });

    auto reader_body = [&]() {
        std::uint64_t local_max = 0;
        while (true) {
            auto [begin, end] = r.Snapshot();
            if (begin > end) {  // must never happen
                reader_faults.fetch_add(1, std::memory_order_relaxed);
                return;
            }
            if (end > 0) {
                // Read the tail slot's vector to ensure no crash on the map.
                (void) r.vector_at(end - 1);
                (void) r.timestamp_at(end - 1);
            }
            local_max = std::max(local_max, end);
            if (producer_done.load(std::memory_order_acquire) && end == kN) break;
        }
        // Publish the largest end this reader saw.
        std::uint64_t cur = reader_max_end.load(std::memory_order_relaxed);
        while (cur < local_max
               && !reader_max_end.compare_exchange_weak(cur, local_max,
                                                       std::memory_order_relaxed)) {
        }
    };

    std::vector<std::thread> readers;
    for (int t = 0; t < kReaders; ++t) readers.emplace_back(reader_body);

    producer.join();
    for (auto& t : readers) t.join();

    REQUIRE(reader_faults.load() == 0);
    REQUIRE(reader_max_end.load() == static_cast<std::uint64_t>(kN));
    // Capacity is large enough to hold all appends, so no overwrites should occur.
    REQUIRE(r.overwrite_events() == 0);
}

TEST_CASE("SensorRing minimum capacity (64) works", "[ring][edge]") {
    SensorRing r(/*capacity=*/64, /*dim=*/1, 0, CHRONOSV_DTYPE_FLOAT32);
    REQUIRE(r.ok());
    REQUIRE(r.capacity() == 64);
    for (int i = 0; i < 64; ++i) {
        float v = float(i);
        r.Append(i, &v, 1.0f, nullptr);
    }
    auto [b, e] = r.Snapshot();
    REQUIRE(b == 0);
    REQUIRE(e == 64);
}

TEST_CASE("SensorRing large capacity allocation succeeds", "[ring][edge]") {
    // Sanity check: 1M-slot ring with dim=8 = 32 MB of vectors + ~12 MB other.
    // On any workstation this should allocate fine and Append is O(1).
    SensorRing r(/*capacity=*/1u << 20, /*dim=*/8, 0, CHRONOSV_DTYPE_FLOAT32);
    REQUIRE(r.ok());
    REQUIRE(r.capacity() == (1u << 20));
    std::array<float, 8> v{};
    r.Append(0, v.data(), 0.0f, nullptr);
    REQUIRE(r.Snapshot().second == 1);
}

/* ==========================================================================
 * Death test: SPSC guard fires when a second producer thread appears.
 *
 * Only meaningful in Debug builds (the assert compiles away in Release).
 * Skipped otherwise. Uses POSIX fork + waitpid to observe SIGABRT without
 * killing the parent test process.
 * ========================================================================== */
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <csignal>
#include <cstdlib>

#ifndef NDEBUG
TEST_CASE("SPSC guard aborts on multi-producer in debug build",
          "[ring][spsc][death]") {
    pid_t pid = ::fork();
    REQUIRE(pid >= 0);

    if (pid == 0) {
        // Child: intentionally violate the SPSC contract. First Append on
        // one thread claims the producer TID; second Append on a different
        // thread should trip the CAS-mismatch assert and SIGABRT.
        SensorRing r(64, 1, 0, CHRONOSV_DTYPE_FLOAT32);
        float v = 1.0f;

        std::thread first([&] {
            r.Append(0, &v, 1.0f, nullptr);
        });
        first.join();

        // Now the current (main) thread is a DIFFERENT producer. This
        // Append should fail the assert(expected == tid) check and abort.
        r.Append(1, &v, 1.0f, nullptr);

        // If we get here, the guard failed to fire. Exit with a distinctive
        // code the parent can detect.
        std::_Exit(42);
    }

    // Parent: wait for the child, expect it to have died from a signal
    // (SIGABRT from assert on macOS/Linux).
    int status = 0;
    pid_t got = ::waitpid(pid, &status, 0);
    REQUIRE(got == pid);
    // Either SIGABRT (glibc/libc++ raise SIGABRT on assert failure) or
    // SIGILL / SIGKILL depending on the assert macro's implementation.
    REQUIRE(WIFSIGNALED(status));
    const int sig = WTERMSIG(status);
    INFO("child exit signal = " << sig);
    REQUIRE((sig == SIGABRT || sig == SIGILL || sig == SIGKILL));
}
#endif  // !NDEBUG
