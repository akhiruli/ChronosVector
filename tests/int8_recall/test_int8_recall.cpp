/*
 * INT8 recall validation harness for ChronosVector.
 *
 * Measures Recall@10 of the INT8 storage path vs the FP32 baseline on a
 * user-supplied dataset in a simple binary format. Prints a summary
 * table with recall, mean query latency, and hot-memory size for both
 * storage dtypes.
 *
 * Requires:
 *   - CHRONOSV_ENABLE_INT8=ON at build time
 *   - A prepared dataset directory containing:
 *       <dir>/train.fbin   uint32 count | uint32 dim | count*dim float32
 *       <dir>/test.fbin    uint32 count | uint32 dim | count*dim float32
 *       <dir>/gt.ibin      uint32 count | uint32 dim | count*dim int32
 *         (gt row i is the ground-truth top-K neighbor indices of test[i]
 *          against train, dim = K, sorted best-first)
 *
 * Usage:
 *   test_int8_recall <dataset_dir> <cosine|euclidean>
 *
 * See tests/int8_recall/README.md for prep scripts (SIFT-1M, BERT) and
 * reference numbers.
 */
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "chronosv/chronos_vector.h"

namespace {

template <typename T>
struct BinFile {
    std::uint32_t count = 0;
    std::uint32_t dim   = 0;
    std::unique_ptr<T[]> data;
};

template <typename T>
BinFile<T> load_bin(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "ERROR: can't open %s\n", path.c_str());
        std::exit(2);
    }
    BinFile<T> b;
    f.read(reinterpret_cast<char*>(&b.count), 4);
    f.read(reinterpret_cast<char*>(&b.dim),   4);
    const std::size_t n = static_cast<std::size_t>(b.count) * b.dim;
    b.data = std::make_unique<T[]>(n);
    f.read(reinterpret_cast<char*>(b.data.get()), n * sizeof(T));
    if (!f) {
        std::fprintf(stderr, "ERROR: short read on %s\n", path.c_str());
        std::exit(2);
    }
    std::printf("  loaded %s: count=%u dim=%u\n", path.c_str(), b.count, b.dim);
    return b;
}

std::uint64_t next_pow2(std::uint64_t x) {
    std::uint64_t p = 64;
    while (p < x) p <<= 1;
    return p;
}

chronosv_engine_t* make_engine(std::uint32_t dim,
                               std::uint64_t ring_cap,
                               chronosv_dtype_t dtype,
                               chronosv_metric_t metric) {
    chronosv_config_t cfg{};
    cfg.abi_version           = CHRONOSV_ABI_VERSION;
    cfg.dim                   = dim;
    cfg.ring_capacity         = ring_cap;
    cfg.window_duration_ms    = 100LL * 3600 * 1000;  /* huge — no eviction */
    cfg.eviction_interval_ms  = 3600 * 1000;
    cfg.storage_dtype         = static_cast<std::uint8_t>(dtype);
    cfg.distance_metric       = static_cast<std::uint8_t>(metric);
    /* No cold_path — pure in-memory. */
    chronosv_error_t err = CHRONOSV_OK;
    chronosv_engine_t* h = chronosv_create(&cfg, &err);
    if (!h) {
        std::fprintf(stderr, "chronosv_create failed: err=%d (%s)\n",
                     err, chronosv_error_string(err));
        std::exit(3);
    }
    return h;
}

void ingest(chronosv_engine_t* eng, const BinFile<float>& train, const char* label) {
    std::printf("  ingesting %u vectors into %s engine...", train.count, label);
    std::fflush(stdout);
    const auto t0 = std::chrono::steady_clock::now();
    for (std::uint32_t i = 0; i < train.count; ++i) {
        chronosv_error_t err = chronosv_append(
            eng, "s", static_cast<std::int64_t>(i),
            train.data.get() + i * train.dim,
            train.dim, nullptr);
        if (err != CHRONOSV_OK) {
            std::fprintf(stderr, "\nappend %u failed: %d\n", i, err);
            std::exit(3);
        }
    }
    const auto dt = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    std::printf(" %.2f s (%.1fk/s)\n", dt, train.count / dt / 1000);
}

double compute_recall(const std::vector<std::int64_t>& retrieved,
                      const std::int32_t* gt,
                      int k) {
    std::unordered_set<std::int32_t> gt_set(gt, gt + k);
    int hits = 0;
    for (int i = 0; i < k && i < static_cast<int>(retrieved.size()); ++i) {
        if (gt_set.count(static_cast<std::int32_t>(retrieved[i]))) ++hits;
    }
    return static_cast<double>(hits) / k;
}

struct QueryStats {
    double total_recall     = 0.0;
    double sum_latency_us   = 0.0;
    std::size_t n           = 0;
};

QueryStats run_queries(chronosv_engine_t* eng,
                       const BinFile<float>& test,
                       const BinFile<std::int32_t>& gt,
                       int top_k,
                       const char* label) {
    std::printf("  running %u queries against %s engine (top-%d)...",
                test.count, label, top_k);
    std::fflush(stdout);
    std::vector<std::int64_t> out_ts(top_k);
    std::vector<float>        out_sc(top_k);
    int result_count = 0;

    QueryStats stats;
    const auto t0 = std::chrono::steady_clock::now();
    for (std::uint32_t q = 0; q < test.count; ++q) {
        const auto q0 = std::chrono::steady_clock::now();
        chronosv_error_t err = chronosv_query_nearest_n(
            eng, "s", test.data.get() + q * test.dim, test.dim, top_k,
            out_ts.data(), out_sc.data(), &result_count);
        const auto q1 = std::chrono::steady_clock::now();
        if (err != CHRONOSV_OK && err != CHRONOSV_WARN_PARTIAL_RESULT) {
            std::fprintf(stderr, "\nquery %u failed: %d\n", q, err);
            std::exit(3);
        }
        stats.sum_latency_us += std::chrono::duration<double, std::micro>(q1 - q0).count();

        std::vector<std::int64_t> retrieved(out_ts.begin(),
                                            out_ts.begin() + result_count);
        stats.total_recall += compute_recall(
            retrieved, gt.data.get() + q * gt.dim, top_k);
        ++stats.n;
    }
    const auto dt = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    std::printf(" %.2f s\n", dt);
    return stats;
}

void usage(const char* argv0) {
    std::fprintf(stderr,
        "Usage: %s <dataset_dir> <cosine|euclidean>\n"
        "\n"
        "  dataset_dir  directory containing train.fbin, test.fbin, gt.ibin\n"
        "  metric       distance metric — must match how ground truth was computed\n"
        "\n"
        "See tests/int8_recall/README.md for prep scripts and reference numbers.\n",
        argv0);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) { usage(argv[0]); return 2; }
    const std::string dir = argv[1];
    const std::string metric_arg = argv[2];

    chronosv_metric_t metric;
    if      (metric_arg == "cosine")    metric = CHRONOSV_METRIC_COSINE;
    else if (metric_arg == "euclidean") metric = CHRONOSV_METRIC_EUCLIDEAN;
    else { usage(argv[0]); return 2; }

    std::printf("=== ChronosVector INT8 recall validation ===\n");
    std::printf("Dataset: %s\n", dir.c_str());
    std::printf("Metric:  %s\n\n", metric_arg.c_str());

    std::printf("Loading data...\n");
    auto train = load_bin<float>(dir + "/train.fbin");
    auto test  = load_bin<float>(dir + "/test.fbin");
    auto gt    = load_bin<std::int32_t>(dir + "/gt.ibin");

    if (train.dim != test.dim) {
        std::fprintf(stderr, "\nERROR: train.dim (%u) != test.dim (%u)\n",
                     train.dim, test.dim);
        return 2;
    }
    if (test.count != gt.count) {
        std::fprintf(stderr, "\nERROR: test.count (%u) != gt.count (%u)\n",
                     test.count, gt.count);
        return 2;
    }
    std::printf("\n");

    const std::uint64_t ring_cap = next_pow2(train.count);
    std::printf("Ring capacity: %llu slots (next pow2 of %u)\n\n",
                static_cast<unsigned long long>(ring_cap), train.count);

    std::printf("--- FP32 engine ---\n");
    chronosv_engine_t* eng_f32 = make_engine(train.dim, ring_cap, CHRONOSV_DTYPE_FLOAT32, metric);
    ingest(eng_f32, train, "FP32");

    std::printf("\n--- INT8 engine ---\n");
    chronosv_engine_t* eng_i8 = make_engine(train.dim, ring_cap, CHRONOSV_DTYPE_INT8, metric);
    ingest(eng_i8, train, "INT8");

    constexpr int kTopK = 10;
    std::printf("\n--- Recall@%d vs ground truth ---\n", kTopK);
    auto s_f32 = run_queries(eng_f32, test, gt, kTopK, "FP32");
    auto s_i8  = run_queries(eng_i8,  test, gt, kTopK, "INT8");

    const double f32_recall = s_f32.total_recall / s_f32.n;
    const double i8_recall  = s_i8.total_recall  / s_i8.n;
    const double gap_pp     = f32_recall - i8_recall;
    const double gap_rel    = gap_pp / f32_recall * 100.0;

    std::printf("\n=========== RESULTS ===========\n");
    std::printf("  FP32  Recall@%d  = %.4f   mean latency %.1f us\n",
                kTopK, f32_recall, s_f32.sum_latency_us / s_f32.n);
    std::printf("  INT8  Recall@%d  = %.4f   mean latency %.1f us\n",
                kTopK, i8_recall,  s_i8.sum_latency_us  / s_i8.n);
    std::printf("  Recall drop: %.2f percentage points (%.2f%% relative)\n",
                gap_pp * 100.0, gap_rel);

    chronosv_stats_t st_f32{}, st_i8{};
    chronosv_get_stats(eng_f32, &st_f32);
    chronosv_get_stats(eng_i8,  &st_i8);
    std::printf("\n  Hot memory:\n");
    std::printf("    FP32: %.1f MiB\n", st_f32.hot_bytes / (1024.0 * 1024.0));
    std::printf("    INT8: %.1f MiB  (%.1fx smaller)\n",
                st_i8.hot_bytes / (1024.0 * 1024.0),
                static_cast<double>(st_f32.hot_bytes) / st_i8.hot_bytes);
    std::printf("===============================\n");

    chronosv_destroy(eng_f32);
    chronosv_destroy(eng_i8);
    return 0;
}
