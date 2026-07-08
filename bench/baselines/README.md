# Benchmark baselines

Reference `bench_engine` output captured on known-good hardware. Used by
`.github/workflows/perf.yml` for regression *visibility* — not (yet) a
hard gate.

## Files

- `macos-arm64.json` — captured on Apple Silicon (M-series), Homebrew
  RocksDB 11.1.2, Release build, using the stabilization flags
  (10 repetitions × 1s min-time + warmup + random interleaving — see
  §"Regenerating baselines" below).
- `linux-arm64.json` — captured on Linux ARM64 inside Docker on Apple
  Silicon, Ubuntu 24.04 with `librocksdb-dev` 8.x, same stabilization
  flags plus `--cpus=4 --memory=8g` Docker resource envelope. Numbers
  here are informational — see the "Interpreting CV" note about Docker
  jitter.

Linux x86_64 baseline will be added when a bare-metal x86 runner is
available. Docker on macOS with Rosetta 2 is slow and noisy enough that
a committed baseline would mislead more than inform.

## What the CI perf gate does today

The `perf` workflow runs `bench_engine` on every push/PR and uploads the
resulting JSON as a workflow artifact. It does **not** compare against
these baselines automatically — GitHub-hosted shared runner variance
(~20% on identical workloads) is too high to gate on.

The maintainer's manual workflow is:

1. See a suspicious PR-vs-main perf delta in the workflow log.
2. Download both JSON artifacts.
3. Diff against `bench/baselines/*.json` to check whether it's a real
   regression or shared-runner noise.
4. If real, block the PR manually.

## When this graduates to a hard gate

When a dedicated (non-shared) runner is available, `perf.yml` gets
uncommented at the bottom and switches to `--fail-on-regression` mode.
The threshold today would be 10% on the reference metrics
(`BM_QueryNearestN/60000/128`, `BM_AppendEngine/128`).

## Regenerating baselines

### macOS ARM64 (native)

```sh
cmake -S . -B build-rel \
  -DCMAKE_BUILD_TYPE=Release \
  -DCHRONOSV_BUILD_BENCH=ON \
  -DCHRONOSV_BUILD_TESTS=OFF
cmake --build build-rel --target bench_engine -j

# Quiesce the machine first — close browsers, background sync, etc.
# The stabilization flags are the same as capture_baseline.sh uses for Linux.
./build-rel/bench/bench_engine \
  --benchmark_min_time=1.0s \
  --benchmark_min_warmup_time=0.5 \
  --benchmark_repetitions=10 \
  --benchmark_enable_random_interleaving=true \
  --benchmark_report_aggregates_only=true \
  --benchmark_out=bench/baselines/macos-arm64.json \
  --benchmark_out_format=json \
  --benchmark_time_unit=us

git add bench/baselines/macos-arm64.json
git commit -m "bench: refresh macOS ARM64 baseline"
```

Takes ~2 minutes. The stabilization flags matter: earlier versions used
`--benchmark_repetitions=3 --benchmark_min_time=0.3s` which produced
useless CVs (>60% on some sub-microsecond append benchmarks). See §
"Interpreting CV" below.

### Linux ARM64 (Docker)

```sh
./docker/capture_baseline.sh              # ARM64 native on Apple Silicon
./docker/capture_baseline.sh --x86        # x86_64 via Rosetta (slower, noisier)
```

Wraps the same stabilization flags plus `--cpus=4 --memory=8g` to
reserve a consistent Docker resource envelope. Prerequisites match
`docker/verify_linux.sh`: Docker running, Eigen submodule populated.

### When to refresh

- After landing a perf-relevant change (kernel rewrite, ring buffer refactor).
- When a compiler major version changes (AppleClang 21 → 22, GCC 13 → 14).
- **Not casually** — a moving baseline defeats the point.

## Interpreting CV

Google Benchmark reports coefficient of variation (`_cv`) as
`stddev / mean` — a rough noise-to-signal ratio for the repetitions.

**Rule-of-thumb ceilings on macOS ARM64 (native) with the stabilization flags:**

| Benchmark class | Acceptable CV | Alarm CV |
|---|---|---|
| Sub-microsecond append (dim ≤ 128) | ≤ 5% | > 15% |
| Larger appends (dim 512-1024) | ≤ 3% | > 10% |
| Query at ≤ 60k × 128 (in-cache) | ≤ 3% | > 10% |
| Query at 100k × 128 (memory-bandwidth-bound) | ≤ 8% | > 20% |
| Query at ≥ 60k × 512 (memory-bandwidth-bound) | ≤ 8% | > 20% |

**Linux ARM64 via Docker on macOS: add 2× tolerance to all rows.** The VM
adds jitter for memory-bound workloads especially. Numbers here are
informational, not gate-worthy — for real Linux regression detection
you need a bare-metal Linux runner.

**If a CV is above the alarm threshold**, the sample is not stable
enough to compare against. Either the machine has other work running,
the Docker resources are constrained, or (for sub-µs benches) fewer
repetitions or shorter min-time are being used. Do not commit such a
baseline; treat the number as informational only.

**Why sub-microsecond benchmarks are noisier**: at ~0.3 µs per append,
even a single L1-miss or scheduler tick adds visible variance to the
sample. Google Benchmark auto-scales iterations to reach `min_time`
but the aggregation across `repetitions` still shows scheduler jitter.
This is a measurement precision floor, not a real workload issue.

**Important — CV of averaged reps ≠ per-call latency.** Each rep-mean is
already the average of ~400,000 individual `append()` iterations (at
`min_time=1s`, ~2.5 µs/call). A 15% CV on `BM_AppendEngine/512` means
"the sample mean drifted 15% across the ten 1-second measurement
windows" — *not* "individual calls sometimes take 15% longer." For the
product-relevant question ("does append have a P99 latency tail?"), use
`BM_AppendLatencyDist` — it times each call individually and reports
`p50_ns` / `p99_ns` / `p999_ns` / `p99_over_p50` as counters. A healthy
result is P99/P50 < 2×; anything > 3× indicates a cache-miss or scheduler
story worth investigating.

## What lives here vs. what doesn't

- **Here:** `bench_engine.json` per architecture, captured on known hardware.
- **Not here:** `bench_compare.json` (competitor comparison — that's a
  point-in-time story documented in the README, not a regression signal).
- **Not here:** `bench_kernels.json` (the microbench numbers). Same reason —
  if kernel micros drift 25%, the `bench_engine` end-to-end catches it
  through the code path that actually matters.
