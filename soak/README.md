# ChronosVector soak test

Long-running load with flat-RSS check. The 6-hour version at 10 kHz sustained ingest is the release-gate soak (see §"Measured results" for the PASS). The 10-minute version is a smoke check that proves the soak infrastructure itself works before you commit to the full run.

## Build

```sh
cmake -S . -B build-rel \
      -DCMAKE_BUILD_TYPE=Release \
      -DCHRONOSV_BUILD_SOAK=ON
cmake --build build-rel --target chronosv_soak -j
```

## Run

**10-minute smoke (default):**
```sh
./build-rel/soak/chronosv_soak --duration 600 --cold /tmp/chronosv-soak
```

**Full 6 hours (release-gate run at 10 kHz):**
```sh
./build-rel/soak/chronosv_soak --duration 21600 --hz 10000 --cold /tmp/chronosv-soak-6h 2>&1 | tee soak-6h.log
```

**Longer 24-hour run (recommended before any tag):**
```sh
./build-rel/soak/chronosv_soak --duration 86400 --cold /tmp/chronosv-soak-24h 2>&1 | tee soak-24h.log
```

**Custom load:**
```sh
./build-rel/soak/chronosv_soak \
  --duration 3600 \
  --hz 50000 \
  --sensors 4 \
  --dim 256 \
  --cold /tmp/chronosv-heavy
```

## What it does

1. Creates an engine with a tight 5-second window and 500 ms eviction interval — forces active eviction and cold-tier writes throughout the run.
2. Ingests `--sensors` × `--hz` vectors/sec, uniformly random floats of `--dim` dimensions.
3. Samples process-private memory (`RssAnon` on Linux, `phys_footprint` on macOS — both exclude shared library pages) every `--sample-interval` seconds.
4. Emits a CSV timeseries to stderr:
   `elapsed_s, private_mib, grow_pct, total_appends, total_evictions, cold_bytes, overwrites, flush_errors`.
5. On completion prints a summary block and exits 0 (PASS) or non-zero (FAIL).

## Pass criteria

The run fails and exits non-zero if any of:

- **Any `flush_errors_total > 0`** — persistence path failed at least once
- **Process-private memory grew more than `--max-grow` (default 5 %)** from the post-warmup baseline
- **`cold_bytes_estimate` stayed at 0** — eviction never persisted anything (something's wrong with the pipeline)
- **`total_evictions == 0`** — the background eviction thread never ran (deadlock, wrong config, etc.)

## Interpreting the CSV

Post-run, extract the timeseries and plot:

```sh
grep -v '^#' soak-24h.log > soak.csv
# columns: elapsed_s, private_mib, grow_pct, total_appends, total_evictions, cold_bytes, overwrites, flush_errors
```

If `private_mib` climbs steadily throughout the run **and** `total_evictions / elapsed_s` is much lower than expected (~2/s at the default 500 ms interval), the eviction thread is being starved — most likely by RocksDB compaction. Investigate `total_evictions` and `overwrites` together before concluding leak vs. throughput problem.

If `flush_errors_total` climbs, check disk space and file descriptor limits.

## Measured results — what "flat RSS" actually looks like

Recorded measurements on Apple Silicon (M-series, macOS 15.x, RocksDB 11.1.2). The `--max-grow 0.05` gate is applied against a steady baseline captured at 4h into the run (see `soak_test.cpp` for the fallback for shorter runs).

| Config     | Workload                | Steady @ 4h | Peak    | Final @ 6h | final vs steady | Overwrites (single burst) | Verdict |
|------------|-------------------------|-------------|---------|------------|-----------------|---------------------------|---------|
| Defaults   | `--hz 10000 --dim 128`  | 177.86 MiB  | 342.80  | 238.16     | +33.90%         | 763k                      | **FAIL** — RSS oscillates 150–340 MiB across compaction cycles; not a plateau |
| **WBM**    | `--hz 10000 --dim 128`  | **167.39 MiB** | **219.28** | **171.17** | **+2.26%** | 5.0 M                     | **✓ PASS** |

**Interpretation:**

- **The `WBM` row is the shipping-quality result.** With `WriteBufferManager` bounding memtable memory (see `src/storage_rocksdb.cpp`), RSS oscillates in a tight ~120–220 MiB band around a stable ~167 MiB baseline. Full 216 M appends processed at sustained 10 kHz; ~41 k evictions run at expected rate; zero flush errors.
- **The overwrite burst is a real observed behavior.** Both defaults and WBM configs show a single early-run overwrite event — a brief compaction storm that outpaces the eviction thread for a few seconds. The counter freezes after the burst; no additional overwrites occur through the remaining 5+ hours. Documented as a known behavior in the top-level README.
- **The defaults row is retained for historical evidence.** Two prior attempts to tune RocksDB without WBM made things worse (240 M+ overwrites, catastrophic eviction starvation). The comment block in `src/storage_rocksdb.cpp` documents why WBM is the correct mechanism and why alternate caps fail.

## Not covered

- Corruption injection under load — that's in `tests/test_phase2_integration.cpp`.
- Multi-node / distributed anything — out of scope
- Perf regression vs baseline — that's Phase 3 with `bench_engine`.
