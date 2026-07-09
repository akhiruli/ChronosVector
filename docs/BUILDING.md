# ChronosVector — Build, Test, Bench & CLI Guide

Everything you need to compile, test, benchmark, and exercise the library.

---

## 1. Prerequisites

| Tool | Minimum version | macOS install | Linux install |
|---|---|---|---|
| CMake | 3.25 | `brew install cmake` | `apt install cmake` |
| C++ compiler | AppleClang 15+ / Clang 17+ / GCC 13+ | comes with Xcode CLT | `apt install g++-13` or `clang-17` |
| **RocksDB** | 8.x+ | `brew install rocksdb` | `apt install librocksdb-dev` |
| git | any recent | `xcode-select --install` | `apt install git` |

Eigen (vendored via git submodule), Catch2, and google/benchmark are pulled at CMake-configure time via FetchContent — no system installs required for those.

**RocksDB is a runtime + link dependency** since Phase 2. Linked dynamically; the `libchronosv` binary itself stays small, but the caller's process links `librocksdb.dylib` / `librocksdb.so` at runtime. Verified against Homebrew RocksDB 11.1.2 (macOS) and Ubuntu 24.04 librocksdb-dev 8.x (Linux).

### Check your environment
```sh
cmake --version    # need >= 3.25
c++ --version      # need C++23 support
```

---

## 2. First-time setup

```sh
cd ChronosVector

# Initialize the Eigen submodule (needed for kernels).
git submodule update --init --recursive

# Verify Eigen is at v3.4.0 (a stable release, not master HEAD).
cd third_party/eigen
git describe --tags     # expect: 3.4.0
cd ../..
```

If the submodule directory is empty, run `git submodule update --init` again. Nothing else needs installing — the build system fetches Catch2 (tests) and google/benchmark (benches) on first configure.

---

## 3. Build

The library follows the standard CMake out-of-source pattern. Pick a build directory (`build`, `build-rel`, `build-tsan`, etc.) per configuration.

### 3.1 Default: Debug with tests

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
```

Produces `build/libchronosv.a` and the test binaries under `build/tests/`.

### 3.2 Release (recommended for anything perf-sensitive)

```sh
cmake -S . -B build-rel -DCMAKE_BUILD_TYPE=Release
cmake --build build-rel -j
```

Uses `-O3 -DNDEBUG`. Assertions compile away; the SPSC guard becomes a no-op.

### 3.3 Release + benchmarks + CLI

```sh
cmake -S . -B build-rel \
      -DCMAKE_BUILD_TYPE=Release \
      -DCHRONOSV_BUILD_BENCH=ON \
      -DCHRONOSV_BUILD_EXAMPLES=ON
cmake --build build-rel -j
```

Adds `build-rel/bench/{bench_kernels,bench_ring,bench_engine}` and `build-rel/examples/chronosv_cli`.

### 3.4 Sanitizer builds

Sanitizers are mutually exclusive; enable at most one per build tree.

```sh
# AddressSanitizer — catches use-after-free, OOB reads/writes, leaks
cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug -DCHRONOSV_ENABLE_ASAN=ON
cmake --build build-asan -j

# ThreadSanitizer — catches data races
cmake -S . -B build-tsan -DCMAKE_BUILD_TYPE=Debug -DCHRONOSV_ENABLE_TSAN=ON
cmake --build build-tsan -j

# UndefinedBehaviorSanitizer
cmake -S . -B build-ubsan -DCMAKE_BUILD_TYPE=Debug -DCHRONOSV_ENABLE_UBSAN=ON
cmake --build build-ubsan -j
```

### 3.5 All CMake options

| Option | Default | Effect |
|---|---|---|
| `CMAKE_BUILD_TYPE` | `Release` | `Debug`, `Release`, `RelWithDebInfo`, `MinSizeRel` |
| `CHRONOSV_BUILD_TESTS` | `ON` | Build unit + integration tests |
| `CHRONOSV_BUILD_BENCH` | `OFF` | Build benchmarks |
| `CHRONOSV_BUILD_EXAMPLES` | `OFF` | Build the CLI + any example programs |
| `CHRONOSV_BUILD_SOAK` | `OFF` | Build the long-running soak test (see §5.8) |
| `CHRONOSV_ENABLE_INT8` | `ON` | Compile INT8 quantized storage path (users opt in at runtime via `cfg.storage_dtype`). See [`docs/INT8.md`](INT8.md) for measured recall numbers |
| `CHRONOSV_BUILD_INT8_RECALL` | `OFF` | Build INT8 recall validation harness (needs external data — see `tests/int8_recall/README.md`) |
| `CHRONOSV_ENABLE_ASAN` | `OFF` | AddressSanitizer |
| `CHRONOSV_ENABLE_TSAN` | `OFF` | ThreadSanitizer |
| `CHRONOSV_ENABLE_UBSAN` | `OFF` | UndefinedBehaviorSanitizer |
| `EIGEN_INCLUDE_DIR` | `third_party/eigen` | Override to a system Eigen install |

### 3.6 Install (optional)

```sh
cmake --install build-rel --prefix /path/to/install
```

Installs:
```
${prefix}/lib/libchronosv.a
${prefix}/lib/cmake/chronosv/chronosv-config.cmake
${prefix}/include/chronosv/chronos_vector.h
${prefix}/include/chronosv/chronos_vector.hpp
${prefix}/include/chronosv/metrics_sink.h
```

Internal headers (`ring_buffer.h`, `types.h`, `kernels.h`) are deliberately not installed.

---

## 4. Running tests

### 4.1 Full test suite

```sh
ctest --test-dir build --output-on-failure
```

Expected: `100% tests passed, 0 tests failed out of 140` (some death tests only run in Debug).

### 4.2 Under sanitizers

Same command, different build dir:

```sh
ctest --test-dir build-asan --output-on-failure   # ~5 s
ctest --test-dir build-tsan --output-on-failure   # ~25 s
```

### 4.3 Filtering (Catch2 tags)

Test executables accept Catch2's filter syntax:

```sh
# Only ring buffer tests
./build/tests/test_ring_buffer

# Filter by tag
./build/tests/test_ring_buffer "[thread]"
./build/tests/test_kernels    "[cosine]"
./build/tests/test_engine     "[eviction]"

# Filter by name
./build/tests/test_engine "chronosv_query_nearest_n happy path"

# Verbose output on success too
./build/tests/test_ring_buffer -s
```

Available tags include `[ring]`, `[kernel]`, `[engine]`, `[cosine]`, `[euclidean]`, `[thread]`, `[edge]`, `[wrap]`, `[eviction]`, `[cpp_wrapper]`, `[error]`, `[death]`.

### 4.4 Test suites at a glance

| Executable | Focus |
|---|---|
| `test_ring_buffer` | SoA layout, wrap, overwrite counter, SPSC, multi-reader, SPSC death test (Debug only) |
| `test_kernels` | L2 norm, cosine + euclidean (curated, boundary, property, NaN, determinism, ordering), INT8 kernels |
| `test_engine` | All 15 C ABI primitives + C++ wrapper + sink callbacks + eviction (sync + background) + overwrite-survives-drop + Phase 2 persistence (create-with-cold-path → flush → destroy → open recovery) + `recover_hot_window` modes + Create schema-verify vs existing metadata |
| `test_storage_backend` | Abstract interface smoke test + `NullStorageBackend` no-op impl + custom subclass polymorphism + `WriteMetadata`/`ReadMetadata` |
| `test_block_codec` | Byte-level on-disk layout; CRC catches single-bit flips; round-trip float32/int8+payload; rejection of truncated/bad-magic/bad-version/unknown-flag/CRC-mismatch inputs |
| `test_storage_rocksdb` | Open/close/round-trip/DropSensor/persistence/corruption injection/ListSensors/metadata via RocksDB tempdir |
| `test_phase2_integration` | N-producer + eviction under sustained load; concurrent flush is idempotent; chronosv_open survives a corrupt persisted block |

Total: 140 tests across all executables. Individual counts drift as tests are added; use `ctest -N` for the current tally.

---

## 5. Running benchmarks

Benchmarks use Google Benchmark. Only meaningful in Release.

### 5.1 Distance kernels

```sh
./build-rel/bench/bench_kernels --benchmark_min_time=0.5s
```

Sweeps `(count, dim)` for cosine and euclidean. The reference workload is `60000 × 128` — the 10-min × 100 Hz sensor stream that drives the design. Also runs the direct-broadcast euclidean kernel for comparison against the fast identity path.

### 5.2 Ring buffer

```sh
./build-rel/bench/bench_ring --benchmark_min_time=0.5s
```

Measures raw `SensorRing::Append` latency, sustained throughput, and Append-with-concurrent-Reader — the last one exercises the SPSC ordering path.

### 5.3 Engine end-to-end

```sh
./build-rel/bench/bench_engine --benchmark_min_time=0.5s
```

Times the full extern "C" wall — sensor-map lookup, kernel call, top-K heap, timestamp write-back, metrics callbacks. This is the number that gates the sub-millisecond query latency claim.

### 5.4 Competitor comparison (Phase 3)

```sh
./build-rel/bench/bench_compare --benchmark_min_time=0.3s
```

ChronosVector's exact brute-force vs hnswlib's approximate HNSW at 10k / 60k / 100k × dim=128. Not a "we're faster" story — different tradeoffs (see the source file header). Documents the crossover at which HNSW's log-N query beats our O(N) scan.

hnswlib is fetched at CMake-configure time; nothing about the shipping library depends on it.

### 5.5 Baselines and CI perf visibility

`bench/baselines/macos-arm64.json` is a checked-in reference `bench_engine` run. The `perf` GitHub Actions workflow (`.github/workflows/perf.yml`) re-runs `bench_engine` on every push/PR and uploads the JSON as an artifact for manual comparison — no automatic gating today (shared runner variance is too high, see `bench/baselines/README.md`).

### 5.6 Reference numbers (Apple Silicon M-series, Release)

| Bench | Target | Measured |
|---|---|---|
| `BM_AppendEngine/128` | any | ~46 ns |
| `BM_AppendLatencyDist/128` P99 | < 1 ms | 84 ns (P99/P50 = 2×) |
| `BM_AppendLatencyDist/512` P99 (post-prefetch) | < 1 ms | 250 ns |
| `BM_AppendLatencyDist/1024` P99 | < 1 ms | 3.5 µs |
| `BM_QueryNearestN/60000/128` | < 1 ms | **~720 µs ✓** |
| `BM_QueryNearestN/100000/128` | (memory-bandwidth-bound at float32) | ~1.3 ms |
| `BM_CosineF32Chunk/60000/128` | – | ~600 µs |
| `BM_EuclideanSqF32Chunk/60000/128` | – | ~600 µs |
| `BM_Snapshot` | – | 1.5 ns |
| `BM_ChronosVector_Query/10000/128` vs `BM_Hnswlib_Query/10000/128` | – | 108 µs vs 92 µs (see README head-to-head) |

Your numbers will vary with CPU, load, and thermal state.

### 5.7 Useful benchmark flags

```sh
--benchmark_filter=REGEX        # only run matching benchmarks
--benchmark_repetitions=5       # median-of-N for less noise
--benchmark_report_aggregates_only=true
--benchmark_out=results.json --benchmark_out_format=json
--benchmark_time_unit=us
```

---

## 6. The CLI (`chronosv_cli`)

A small line-based REPL for exercising the C++ API without writing a test harness. **Not** a production tool — for debug, demo, and manual verification only.

### 6.1 Build

```sh
cmake -S . -B build-rel -DCMAKE_BUILD_TYPE=Release -DCHRONOSV_BUILD_EXAMPLES=ON
cmake --build build-rel --target chronosv_cli -j
```

### 6.2 Run

```sh
./build-rel/examples/chronosv_cli
```

You'll get a `> ` prompt. Type `help` to see all commands. Input is one command per line. Blank lines and lines starting with `#` are ignored (so you can pipe scripts).

### 6.3 Command reference

| Command | Purpose |
|---|---|
| `create dim=D [cap=C] [metric=cosine\|euclidean] [window_ms=N] [interval_ms=N]` | Instantiate a new engine. `dim` is required; other fields are default. |
| `preallocate <sensor>` | Force-create a sensor's ring buffer now (avoids first-Append allocation later). |
| `append <sensor> <ts_ms> <v1,v2,...>` | Ingest one vector. Comma-separated, no spaces. |
| `query <sensor> <v1,v2,...> [n=N]` | k-NN over the current hot window. Default `n=5`. Prints ranked `ts=…  score=…`. |
| `range <sensor> <t_start_ms> <t_end_ms> [max=N]` | All vectors with `ts` in `[start, end]`. Default `max=10`. Prints full vectors. |
| `anomaly <sensor> <v1,v2,...> [t=THRESHOLD]` | Check whether `v` is anomalous vs. the window mean. Default `t=0.5`. |
| `drop <sensor>` | Remove a sensor's data. |
| `list` | Enumerate registered sensor IDs. |
| `stats` | Print `chronosv_stats_t` snapshot (UUID, sensor count, append/query/eviction counters, overwrite events, memory). |
| `maintain <window_ms>` | Set window and trigger an immediate eviction pass. |
| `close` | Close the engine. Further commands (except `quit`) will error with `CLOSED`. |
| `help` | Print command list. |
| `quit` / `exit` | Leave the REPL. |

### 6.4 Example session

```sh
$ ./build-rel/examples/chronosv_cli
chronosv_cli 0.1.0 — type `help` for commands
> create dim=4
engine created (dim=4)
> append s1 100 1,0,0,0
ok
> append s1 200 0.9,0.1,0,0
ok
> append s1 300 0,1,0,0
ok
> query s1 1,0,0,0 n=3
  ts=100  score=1.000000
  ts=200  score=0.993884
  ts=300  score=0.000000
> stats
uuid:                       c709013f-f6f3-4036-8018-68821440ed39
abi_version:                1
sensor_count:               1
sensor_cap:                 0
total_appends:              3
total_queries:              1
total_anomaly_checks:       0
total_evictions:            0
total_dropped_sensors:      0
total_overwrite_events:     0
total_overwritten_entries:  0
hot_bytes:                  1835008
> range s1 100 200
  ts=100  vec=[1.0000,0.0000,0.0000,0.0000]
  ts=200  vec=[0.9000,0.1000,0.0000,0.0000]
> anomaly s1 0,1,0,0 t=0.5
anomaly
> anomaly s1 0.95,0.05,0,0 t=0.5
normal
> maintain 100
ok
> list
  s1
1 sensor(s)
> drop s1
ok
> list
0 sensor(s)
> quit
```

### 6.5 Scripted use

The CLI reads stdin, so you can pipe:

```sh
cat <<'EOF' | ./build-rel/examples/chronosv_cli
create dim=4
append s1 100 1,0,0,0
append s1 200 0,1,0,0
query s1 1,0,0,0 n=2
stats
quit
EOF
```

### 6.6 Limitations

- No readline / no history. Bare `getline`. Use `rlwrap ./chronosv_cli` if you want basic line editing.
- No batch append syntax. One `append` per line.
- No `open` command in the REPL yet (API is live in Phase 2 — the CLI just hasn't been extended).
- Vector components are separated by commas with no spaces (spaces split tokens).

---

### 5.8 Soak test

Long-running load with flat-RSS check. Validated at 10 kHz × 128-dim for 6 hours (see `soak/README.md` for the measured PASS result). Full details in `soak/README.md`.

### Build

```sh
cmake -S . -B build-rel \
      -DCMAKE_BUILD_TYPE=Release \
      -DCHRONOSV_BUILD_SOAK=ON
cmake --build build-rel --target chronosv_soak -j
```

### Run

**10-minute smoke** (infrastructure check):
```sh
./build-rel/soak/chronosv_soak --duration 600 --cold /tmp/chronosv-soak
```

**Full 24 hours** (signoff run):
```sh
./build-rel/soak/chronosv_soak --duration 86400 --cold /tmp/chronosv-soak-24h 2>&1 | tee soak-24h.log
```

### Pass criteria (auto-enforced)

- `flush_errors_total == 0`
- Process-private memory grew ≤ 5% from steady baseline (for runs ≥ 1 h). Shorter smoke runs use a relaxed 100% threshold since RocksDB oscillates during warmup.
- `cold_bytes_estimate` grew above zero
- `total_evictions > 0`

Exits 0 on PASS, non-zero on FAIL. See `soak/README.md` for CSV interpretation and troubleshooting.

---

## 7. Repo layout

```
ChronosVector/
├── CMakeLists.txt              # top-level build config
├── README.md
├── LICENSE                     # Apache-2.0
├── CONTRIBUTING.md
├── .gitignore
├── .gitmodules                 # eigen submodule
├── docs/
│   ├── BUILDING.md             # this file
│   └── INT8.md                 # INT8 storage: when to use + measured recall/perf
├── include/chronosv/
│   ├── chronos_vector.h        # PUBLIC — the C ABI contract
│   ├── chronos_vector.hpp      # PUBLIC — header-only C++ wrapper
│   ├── metrics_sink.h          # PUBLIC — C fn-ptr sink
│   ├── ring_buffer.h           # internal — lock-free SPSC ring
│   ├── storage_backend.h       # internal — cold-tier abstract iface
│   └── types.h                 # internal — shared PODs
├── src/
│   ├── engine.cpp              # engine core + extern "C" wall + eviction thread
│   ├── storage_rocksdb.{cpp,h} # default StorageBackend impl
│   ├── block_codec.{cpp,h}     # binary block format
│   └── kernels.{cpp,h}         # Eigen SIMD distance kernels (f32 + i8)
├── tests/                      # 140 tests, Catch2
│   ├── test_ring_buffer.cpp
│   ├── test_kernels.cpp
│   ├── test_kernels_int8.cpp
│   ├── test_engine.cpp
│   ├── test_storage_backend.cpp
│   ├── test_storage_rocksdb.cpp
│   ├── test_block_codec.cpp
│   ├── test_phase2_integration.cpp
│   └── int8_recall/            # opt-in INT8 recall harness (needs external data)
│       ├── test_int8_recall.cpp
│       ├── prepare_sift1m.sh
│       ├── prepare_bert.py
│       ├── CMakeLists.txt
│       └── README.md
├── bench/
│   ├── bench_ring.cpp
│   ├── bench_kernels.cpp
│   ├── bench_engine.cpp        # includes BM_AppendLatencyDist for P99 diagnostics
│   ├── bench_compare.cpp       # ChronosVector vs hnswlib
│   └── baselines/              # checked-in macOS + Linux JSON references
├── examples/
│   ├── chronosv_cli.cpp        # debug REPL
│   └── anomaly_stream.cpp      # walkthrough: metrics sink + anomaly detection
├── soak/
│   ├── soak_test.cpp           # long-running load + flat-RSS check
│   └── README.md               # workload table, measured results
├── docker/
│   ├── Dockerfile.linux        # Ubuntu 24.04 build recipe
│   ├── build_and_test.sh       # container-side helper: build all configs + ctest
│   ├── verify_linux.sh         # host-side wrapper: cross-verify on Ubuntu 24.04
│   ├── capture_baseline.sh     # (re)generate Linux bench baseline
│   └── README.md               # docker workflow notes
├── .github/workflows/
│   ├── ci.yml                  # matrix build + tests
│   └── perf.yml                # bench_engine visibility (not gated)
└── third_party/
    └── eigen/                  # submodule, pinned to 3.4.0
```

---

## 8. Troubleshooting

### "cmake: command not found"
Install CMake ≥ 3.25 (`brew install cmake` on macOS, distro package manager on Linux).

### "Eigen not found at .../third_party/eigen"
The submodule isn't initialized:
```sh
git submodule update --init --recursive
```

### C++23 deprecation errors from Eigen
Fixed — the build system marks Eigen as a `SYSTEM` include so its `std::float_denorm_style` deprecation warnings don't trip `-Werror`. If you're seeing them, you're on an older `CMakeLists.txt`; re-pull.

### "SPSC contract violated: multiple producer threads on one sensor"
You called `chronosv_append` from two different threads for the same `sensor_id`. The API contract is SPSC per sensor — serialize your producers externally, or use one sensor per producer thread. Only fires in Debug builds; in Release it silently corrupts the ring.

### Tests fail under TSan intermittently
The `SensorRing SPSC with wrap` test is a survival-only test (correctness of overwrite counters, not consumer observation ordering). If TSan flags a race during producer-laps-consumer, rerun once — TSan detection there is nondeterministic on macOS. Persistent failures indicate a real bug.

### Benchmark shows "clock rate from sysctl: hw.cpufrequency: No such file"
Harmless. macOS ARM64 doesn't expose that MSR; timings are still accurate, only the metadata printout is off.

### Cross-compiler mismatch on Linux
CI matrix should test both GCC 13 and Clang 17. Locally you can pick with:
```sh
CXX=g++-13   cmake -S . -B build-gcc  ...
CXX=clang++  cmake -S . -B build-clang ...
```

## 9. Getting help

- **API questions**: read `include/chronosv/chronos_vector.h` — the header comments are the API reference, with full Doxygen tags for LSP hover.
- **Bug reports**: file an issue on the project's GitHub repository. See `CONTRIBUTING.md` for PR guidelines.
