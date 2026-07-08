# Contributing to ChronosVector

Thanks for the interest. This guide covers what you need to know before opening a PR.

## Before opening a PR

Please make sure your change:

1. **Builds and passes tests under all four sanitizer configurations** on both platforms:
   - macOS ARM64 (or native x86_64 if that's your dev box)
   - Linux ARM64 or x86_64 (use `./docker/verify_linux.sh` if you're on macOS)

   The four configs are: Debug, Release, Debug+ASan, Debug+TSan. See [`docs/BUILDING.md`](docs/BUILDING.md) §4 for the full recipe.

2. **Doesn't regress the reference benchmark.** After building Release with `-DCHRONOSV_BUILD_BENCH=ON`, run:
   ```sh
   ./build-rel/bench/bench_engine --benchmark_min_time=0.3s
   ```
   The `BM_QueryNearestN/60000/128` result must stay under 1 ms P99 (the reference-workload target).

3. **Includes tests.** Any behavior change needs a test; any bug fix needs a regression test. See `tests/` for the pattern — Catch2 v3, one file per module, tags for filtering.

4. **Doesn't add new dependencies.** RocksDB + Eigen (vendored) are the only runtime deps and won't grow without a strong reason. Catch2 and google/benchmark are test-only via FetchContent.

## How to test locally

Standard workflow — everything mirrors what CI does:

```sh
# From repo root, first time only:
git submodule update --init --recursive

# Build + test all four configs on macOS:
for cfg in \
    "Debug:" \
    "Release:" \
    "Debug:CHRONOSV_ENABLE_ASAN=ON" \
    "Debug:CHRONOSV_ENABLE_TSAN=ON"; do
    type="${cfg%%:*}"
    extra="${cfg##*:}"
    dir="build-$(echo "$cfg" | tr : - | tr A-Z a-z)"
    cmake -S . -B "$dir" -DCMAKE_BUILD_TYPE="$type" ${extra:+-D$extra}
    cmake --build "$dir" -j
    ctest --test-dir "$dir" --output-on-failure
done

# Linux verification (needs Docker Desktop running):
./docker/verify_linux.sh
```

For iterative dev, the fast path is just Debug + your changed area:
```sh
cmake --build build --target test_engine -j && \
  ./build/tests/test_engine "[phase2]"
```

## Coding style

We don't (yet) ship a `.clang-format` — please match the surrounding code:

- 4-space indent, no tabs
- Curly braces on the same line for functions, control flow: `if (...) {`
- `snake_case` for functions and variables; `PascalCase` for types; `SCREAMING_SNAKE` for macros and constants
- Namespace `chronosv::internal` for internal C++; `extern "C"` for the public ABI in `chronos_vector.h`
- Comments on **why**, not what — well-named identifiers already say what

## Commit messages

- Short imperative subject line (< 72 chars)
- Blank line, then body explaining the *why* if non-obvious
- Reference issue numbers when relevant

Example:
```
Fix concurrent flush race on evict_scratch_

chronosv_flush called EvictOnce directly from user threads, but
evict_scratch_ was documented as "single-writer" for the background
eviction jthread. ASan caught it as a heap-buffer-overflow when the
concurrent-flush integration test ran 8 threads simultaneously.

Fix: dedicated eviction_pass_mu_ mutex around EvictOnce. Serializes
background + user-invoked evictions. Contention is bounded (eviction
is cold path). Regression: tests/test_phase2_integration.cpp.
```

## License / DCO

By submitting a pull request, you agree that your contribution is licensed under Apache-2.0 (the project license — see [`LICENSE`](LICENSE)).

We may add a Developer Certificate of Origin (DCO) sign-off requirement in a later release. If we do, `git commit -s` will be enough — no separate CLA.

## Scope reminders (things that get rejected without a strong reason)

- **Distributed clustering, replication, consensus** — use Qdrant or Milvus.
- **SQL parser, transactions, query planner.**
- **General-purpose vector database features** — HNSW/IVF/graph indexes. The differentiator is sliding-window, not "one more vector DB."
- **LLM / agent memory abstractions** — a layer for that would live in a separate repo.
- **Variable-length per-vector payloads** — breaks the bounded-RAM guarantee.
- **Cold-tier reads in `query_range`** — v0.1 is hot-only; a separate `query_cold_range` may come later.
- **Per-query distance metric** — currently pinned at engine construction to keep the hot path branch-free.

If your idea maps to one of these, please open an issue first to discuss — it's usually a "this belongs in a different repo" conversation, and finding out before you write the code saves everyone time.

## Getting help

- **API questions**: read `include/chronosv/chronos_vector.h` header comments (also the API reference).
- **Build issues**: [`docs/BUILDING.md`](docs/BUILDING.md) §8 has the common troubleshooting.
- **Bug reports / feature requests**: GitHub issues.

Welcome aboard.
