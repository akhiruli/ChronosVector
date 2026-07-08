# Linux verification via Docker

Purpose: verify the library compiles and tests pass on Linux from a macOS
dev box, matching what CI runs on GitHub Actions. This is a maintainer /
contributor tool for pre-push validation, not part of the shipped library.

## Prerequisites

1. **Docker Desktop** (or Colima / OrbStack / Podman) installed and running.
   ```sh
   # Docker Desktop
   brew install --cask docker && open -a Docker
   # OR Colima (lighter)
   brew install colima docker && colima start
   ```
2. **Eigen submodule populated on the host** (the container can't reach the
   git remote by design — we exclude `.git` from the build context):
   ```sh
   git submodule update --init --recursive
   ```

## Usage

From repo root:

```sh
# ARM64 Linux, native on Apple Silicon (~5 min first build, ~30 s on rebuild)
./docker/verify_linux.sh

# x86_64 Linux via Rosetta 2 emulation — slower (~15 min) but exercises the
# AVX2/AVX-512 Eigen SIMD path that NEON on Apple Silicon does not.
./docker/verify_linux.sh --x86

# Both
./docker/verify_linux.sh --both
```

## What it does

For each requested platform:

1. Builds an Ubuntu 24.04 image with GCC 13, CMake 3.28, and the repo copied in.
2. Inside the container, builds the library and tests under FOUR configurations:
   - Release
   - Debug
   - Debug + AddressSanitizer
   - Debug + ThreadSanitizer
3. Runs the full `ctest` suite under each configuration.
4. Fails the `docker build` (loud, red output) if any config fails to compile
   or if any test fails.

## Expected result

If everything works, the last line is:

```
=== linux/arm64: PASS
=== ChronosVector Linux verification complete ===
```

## Files

- `Dockerfile.linux` — the build recipe.
- `build_and_test.sh` — one-config helper called by the Dockerfile.
- `verify_linux.sh` — host-side wrapper that runs `docker buildx build`
  for one or both platforms.

## When to run

- After any nontrivial code change to library internals.
- Before opening a PR (mirrors what CI will do — catches issues early).
- Before publishing to a public repo (catch any macOS-specific assumptions).

## Not covered

- **`bench_engine` performance regression check** — that runs in the `perf`
  GitHub Actions workflow (`.github/workflows/perf.yml`); see
  `bench/baselines/README.md`.
- **6-hour / 24-hour soak** — needs a longer-running Linux host than a
  Docker session on a laptop. See `soak/README.md` for the workflow.
- **Benchmark comparison vs USearch / hnswlib** — `bench_compare` needs
  hnswlib fetched via CMake; runs manually with `bench_compare` in
  Release mode.
