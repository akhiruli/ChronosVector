#!/bin/bash
# Run bench_engine inside an Ubuntu container matching Dockerfile.linux and
# write the JSON output to bench/baselines/linux-arm64.json (or linux-amd64
# if --x86 is passed) on the host.
#
# This is a maintainer tool for refreshing the committed baselines under
# bench/baselines/. Not run automatically — the point of a baseline is that
# it's stable, not that it moves on every commit.
#
# Usage:
#   ./docker/capture_baseline.sh              # ARM64 native on Apple Silicon
#   ./docker/capture_baseline.sh --x86        # x86_64 via Rosetta
#
# Prerequisites match verify_linux.sh: Docker running, Eigen submodule
# populated on the host.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PLATFORM="linux/arm64"
ARCH_TAG="arm64"
if [[ "${1:-}" == "--x86" ]]; then
    PLATFORM="linux/amd64"
    ARCH_TAG="amd64"
fi

if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker not found on PATH." >&2
    exit 1
fi
if [[ ! -f third_party/eigen/Eigen/Core ]]; then
    echo "ERROR: third_party/eigen not populated. Run 'git submodule update --init --recursive'." >&2
    exit 1
fi

OUT="bench/baselines/linux-${ARCH_TAG}.json"
echo "Capturing $PLATFORM bench_engine baseline → $OUT"

# Build + run inside a fresh container. Ubuntu 24.04 matches verify_linux.sh
# for consistency; librocksdb-dev version is whatever the distro ships.
#
# --cpus=4 --memory=8g reserves a CONSISTENT resource envelope so re-runs
# don't drift with host CPU pressure. Requires Docker Desktop to have
# enough CPUs/RAM allocated to the VM (check Preferences → Resources —
# defaults to 4 CPU / 8 GB on Apple Silicon so we're in the safe range).
docker run --rm \
    --platform "$PLATFORM" \
    --cpus=4 \
    --memory=8g \
    -v "$REPO_ROOT:/src" \
    -w /src \
    ubuntu:24.04 \
    bash -eo pipefail -c '
        export DEBIAN_FRONTEND=noninteractive
        apt-get update >/dev/null
        apt-get install -y --no-install-recommends \
            build-essential cmake git ca-certificates librocksdb-dev >/dev/null
        cmake --version
        g++ --version | head -1
        cmake -S . -B build-baseline \
            -DCMAKE_BUILD_TYPE=Release \
            -DCHRONOSV_BUILD_BENCH=ON \
            -DCHRONOSV_BUILD_TESTS=OFF >/dev/null
        cmake --build build-baseline --target bench_engine -j 2
        # Stabilization flags — see bench/baselines/README.md for rationale.
        #   min_time=1.0s          — longer per-repetition avg smooths sub-µs noise
        #   min_warmup_time=0.5s   — caches primed before measuring
        #   repetitions=10         — enough samples for stddev to matter
        #   random_interleaving    — spreads systematic effects across measurements
        ./build-baseline/bench/bench_engine \
            --benchmark_min_time=1.0s \
            --benchmark_min_warmup_time=0.5 \
            --benchmark_repetitions=10 \
            --benchmark_enable_random_interleaving=true \
            --benchmark_report_aggregates_only=true \
            --benchmark_out='"$OUT"' \
            --benchmark_out_format=json \
            --benchmark_time_unit=us
    '

echo ""
echo "=== Baseline captured: $OUT"
ls -lh "$OUT"
