#!/bin/bash
# Run the ChronosVector Linux verification via Docker.
#
# Usage:
#   ./docker/verify_linux.sh            # ARM64 native (fast on Apple Silicon)
#   ./docker/verify_linux.sh --x86      # x86_64 via Rosetta (slower, catches SIMD path)
#   ./docker/verify_linux.sh --both     # ARM64 then x86_64
#
# Requires Docker (or a compatible runtime like Colima / OrbStack).
#
# The Eigen submodule MUST be populated on the host before running:
#     git submodule update --init --recursive
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Preflight
if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker not found on PATH." >&2
    echo "Install Docker Desktop, Colima, OrbStack, or Podman first." >&2
    exit 1
fi
if [[ ! -f third_party/eigen/Eigen/Core ]]; then
    echo "ERROR: third_party/eigen not populated." >&2
    echo "Run: git submodule update --init --recursive" >&2
    exit 1
fi

run_platform() {
    local platform="$1"
    local tag="chronosv-linux-verify:${platform##*/}"
    echo ""
    echo "########################################################################"
    echo "# Verifying on $platform"
    echo "########################################################################"
    # Use `docker buildx` for cross-platform builds. It falls back cleanly
    # to `docker build` on single-platform setups.
    docker buildx build \
        --platform "$platform" \
        --load \
        -f docker/Dockerfile.linux \
        -t "$tag" \
        .
    echo ""
    echo "=== $platform: PASS"
}

case "${1:-}" in
    --x86)
        run_platform "linux/amd64"
        ;;
    --both)
        run_platform "linux/arm64"
        run_platform "linux/amd64"
        ;;
    ""|--arm64|--arm)
        run_platform "linux/arm64"
        ;;
    -h|--help)
        sed -n '1,20p' "$0" | sed 's/^# //;s/^#//'
        exit 0
        ;;
    *)
        echo "Unknown arg: $1" >&2
        echo "Try --arm64 (default), --x86, --both, or --help" >&2
        exit 2
        ;;
esac

echo ""
echo "=== ChronosVector Linux verification complete ==="
