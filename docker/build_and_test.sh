#!/bin/bash
# Build + test one configuration. Called from Dockerfile.linux and safe to
# run standalone if you're inside a Linux environment.
#
# Usage:
#   ./docker/build_and_test.sh <name> "<extra cmake args>"
#
# Example:
#   ./docker/build_and_test.sh asan "-DCMAKE_BUILD_TYPE=Debug -DCHRONOSV_ENABLE_ASAN=ON"
#
# Exits non-zero on ANY failure (configure, build, or test). This is what
# makes `docker build` fail loudly instead of silently producing a bad image.
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 <config-name> <cmake-args>" >&2
    exit 2
fi

NAME="$1"
CMAKE_ARGS="$2"
BUILD_DIR="build-linux-${NAME}"

echo ""
echo "======================================================================"
echo "=== Configuring $BUILD_DIR"
echo "===   $CMAKE_ARGS"
echo "======================================================================"

# shellcheck disable=SC2086
cmake -S . -B "$BUILD_DIR" $CMAKE_ARGS

echo ""
echo "=== Building $BUILD_DIR"
# Cap parallelism to avoid OOM inside Docker containers with default memory
# limits (~4-8 GB). Debug builds of Catch2 spike to ~500 MB per cc1plus,
# and unbounded -j on a machine with 8+ cores can blow past the container
# memory limit. Override via CHRONOSV_BUILD_JOBS env if you know your host
# has plenty of RAM allocated to Docker.
JOBS="${CHRONOSV_BUILD_JOBS:-2}"
cmake --build "$BUILD_DIR" -j "$JOBS"

echo ""
echo "=== Running tests in $BUILD_DIR"
ctest --test-dir "$BUILD_DIR" --output-on-failure

echo ""
echo "=== $BUILD_DIR: PASS"
