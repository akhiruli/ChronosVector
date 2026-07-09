#!/bin/bash
# Download SIFT-1M and convert to the .fbin / .ibin format the harness reads.
#
# SIFT-1M: 1,000,000 base vectors + 10,000 test queries, 128-dim FP32 SIFT
# descriptors, Euclidean distance ground truth. Standard ANN benchmark
# dataset from ann-benchmarks.com.
#
# Output: $OUTDIR/{train.fbin, test.fbin, gt.ibin}
#
# Prereqs:
#   - curl or wget
#   - Python 3 with numpy and h5py (installed into a local venv)

set -euo pipefail

OUTDIR="${1:-/tmp/chronosv-sift1m}"
mkdir -p "$OUTDIR"
cd "$OUTDIR"

if [[ ! -f sift-128-euclidean.hdf5 ]]; then
    echo "Downloading SIFT-1M HDF5 (~500 MB) from ann-benchmarks.com..."
    curl -# -o sift-128-euclidean.hdf5 \
         http://ann-benchmarks.com/sift-128-euclidean.hdf5
else
    echo "Already downloaded: $OUTDIR/sift-128-euclidean.hdf5"
fi

# Create a scoped venv so we don't pollute the user's environment
VENV="$OUTDIR/.venv"
if [[ ! -d "$VENV" ]]; then
    echo "Creating Python venv at $VENV..."
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install --quiet numpy h5py
fi

echo "Converting HDF5 → .fbin / .ibin..."
"$VENV/bin/python" - <<PY
import h5py, numpy as np, os
src = "$OUTDIR/sift-128-euclidean.hdf5"
with h5py.File(src, "r") as f:
    train = np.array(f["train"], dtype=np.float32)
    test  = np.array(f["test"],  dtype=np.float32)
    gt    = np.array(f["neighbors"], dtype=np.int32)
def dump_vecs(path, arr):
    with open(path, "wb") as f:
        np.array([arr.shape[0], arr.shape[1]], dtype=np.uint32).tofile(f)
        arr.astype(np.float32).tofile(f)
    print(f"  {path}: {os.path.getsize(path):,} bytes")
def dump_gt(path, arr):
    with open(path, "wb") as f:
        np.array([arr.shape[0], arr.shape[1]], dtype=np.uint32).tofile(f)
        arr.astype(np.int32).tofile(f)
    print(f"  {path}: {os.path.getsize(path):,} bytes")
dump_vecs("$OUTDIR/train.fbin", train)
dump_vecs("$OUTDIR/test.fbin",  test)
dump_gt("$OUTDIR/gt.ibin",     gt)
PY

echo ""
echo "=== SIFT-1M ready at $OUTDIR ==="
echo "Run: <build>/tests/int8_recall/test_int8_recall $OUTDIR euclidean"
