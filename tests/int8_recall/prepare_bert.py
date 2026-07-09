"""
Generate BERT-style sentence embeddings and ground truth for INT8 validation.

Uses sentence-transformers/all-MiniLM-L6-v2 (384-dim) on the 20 Newsgroups
corpus split into sentence-ish chunks. 50k train + 5k queries + cosine
top-100 ground truth (computed with numpy).

Prereqs (installed into a local venv by the script):
  - Python 3.11+ (Python 3.14 has a huggingface_hub httpx bug)
  - numpy, sentence-transformers, scikit-learn
  - Network access to HuggingFace to fetch the model on first run

Usage:
  python3 prepare_bert.py [OUTDIR]

  OUTDIR defaults to /tmp/chronosv-bert

Output: $OUTDIR/{train.fbin, test.fbin, gt.ibin}

Note: on some corporate networks with MITM SSL inspection the model
download may fail with SSL cert verification errors. The script disables
SSL verification for the model download only — safe for a local one-off
validation, NOT appropriate for production code.
"""
import os
import re
import ssl
import sys
import numpy as np

# ---------------------------------------------------------------------------
# Env setup: bypass SSL cert verification for the HuggingFace model download.
# Necessary on corporate networks with MITM proxies whose CA isn't in
# Python's certifi bundle. Scoped to this one script only.
# ---------------------------------------------------------------------------
import urllib3
urllib3.disable_warnings()
ssl._create_default_https_context = ssl._create_unverified_context
import requests
_orig = requests.Session.request
def _no_verify(self, *a, **k):
    k['verify'] = False
    return _orig(self, *a, **k)
requests.Session.request = _no_verify

from sentence_transformers import SentenceTransformer
from sklearn.datasets import fetch_20newsgroups

OUTDIR = sys.argv[1] if len(sys.argv) > 1 else "/tmp/chronosv-bert"
os.makedirs(OUTDIR, exist_ok=True)
print(f"Output directory: {OUTDIR}\n")

print("Loading text corpus (20 newsgroups, ~18k documents)...")
data = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
print(f"  got {len(data.data)} documents")

print("Splitting into sentence-ish chunks...")
sent_re = re.compile(r"(?<=[.!?])\s+")
sentences = []
for doc in data.data:
    doc = doc.strip()
    if not doc:
        continue
    for s in sent_re.split(doc):
        s = s.strip()
        if 20 <= len(s) <= 300 and len(s.split()) >= 4:
            sentences.append(s)
    if len(sentences) >= 60000:
        break
sentences = sentences[:55000]
print(f"  got {len(sentences)} sentences")

print("\nLoading model (all-MiniLM-L6-v2, 384-dim; ~90 MB download on first run)...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("\nEmbedding sentences...")
emb = model.encode(sentences, batch_size=128,
                   show_progress_bar=True,
                   convert_to_numpy=True,
                   normalize_embeddings=True).astype(np.float32)
print(f"  shape = {emb.shape}, dtype = {emb.dtype}")
print(f"  L2 norms: min={np.linalg.norm(emb, axis=1).min():.6f}, "
      f"max={np.linalg.norm(emb, axis=1).max():.6f}")

# 50k train + 5k queries
n_test = 5000
train = emb[:-n_test]
test  = emb[-n_test:]
print(f"\nSplit: train {train.shape}, test {test.shape}")

print(f"\nComputing cosine top-100 ground truth (exact, via numpy)...")
K = 100
gt = np.zeros((test.shape[0], K), dtype=np.int32)
chunk = 500
for i in range(0, test.shape[0], chunk):
    q = test[i:i+chunk]
    sims = q @ train.T
    idx = np.argpartition(-sims, K, axis=1)[:, :K]
    row = np.arange(idx.shape[0])[:, None]
    gt[i:i+chunk] = idx[row, np.argsort(-sims[row, idx], axis=1)]

def dump_vecs(path, arr):
    with open(path, "wb") as f:
        np.array([arr.shape[0], arr.shape[1]], dtype=np.uint32).tofile(f)
        arr.astype(np.float32).tofile(f)
    print(f"  wrote {path}: {os.path.getsize(path):,} bytes")

def dump_gt(path, arr):
    with open(path, "wb") as f:
        np.array([arr.shape[0], arr.shape[1]], dtype=np.uint32).tofile(f)
        arr.astype(np.int32).tofile(f)
    print(f"  wrote {path}: {os.path.getsize(path):,} bytes")

print("\nDumping...")
dump_vecs(f"{OUTDIR}/train.fbin", train)
dump_vecs(f"{OUTDIR}/test.fbin",  test)
dump_gt(f"{OUTDIR}/gt.ibin",     gt)
print(f"\n=== BERT embeddings ready at {OUTDIR} ===")
print(f"Run: <build>/tests/int8_recall/test_int8_recall {OUTDIR} cosine")
