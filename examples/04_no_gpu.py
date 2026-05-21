#!/usr/bin/env python3
"""DNAtok on a CPU-only laptop — no GPU needed.

DNAtok auto-detects the device. If CUDA is unavailable the wrapper
runs on CPU and still produces bit-identical output to the upstream
Hugging Face tokenizer; you don't get the GPU speedup, but the API,
bit-identity gate, and example pipelines all work.

Useful for:
  - Local development / debugging without a GPU
  - CI testing pipelines on GitHub Actions runners
  - Notebook prototyping before deploying to an H100/H200 batch

Run:
    python3 examples/04_no_gpu.py
"""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Force CPU explicitly — DNAtok would auto-detect anyway, but this
# makes the no-GPU behaviour reproducible.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import dnatok_compat  # noqa: F401
import torch
from transformers import AutoTokenizer
from dna_tokenizer import DNATok

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device:         cpu (forced)")
print()

# Load any of the 7 supported families — DNABERT-2 is small and quick.
hf = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)

class _Emb:
    def __init__(self, t):
        self.tokenizer = t
        v = int(getattr(t, "vocab_size", 0)) or len(t.get_vocab())
        self.embed_table = torch.nn.Embedding(v + 4, 16)        # CPU tensor
    def embed_tokens(self, ids): return self.embed_table(ids)

dn = DNATok(_Emb(hf), normalize_case=False, handle_invalid_chars=False)
dn.discover()

print(f"Selected path:  ", end="")
if dn.bpe_backend is not None: print("gpu-bpe-kernel (not usable on CPU; falls through)")
elif dn.lmm_bpe is not None:   print("cached-lmm (CPU)")
elif dn.kmer_k is not None:    print(f"kmer-{dn.kmer_k} (CPU)")
elif dn.ascii_lut is not None: print("ascii-lut (CPU)")
else:                          print("hf-fallback (CPU)")

# A handful of real chr21-style sequences.
seqs = [
    "ACGTACGTACGTACGTACGTACGTACGTACGTACGT",
    "AAAATTTTGGGGCCCCAAAATTTTGGGGCCCC",
    "AGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCT",
]

# DNAtok output
dn_ids = dn.encode_batch_to_ids(seqs)        # CPU pinned int64
print(f"DNAtok output:  {tuple(dn_ids.shape)} dtype={dn_ids.dtype}")

# HF reference
hf_ids = hf(seqs, add_special_tokens=False, padding=False, truncation=False,
            return_tensors=None)["input_ids"]

# Bit-identity check (strip pad first)
pad = int(dn.id_pad); side = dn.padding_side
n_match = 0
for i, row in enumerate(dn_ids.tolist()):
    if side == "left":
        j = 0
        while j < len(row) and row[j] == pad: j += 1
        v = row[j:]
    else:
        j = len(row)
        while j > 0 and row[j - 1] == pad: j -= 1
        v = row[:j]
    if v == list(hf_ids[i]): n_match += 1

print(f"Bit-identity:   {n_match} / {len(seqs)}  ", "✓" if n_match == len(seqs) else "✗")
