#!/usr/bin/env python3
"""DNAtok hello-world — the minimal 12-line example.

This is the shortest path from `pip install dnatokenizer` (or
`docker run dnatok:latest`) to working bit-identical GPU tokenization.

Run:
    python3 examples/01_hello_world.py
"""
import torch
from transformers import AutoTokenizer
from dna_tokenizer import DNATok

hf = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)

class _Emb:
    def __init__(self, t):
        self.tokenizer = t
        self.embed_table = torch.nn.Embedding(len(t.get_vocab()) + 4, 16).to(
            "cuda" if torch.cuda.is_available() else "cpu")
    def embed_tokens(self, ids): return self.embed_table(ids)

dn = DNATok(_Emb(hf), normalize_case=False, handle_invalid_chars=False)
dn.discover()                                # auto-select the fastest correct path

ids = dn.encode_batch_to_ids(["ACGTACGTACGTACGT", "GCATGCATGCATGCAT"])
print(ids)                                   # [B, T] CPU pinned int64, bit-identical to HF
