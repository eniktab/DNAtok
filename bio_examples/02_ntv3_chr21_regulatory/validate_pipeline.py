#!/usr/bin/env python3
"""Validate the NTv3 chr21 case study pipeline on small synthetic input.

Tests:
  1. NTv3-650M-131kb tokenizer loads from the HF cache.
  2. DNAtok wraps it and produces bit-identical token IDs to HF on
     a small number of long DNA sequences.
  3. DNAtok is meaningfully faster than HF.

We use 16 kbp test sequences (instead of the case study's 131 kbp) so
this can run quickly on a single GPU. The relative speedup pattern is
the same — single-nucleotide tokenisation is throughput-linear in
sequence length.

Runs in tens of seconds. Does NOT download chr21; that's the full run.
"""
from __future__ import annotations
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "bio_examples"))

from _common.validate import validate_tokenization, print_result


def gen_dna(L: int, rng: random.Random) -> str:
    bases = "ACGT"
    return "".join(rng.choice(bases) for _ in range(L))


def main() -> int:
    import torch
    from transformers import AutoTokenizer

    # 20 sequences × 16 kbp. The full case study uses 131 kbp; we use
    # 16 kbp here to keep the validation under a minute.
    rng = random.Random(42)
    seqs = [gen_dna(16_384, rng) for _ in range(20)]

    print("Loading NTv3-650M-131kb tokenizer ...")
    hf_tok = AutoTokenizer.from_pretrained(
        "InstaDeepAI/NTv3_650M_post_131kb", trust_remote_code=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    result = validate_tokenization(
        model_name="NTv3-650M-131kb (chr21 long-context tiling)",
        hf_tok=hf_tok,
        seqs=seqs,
        device=device,
    )
    print_result(result)

    out_path = Path(__file__).parent / "validation_result.json"
    with open(out_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    print(f"\nWrote {out_path}")
    return 0 if result.ids_bit_identical else 1


if __name__ == "__main__":
    sys.exit(main())
