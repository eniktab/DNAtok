#!/usr/bin/env python3
"""Validate the DNABERT-2 case study pipeline on a small synthetic input.

Tests:
  1. DNABERT-2 tokenizer loads from the HF cache.
  2. DNAtok wraps it and produces bit-identical token IDs to HF on 100
     random 1 kbp DNA sequences (the regime the case study uses for
     chr21 tiling).
  3. DNAtok is meaningfully faster than HF.

Runs in seconds. Does NOT download chr21 / ENCODE; that's the full run.
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

    # 100 synthetic 1 kbp reads — the chr21 case study tiles at 1 kbp.
    rng = random.Random(42)
    seqs = [gen_dna(1000, rng) for _ in range(100)]

    print("Loading DNABERT-2 tokenizer ...")
    hf_tok = AutoTokenizer.from_pretrained(
        "zhihan1996/DNABERT-2-117M", trust_remote_code=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    result = validate_tokenization(
        model_name="DNABERT-2 (chr21 1 kbp tiling)",
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
