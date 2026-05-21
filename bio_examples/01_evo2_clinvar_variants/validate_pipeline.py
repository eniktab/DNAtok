#!/usr/bin/env python3
"""Validate the Evo2 ClinVar case study pipeline on small synthetic input.

Tests:
  1. Evo2-1b tokenizer loads from the HF cache.
  2. DNAtok wraps it and produces bit-identical token IDs to HF on
     100 random 4 kbp DNA sequences (4 kbp = the variant-window size
     used by the full ClinVar case study).
  3. DNAtok is meaningfully faster than HF.

Runs in seconds. Does NOT download ClinVar / hg38; that's the full run.
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
    from benchmarks.model_registry import MODEL_SPECS, resolve_model_path
    from benchmarks.tokenizer_adapters import load_hf_tokenizer

    # 100 synthetic 4 kbp reads — the ClinVar case study uses 4 kbp
    # ref + alt windows centred on each SNV.
    rng = random.Random(42)
    seqs = [gen_dna(4096, rng) for _ in range(100)]

    print("Loading Evo2-1b tokenizer ...")
    spec = next(s for s in MODEL_SPECS if s.name == "Evo2_1b_base")
    path = resolve_model_path(spec)
    if path is None:
        print("Evo2-1b weights not in HF cache; download to validate.")
        return 2
    hf_tok = load_hf_tokenizer(str(path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    result = validate_tokenization(
        model_name="Evo2-1b (ClinVar 4 kbp ref/alt windows)",
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
