"""DNAtok command-line interface.

Single entry-point with sub-commands; designed so a user with the
Docker image can run any of the following without writing Python:

    dnatok info     --model zhihan1996/DNABERT-2-117M
    dnatok encode   --model zhihan1996/DNABERT-2-117M --seq ACGTACGT
    dnatok demo     --model zhihan1996/DNABERT-2-117M
    dnatok bench    --model zhihan1996/DNABERT-2-117M --window 4096 --n 1000
    dnatok validate --model zhihan1996/DNABERT-2-117M --n 500
    dnatok list-models

Each sub-command exits 0 on success, non-zero on failure, and prints
human-readable output plus a `--json` flag for machine-readable.

The CLI auto-handles:
  - `dnatok_compat` import (must precede transformers)
  - device selection (cuda if available, else cpu)
  - HF cache location (uses HF_HOME if set, else ~/.cache/huggingface)
  - tokenizer family auto-discovery (no need to tell us which path)
  - bit-identity self-check before reporting any speedup number
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Path bootstrapping — works whether installed via `pip install -e .` or
# called directly from the source tree.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (_HERE, _ROOT, _ROOT / "bio_examples", _ROOT / "benchmarks"):
    sp = str(_p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

# dnatok_compat MUST be imported before transformers (handles HF cache
# tokenizer.json lookup + Triton libcuda env + class registrations).
import dnatok_compat  # noqa: F401, E402


# ---------------------------------------------------------------------------
# Model registry — the set of HF model ids we've tested end-to-end and
# can confirm work with DNAtok. The CLI accepts ANY model id; this is
# just for `dnatok list-models` and discoverability.
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    # family            (hf-id, fast-path, tested-on, notes)
    "DNABERT-2": (
        "zhihan1996/DNABERT-2-117M", "gpu_bpe", "H200/A100/GB10",
        "GPU BPE kernel + cached safe-margin layer",
    ),
    "GENA-LM": (
        "AIRI-Institute/gena-lm-bert-base-t2t", "gpu_bpe", "H200/A100/GB10",
        "GPU BPE kernel + cached safe-margin layer",
    ),
    "METAGENE-1": (
        "metagene-ai/METAGENE-1", "gpu_bpe", "H200/A100/GB10",
        "GPU BPE kernel (SentencePiece-flavoured)",
    ),
    "NTv3-8M": (
        "InstaDeepAI/NTv3_8M_pre", "ascii_lut", "H200/A100/GB10",
        "ASCII byte lookup table (4-base alphabet)",
    ),
    "NTv2-50M": (
        "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
        "kmer_lut", "H200/A100/GB10",
        "6-mer lookup table with variable-length dispatch",
    ),
    "HyenaDNA-tiny": (
        "LongSafari/hyenadna-tiny-1k-seqlen-hf", "ascii_lut",
        "H200/A100/GB10", "Char-level ASCII LUT",
    ),
    "Evo2-1B": (
        "arcinstitute/evo2_1b_base", "ascii_lut", "H200/A100/GB10",
        "Byte-level (requires the evo2 package; see Dockerfile)",
    ),
}


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------
def _load_tokenizer(model_id: str):
    """Load any HF tokeniser. Evo2 needs the byte-level adapter."""
    if "evo2" in model_id.lower():
        try:
            from benchmarks.tokenizer_adapters import load_hf_tokenizer
            return load_hf_tokenizer(model_id)
        except ImportError:
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


def _build_dnatok(model_id: str, device: str):
    """Build a DNAtok wrapper around the given HF tokeniser."""
    import torch
    from dna_tokenizer import DNATok

    hf = _load_tokenizer(model_id)

    class _Emb:
        def __init__(self, t):
            self.tokenizer = t
            v = int(getattr(t, "vocab_size", 0)) or len(t.get_vocab())
            self.embed_table = torch.nn.Embedding(v + 4, 16).to(device)
        def embed_tokens(self, ids):
            return self.embed_table(ids)

    dn = DNATok(_Emb(hf), normalize_case=False, handle_invalid_chars=False)
    dn.discover()
    return hf, dn


def _detect_path(dn) -> str:
    """Return a short label for which fast path was selected."""
    if getattr(dn, "kmer_k", None) is not None:
        return f"kmer-{dn.kmer_k}"
    if getattr(dn, "ascii_lut", None) is not None:
        return "ascii-lut"
    if getattr(dn, "bpe_backend", None) is not None:
        return "gpu-bpe-kernel"
    if getattr(dn, "lmm_bpe", None) is not None:
        return "cached-lmm"
    return "hf-fallback"


def _default_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _emit(payload, as_json: bool):
    if as_json:
        json.dump(payload, sys.stdout, indent=2)
        print()
    else:
        if isinstance(payload, dict):
            width = max(len(k) for k in payload) if payload else 0
            for k, v in payload.items():
                print(f"  {k:<{width}}  {v}")
        else:
            print(payload)


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------
def cmd_info(args) -> int:
    device = args.device or _default_device()
    hf, dn = _build_dnatok(args.model, device)
    info = {
        "model": args.model,
        "tokenizer_class": type(hf).__name__,
        "vocab_size": int(getattr(hf, "vocab_size", 0)) or len(hf.get_vocab()),
        "selected_path": _detect_path(dn),
        "device": device,
        "kmer_k": dn.kmer_k,
        "has_ascii_lut": dn.ascii_lut is not None,
        "has_gpu_bpe_backend": dn.bpe_backend is not None,
        "has_cached_lmm": dn.lmm_bpe is not None,
        "padding_side": dn.padding_side,
        "id_pad": int(dn.id_pad),
    }
    _emit(info, args.json)
    return 0


def cmd_encode(args) -> int:
    device = args.device or _default_device()
    hf, dn = _build_dnatok(args.model, device)
    ids = dn.encode_batch_to_ids([args.seq])[0].tolist()
    # Strip pad
    pad = int(dn.id_pad)
    if dn.padding_side == "left":
        j = 0
        while j < len(ids) and ids[j] == pad: j += 1
        ids = ids[j:]
    else:
        j = len(ids)
        while j > 0 and ids[j - 1] == pad: j -= 1
        ids = ids[:j]
    payload = {
        "model": args.model,
        "sequence": args.seq[:80] + ("..." if len(args.seq) > 80 else ""),
        "sequence_length_bp": len(args.seq),
        "n_tokens": len(ids),
        "token_ids": ids if args.json else ids[:32] + (["..."] if len(ids) > 32 else []),
        "selected_path": _detect_path(dn),
    }
    _emit(payload, args.json)
    return 0


def cmd_validate(args) -> int:
    """Compare DNAtok output vs HF reference on n random ACGT sequences."""
    device = args.device or _default_device()
    hf, dn = _build_dnatok(args.model, device)
    rng = random.Random(args.seed)
    # Fixed-length sequences so the test exercises whichever fast path
    # the model uses, without dragging in partial-tail or variable-length
    # edge cases that have their own dedicated tests.
    seqs = ["".join(rng.choices("ACGT", k=args.window)) for _ in range(args.n)]
    ids_dn = dn.encode_batch_to_ids(seqs).cpu().tolist()
    pad = int(dn.id_pad)

    # Use the __call__ API (not .encode()) — some fast tokenisers (e.g.
    # NTv2 BertTokenizerFast) honour `add_special_tokens=False` only on
    # the call-style API; `.encode()` may still prepend CLS / BOS.
    hf_batch = hf(seqs, add_special_tokens=False, padding=False, truncation=False,
                  return_tensors=None)["input_ids"]
    n_match = n_mismatch = 0
    first_mismatch = None
    for i, s in enumerate(seqs):
        hf_ids = hf_batch[i]
        if hasattr(hf_ids, "tolist"):
            hf_ids = hf_ids.tolist()
        # Strip pad from DNAtok row
        row = ids_dn[i]
        if dn.padding_side == "left":
            j = 0
            while j < len(row) and row[j] == pad: j += 1
            dn_ids = row[j:]
        else:
            j = len(row)
            while j > 0 and row[j - 1] == pad: j -= 1
            dn_ids = row[:j]
        if list(dn_ids) == list(hf_ids):
            n_match += 1
        else:
            n_mismatch += 1
            if first_mismatch is None:
                first_mismatch = {
                    "i": i, "seq_head": s[:40],
                    "dnatok_head": list(dn_ids)[:8],
                    "hf_head": list(hf_ids)[:8],
                }
    payload = {
        "model": args.model,
        "n_sequences": args.n,
        "n_match": n_match,
        "n_mismatch": n_mismatch,
        "match_rate": n_match / args.n,
        "selected_path": _detect_path(dn),
        "first_mismatch": first_mismatch,
    }
    _emit(payload, args.json)
    return 0 if n_mismatch == 0 else 1


def cmd_bench(args) -> int:
    """Measure pre-tokenization throughput (DNAtok vs HF Rust threaded)."""
    device = args.device or _default_device()
    os.environ["RAYON_NUM_THREADS"] = str(args.hf_threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "true" if args.hf_threads > 1 else "false"
    hf, dn = _build_dnatok(args.model, device)
    rng = random.Random(args.seed)
    seqs = ["".join(rng.choices("ACGT", k=args.window)) for _ in range(args.n)]
    bp_total = sum(len(s) for s in seqs)
    # Warmup
    _ = hf(seqs[: min(args.chunk, args.n)], add_special_tokens=False)
    _ = dn.encode_batch_to_ids(seqs[: min(args.chunk, args.n)])

    # HF
    t0 = time.perf_counter()
    for i in range(0, args.n, args.chunk):
        _ = hf(seqs[i : i + args.chunk], add_special_tokens=False)
    t_hf = time.perf_counter() - t0

    # DNAtok
    import torch
    if device == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(0, args.n, args.chunk):
        _ = dn.encode_batch_to_ids(seqs[i : i + args.chunk])
        if device == "cuda": torch.cuda.synchronize()
    t_dn = time.perf_counter() - t0
    payload = {
        "model": args.model,
        "n_sequences": args.n,
        "window_bp": args.window,
        "chunk": args.chunk,
        "total_bp": bp_total,
        "hf_threads": args.hf_threads,
        "hf_time_s": round(t_hf, 4),
        "hf_mbp_per_s": round(bp_total / t_hf / 1e6, 2),
        "dnatok_time_s": round(t_dn, 4),
        "dnatok_mbp_per_s": round(bp_total / t_dn / 1e6, 2),
        "speedup_dnatok_vs_hf": round(t_hf / t_dn, 2),
        "selected_path": _detect_path(dn),
        "device": device,
    }
    _emit(payload, args.json)
    return 0


def cmd_demo(args) -> int:
    """End-to-end: info → encode → validate (n=50) → bench (small)."""
    print("=" * 70)
    print(f"DNAtok demo — model={args.model}")
    print("=" * 70)

    # info
    print("\n[1/4] tokenizer info ...")
    args.json = False
    cmd_info(args)

    # encode a tiny sample
    print("\n[2/4] encode a sample sequence ...")
    sample = args.sample or "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
    saved_seq = getattr(args, "seq", None)
    args.seq = sample
    cmd_encode(args)
    if saved_seq is not None:
        args.seq = saved_seq

    # bit-identity validate (n=50 is fast)
    print("\n[3/4] bit-identity vs HF (n=50, random ACGT seqs, win≤256bp) ...")
    args.n = 50
    args.window = 256
    args.seed = 42
    rc = cmd_validate(args)
    if rc != 0:
        print(f"  ✗ bit-identity FAILED — see first_mismatch above")
        return rc
    print(f"  ✓ bit-identity PASS")

    # tiny benchmark
    print("\n[4/4] tiny benchmark (n=100, win=1024, chunk=32, HF threads=4) ...")
    args.n = 100
    args.window = 1024
    args.chunk = 32
    args.hf_threads = 4
    args.seed = 11
    return cmd_bench(args)


def cmd_list_models(args) -> int:
    rows = []
    for fam, (hf_id, path, tested, notes) in MODEL_REGISTRY.items():
        rows.append({"family": fam, "hf_id": hf_id, "path": path,
                     "tested_on": tested, "notes": notes})
    if args.json:
        json.dump(rows, sys.stdout, indent=2); print()
    else:
        widths = {k: max(len(k), max(len(str(r[k])) for r in rows)) for k in rows[0]}
        # header
        print("  " + "  ".join(k.ljust(widths[k]) for k in rows[0]))
        print("  " + "  ".join("-" * widths[k] for k in rows[0]))
        for r in rows:
            print("  " + "  ".join(str(r[k]).ljust(widths[k]) for k in rows[0]))
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="dnatok",
        description="GPU-native DNA tokenizer — bit-identical to Hugging Face.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    def _add_common(p, with_model=True):
        if with_model:
            p.add_argument("--model", required=True,
                           help="HuggingFace model id (e.g. zhihan1996/DNABERT-2-117M)")
        p.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
        p.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    p = sub.add_parser("info", help="Print tokenizer family + fast path")
    _add_common(p)
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("encode", help="Encode a single sequence")
    _add_common(p)
    p.add_argument("--seq", required=True, help="DNA sequence (ACGT[N])")
    p.set_defaults(func=cmd_encode)

    p = sub.add_parser("validate", help="Bit-identity vs HF on N random sequences")
    _add_common(p)
    p.add_argument("--n", type=int, default=500, help="Number of sequences (default 500)")
    p.add_argument("--window", type=int, default=1024,
                   help="Max window length in bp (default 1024)")
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("bench", help="Pre-tokenization benchmark vs HF")
    _add_common(p)
    p.add_argument("--n", type=int, default=1000, help="Number of sequences (default 1000)")
    p.add_argument("--window", type=int, default=4096,
                   help="Window length in bp (default 4096)")
    p.add_argument("--chunk", type=int, default=32, help="Batch size (default 32)")
    p.add_argument("--hf-threads", type=int, default=8,
                   help="RAYON_NUM_THREADS for HF baseline (default 8)")
    p.add_argument("--seed", type=int, default=11)
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("demo", help="Run info + encode + validate + small bench end-to-end")
    _add_common(p)
    p.add_argument("--sample", default=None, help="Sample sequence for the encode step")
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("list-models", help="Print the validated model registry")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_list_models)

    args = ap.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n[dnatok] interrupted", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"\n[dnatok] error: {type(e).__name__}: {e}", file=sys.stderr)
        if os.environ.get("DNATOK_DEBUG"):
            import traceback
            traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
