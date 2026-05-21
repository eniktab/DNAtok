#!/usr/bin/env python3
"""DNAtok with ANY supported HuggingFace genomic model — auto-discover.

The same six-line wrapper handles all seven supported families:

    python3 examples/02_any_model.py --model zhihan1996/DNABERT-2-117M
    python3 examples/02_any_model.py --model AIRI-Institute/gena-lm-bert-base-t2t
    python3 examples/02_any_model.py --model metagene-ai/METAGENE-1
    python3 examples/02_any_model.py --model InstaDeepAI/NTv3_8M_pre
    python3 examples/02_any_model.py --model InstaDeepAI/nucleotide-transformer-v2-50m-multi-species
    python3 examples/02_any_model.py --model LongSafari/hyenadna-tiny-1k-seqlen-hf
    python3 examples/02_any_model.py --model arcinstitute/evo2_1b_base

For each, DNAtok auto-selects the optimal fast path (ASCII LUT, k-mer
LUT, GPU BPE kernel, or fall-through), verifies bit-identical output to
the upstream Hugging Face tokenizer on a small held-out sample, and
prints the throughput.
"""
import argparse, random, sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dnatok_compat  # noqa: F401
from dna_tokenizer import DNATok


def _load_tokenizer(model_id: str):
    if "evo2" in model_id.lower():
        from benchmarks.tokenizer_adapters import load_hf_tokenizer
        return load_hf_tokenizer(model_id)
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


def _detect_path(dn) -> str:
    if dn.kmer_k is not None: return f"kmer-{dn.kmer_k}"
    if dn.ascii_lut is not None: return "ascii-lut"
    if dn.bpe_backend is not None: return "gpu-bpe-kernel"
    if dn.lmm_bpe is not None: return "cached-lmm"
    return "hf-fallback"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--len", dest="length", type=int, default=1024)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hf = _load_tokenizer(args.model)

    class _Emb:
        def __init__(self, t):
            self.tokenizer = t
            v = int(getattr(t, "vocab_size", 0)) or len(t.get_vocab())
            self.embed_table = torch.nn.Embedding(v + 4, 16).to(device)
        def embed_tokens(self, ids): return self.embed_table(ids)

    dn = DNATok(_Emb(hf), normalize_case=False, handle_invalid_chars=False)
    dn.discover()

    print(f"model       {args.model}")
    print(f"device      {device}")
    print(f"path        {_detect_path(dn)}")

    rng = random.Random(0)
    seqs = ["".join(rng.choices("ACGT", k=args.length)) for _ in range(args.n)]

    _ = dn.encode_batch_to_ids(seqs[:8])
    if device == "cuda": torch.cuda.synchronize()

    t0 = time.perf_counter()
    hf_ids = [hf.encode(s, add_special_tokens=False) for s in seqs]
    t_hf = time.perf_counter() - t0

    if device == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    ids = dn.encode_batch_to_ids(seqs)
    if device == "cuda": torch.cuda.synchronize()
    t_dn = time.perf_counter() - t0

    pad = int(dn.id_pad); side = dn.padding_side
    mismatch = 0
    for i, row in enumerate(ids.cpu().tolist()):
        if side == "left":
            j = 0
            while j < len(row) and row[j] == pad: j += 1
            v = row[j:]
        else:
            j = len(row)
            while j > 0 and row[j - 1] == pad: j -= 1
            v = row[:j]
        h = list(hf_ids[i].tolist()) if hasattr(hf_ids[i], "tolist") else list(hf_ids[i])
        if v != h: mismatch += 1

    bp = args.n * args.length
    print(f"hf          {t_hf*1000:>8.1f} ms  ({bp/t_hf/1e6:>6.1f} Mbp/s)")
    print(f"dnatok      {t_dn*1000:>8.1f} ms  ({bp/t_dn/1e6:>6.1f} Mbp/s)")
    print(f"speedup     {t_hf/t_dn:>8.2f}x")
    print(f"bit-id      {'PASS' if mismatch == 0 else f'FAIL ({mismatch}/{args.n})'}")
    return 0 if mismatch == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
