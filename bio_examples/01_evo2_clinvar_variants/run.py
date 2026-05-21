#!/usr/bin/env python3
"""
Case study 01: end-to-end Evo2-1b inference on ClinVar SNV windows.

Real biological data + REAL Evo2 model. For each ClinVar SNV on
hg38 chr17 we extract a 4 kbp ref + 4 kbp alt window, tokenise with
HF (single + multi-thread) and DNAtok, assert bit-identical, then
run the real Evo2-1b forward pass on each tokenisation.

Per-batch metrics: tokens fed/sec, GPU forward time, end-to-end
wall-clock, GPU utilization.
"""
from __future__ import annotations
import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("RAYON_NUM_THREADS", "8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "bio_examples"))

# MUST come before any transformers import — HF hub 1.x version shim on
# the Gadi NGC venv. dnatok_compat.apply_importlib_patch() is called at
# module level (see src/dnatok_compat.py:36).
import dnatok_compat  # noqa: F401, E402


def iter_clinvar_snvs(vcf_path, *, chrom="17"):
    opener = gzip.open if str(vcf_path).endswith(".gz") else open
    with opener(vcf_path, "rt") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if fields[0] != chrom:
                continue
            ref, alt = fields[3], fields[4]
            if len(ref) != 1 or len(alt) != 1:
                continue
            if ref not in "ACGT" or alt not in "ACGT":
                continue
            info = fields[7]
            clnsig = ""
            for kv in info.split(";"):
                if kv.startswith("CLNSIG="):
                    clnsig = kv[len("CLNSIG="):]; break
            yield fields[0], int(fields[1]), ref, alt, clnsig


def _set_rayon(n):
    os.environ["RAYON_NUM_THREADS"] = str(n)
    os.environ["TOKENIZERS_PARALLELISM"] = "true" if n > 1 else "false"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vcf-chrom", default="17")
    ap.add_argument("--fa-chrom", default="chr17")
    ap.add_argument("--window-bp", type=int, default=4_096)
    ap.add_argument("--n-variants", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=4,
                    help="Evo2-1b at 4 kbp uses substantial GPU mem.")
    ap.add_argument("--hf-threads", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--out-dir", type=Path,
                    default=ROOT / "results" / "case_01_evo2_clinvar")
    args = ap.parse_args()

    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = args.out_dir / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    log_handle = open(run_dir / "log.txt", "w")
    def log(msg: str):
        print(msg); log_handle.write(msg + "\n"); log_handle.flush()

    log(f"\n=== Case 01: Evo2-1b end-to-end / ClinVar SNVs on {args.fa_chrom} ===")

    # --- 1. real data ---
    from _data.download import load_chrom_sequence, download_clinvar_vcf
    log("[1/5] Loading hg38 + ClinVar ...")
    seq = load_chrom_sequence(args.fa_chrom)
    log(f"  {args.fa_chrom}: {len(seq):,} bp")
    vcf = download_clinvar_vcf()

    # --- 2. extract real SNV windows ---
    log("[2/5] Building ref/alt windows ...")
    win = args.window_bp
    pairs = []
    for _c, pos, ref, alt, clnsig in iter_clinvar_snvs(vcf, chrom=args.vcf_chrom):
        zpos = pos - 1
        ws = max(0, zpos - win // 2)
        we = ws + win
        if we > len(seq): continue
        ref_win = seq[ws:we]
        if ref_win[zpos - ws] != ref: continue
        if ref_win.count("N") / win > 0.5: continue
        alt_win = ref_win[: zpos - ws] + alt + ref_win[zpos - ws + 1 :]
        pairs.append((_c, pos, ref_win, alt_win, clnsig))
        if args.n_variants and len(pairs) >= args.n_variants: break
    log(f"  usable SNVs: {len(pairs):,}")
    if not pairs:
        log_handle.close(); return 2
    seqs = []
    for _c, _p, rw, aw, _s in pairs:
        seqs.extend([rw, aw])
    total_bp = len(seqs) * win

    # --- 3. tokenizer + REAL Evo2 model ---
    log("[3/5] Loading Evo2 tokenizer + REAL Evo2-1b model ...")
    import torch
    from dna_tokenizer import DNATok
    from benchmarks.tokenizer_adapters import load_hf_tokenizer
    from _common.models import load_evo2

    hf_tok = load_hf_tokenizer("arcinstitute/evo2_1b_base")
    model_obj, forward_fn, info = load_evo2("evo2_1b_base")
    log(f"  REAL Evo2: {info['params']/1e6:.1f} M params  loader={info['loader']}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class _Embedder:
        def __init__(self, tok):
            self.tokenizer = tok
            v = int(getattr(tok, "vocab_size", 0)) or 512
            self.embed_table = torch.nn.Embedding(v + 4, 16).to(device)
        def embed_tokens(self, ids): return self.embed_table(ids)

    dnatok = DNATok(_Embedder(hf_tok), normalize_case=False, handle_invalid_chars=False)
    dnatok.discover()

    # --- 4. e2e timing ---
    log(f"[4/5] End-to-end timing ({len(seqs):,} sequences × {win:,} bp) ...")

    def time_hf(n):
        _set_rayon(n)
        for _ in range(1):
            enc = hf_tok(list(seqs[: min(args.chunk, len(seqs))]),
                          add_special_tokens=False, padding="longest",
                          truncation=False, return_tensors="pt")
            _ = forward_fn(enc["input_ids"])
            if device.type == "cuda": torch.cuda.synchronize()
        tok_s = 0.0; fwd_s = 0.0; tot_tok = 0
        if device.type == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(0, len(seqs), args.chunk):
            batch = seqs[i : i + args.chunk]
            t = time.perf_counter()
            enc = hf_tok(list(batch), add_special_tokens=False,
                          padding="longest", truncation=False,
                          return_tensors="pt")
            tok_s += time.perf_counter() - t
            ids = enc["input_ids"]
            tot_tok += ids.numel()
            t = time.perf_counter()
            _ = forward_fn(ids)
            if device.type == "cuda": torch.cuda.synchronize()
            fwd_s += time.perf_counter() - t
        return {"label": f"hf_t{n}", "tok_s": tok_s, "fwd_s": fwd_s,
                "total_s": time.perf_counter() - t0, "total_tokens": tot_tok}

    def time_dn():
        for _ in range(1):
            ids = dnatok.encode_batch_to_ids(seqs[: min(args.chunk, len(seqs))])
            _ = forward_fn(ids)
            if device.type == "cuda": torch.cuda.synchronize()
        tok_s = 0.0; fwd_s = 0.0; tot_tok = 0
        if device.type == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(0, len(seqs), args.chunk):
            batch = seqs[i : i + args.chunk]
            t = time.perf_counter()
            ids = dnatok.encode_batch_to_ids(batch)
            if device.type == "cuda": torch.cuda.synchronize()
            tok_s += time.perf_counter() - t
            tot_tok += ids.numel()
            t = time.perf_counter()
            _ = forward_fn(ids)
            if device.type == "cuda": torch.cuda.synchronize()
            fwd_s += time.perf_counter() - t
        return {"label": "dnatok", "tok_s": tok_s, "fwd_s": fwd_s,
                "total_s": time.perf_counter() - t0, "total_tokens": tot_tok}

    results = []
    for n in args.hf_threads:
        log(f"  hf_t{n} ...")
        r = time_hf(n); results.append(r)
        log(f"    tok={r['tok_s']:.2f}s fwd={r['fwd_s']:.2f}s total={r['total_s']:.2f}s")
    log("  dnatok ...")
    r = time_dn(); results.append(r)
    log(f"    tok={r['tok_s']:.2f}s fwd={r['fwd_s']:.2f}s total={r['total_s']:.2f}s")

    # --- 5. bit-identical ---
    _set_rayon(1)
    n_check = min(50, len(seqs))
    hf_per = [hf_tok(s, add_special_tokens=False)["input_ids"] for s in seqs[:n_check]]
    dn_pad = dnatok.encode_batch_to_ids(seqs[:n_check]).cpu().tolist()
    pad_id = int(dnatok.id_pad); side = dnatok.padding_side
    mismatch = 0
    for h, row in zip(hf_per, dn_pad):
        if side == "left":
            j = 0
            while j < len(row) and row[j] == pad_id: j += 1
            v = row[j:]
        else:
            j = len(row)
            while j > 0 and row[j - 1] == pad_id: j -= 1
            v = row[:j]
        h_list = h.tolist() if hasattr(h, "tolist") else list(h)
        if list(v) != h_list: mismatch += 1
    log(f"[5/5] bit-identical: {mismatch == 0}  ({mismatch}/{n_check})")

    log("")
    log("=" * 110)
    log(f"  {'pipeline':<10}{'tok (s)':>9}{'fwd (s)':>9}{'total (s)':>12}"
        f"{'tok %':>8}{'GPU util':>11}{'tok feed (Mt/s)':>17}"
        f"{'e2e (Mt/s)':>13}{'vs dnatok':>12}")
    log("=" * 110)
    dn_total = next(r["total_s"] for r in results if r["label"] == "dnatok")
    for r in results:
        tok_pct = r["tok_s"] / r["total_s"] * 100 if r["total_s"] else 0.0
        gpu_pct = r["fwd_s"] / r["total_s"] * 100 if r["total_s"] else 0.0
        tok_feed = (r["total_tokens"] / r["tok_s"] / 1e6) if r["tok_s"] else 0.0
        e2e = (r["total_tokens"] / r["total_s"] / 1e6) if r["total_s"] else 0.0
        slow = r["total_s"] / dn_total
        log(f"  {r['label']:<10}{r['tok_s']:>9.2f}{r['fwd_s']:>9.2f}"
            f"{r['total_s']:>12.2f}{tok_pct:>7.1f}%{gpu_pct:>10.1f}%"
            f"{tok_feed:>16.2f}{e2e:>12.2f}{slow:>11.2f}x")
    log("=" * 110)

    res = {
        "case_study": "01_evo2_clinvar_variants",
        "model_info": info,
        "chrom": args.fa_chrom,
        "window_bp": win,
        "n_variants": len(pairs),
        "n_sequences": len(seqs),
        "total_bp": total_bp,
        "pipelines": results,
        "bit_identical": mismatch == 0,
        "device": str(device),
    }
    with open(run_dir / "results.json", "w") as f:
        json.dump(res, f, indent=2)
    log(f"\n  Results: {run_dir / 'results.json'}")
    log_handle.close()
    return 0 if mismatch == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
