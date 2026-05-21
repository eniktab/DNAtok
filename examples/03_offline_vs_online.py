#!/usr/bin/env python3
"""DNAtok with HuggingFace cache — online (default) vs offline (Gadi).

Two deployment patterns covered in this example:

  ONLINE (default, e.g. laptop or compute node with internet):
      docker run --rm --gpus all \\
          -v ~/.cache/huggingface:/work/.hf-cache \\
          dnatok:dev \\
          dnatok demo --model zhihan1996/DNABERT-2-117M
      # On first run, transformers downloads weights to /work/.hf-cache.
      # Subsequent runs read from the mounted cache (no re-download).

  OFFLINE (e.g. NCI Gadi, no internet from compute nodes):
      # 1) From a login node WITH internet, pre-download the model:
      ssh gadi
      module load python3/...
      export HF_HOME=/g/data/te53/<account>/data/scratch/hf-cache
      python3 -c "
      from transformers import AutoTokenizer
      AutoTokenizer.from_pretrained('zhihan1996/DNABERT-2-117M', trust_remote_code=True)"

      # 2) On the compute node, run with HF_HUB_OFFLINE=1:
      apptainer exec --nv \\
          --bind /g/data/te53/<account>/data/scratch/hf-cache:/work/.hf-cache \\
          --env HF_HOME=/work/.hf-cache \\
          --env HF_HUB_OFFLINE=1 \\
          --env TRANSFORMERS_OFFLINE=1 \\
          dnatok.sif dnatok demo --model zhihan1996/DNABERT-2-117M

The DNATok library itself does not force offline mode; you control it
via env vars. This Python example shows the same logic from Python.
"""
import argparse, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="zhihan1996/DNABERT-2-117M")
    ap.add_argument("--offline", action="store_true",
                    help="Force HF_HUB_OFFLINE=1 (Gadi-style; weights must be pre-cached)")
    ap.add_argument("--cache-dir", default=os.environ.get(
        "HF_HOME", str(Path.home() / ".cache" / "huggingface")))
    args = ap.parse_args()

    # ------------------------------------------------------------------
    # Mode selection — set BEFORE importing transformers.
    # ------------------------------------------------------------------
    os.environ["HF_HOME"] = args.cache_dir
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        print(f"[mode]  OFFLINE  (cache={args.cache_dir})")
    else:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        print(f"[mode]  ONLINE  (cache={args.cache_dir})")

    # ------------------------------------------------------------------
    # Standard DNATok bring-up — identical between modes.
    # ------------------------------------------------------------------
    import dnatok_compat  # noqa: F401, must precede transformers
    import torch
    from transformers import AutoTokenizer
    from dna_tokenizer import DNATok

    try:
        hf = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    except OSError as e:
        if args.offline:
            print(f"[error] OFFLINE mode requires pre-cached weights for {args.model}.")
            print(f"        Run once with --no-offline (online) to download, then retry.")
            print(f"        Or copy a pre-existing $HF_HOME directory into {args.cache_dir}.")
        else:
            print(f"[error] Could not download {args.model}: {e}")
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"

    class _Emb:
        def __init__(self, t):
            self.tokenizer = t
            v = int(getattr(t, "vocab_size", 0)) or len(t.get_vocab())
            self.embed_table = torch.nn.Embedding(v + 4, 16).to(device)
        def embed_tokens(self, ids): return self.embed_table(ids)

    dn = DNATok(_Emb(hf), normalize_case=False, handle_invalid_chars=False)
    dn.discover()
    ids = dn.encode_batch_to_ids(["ACGTACGTACGTACGTACGT"])
    print(f"[ok]    {args.model}: shape={tuple(ids.shape)} dtype={ids.dtype} device={device}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
