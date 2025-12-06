
import os
import sys
import time
import pathlib
import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForMaskedLM, AutoModelForCausalLM, AutoTokenizer
from datetime import datetime

# Add src to path
ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

RESULTS_DIR = ROOT / "results"

from dna_tokenizer import DNATok

# --- Adapters ---

class NTAdapter(torch.nn.Module):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.pad_token_id = getattr(tokenizer, "pad_token_id", 0)
        self.max_position_embeddings = getattr(getattr(model, "config", None), "max_position_embeddings", None)
        self._embed = self._resolve_embed()

    def _resolve_embed(self):
        for attr in ("embed_tokens", "tok_embeddings"):
            fn = getattr(self.model, attr, None)
            if callable(fn):
                return fn
        getter = getattr(self.model, "get_input_embeddings", None)
        if callable(getter):
            emb = getter()
            if callable(emb):
                return emb
        return None

    def embed_tokens(self, input_ids, rc_invariant=False):
        if self._embed is None:
            raise RuntimeError("NTAdapter could not find input embedding layer.")
        return self._embed(input_ids)

class HyenaAdapter(torch.nn.Module):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.pad_token_id = getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "pad_id", 0)
        inner = getattr(model, "model", model)
        self.max_position_embeddings = getattr(getattr(model, "config", None), "max_position_embeddings", None) or getattr(inner, "max_position_embeddings", None)
        self._embed = self._resolve_embed(inner)

    def _resolve_embed(self, inner):
        for attr in ("embed_tokens", "tok_embeddings"):
            fn = getattr(inner, attr, None)
            if callable(fn):
                return fn
        getter = getattr(inner, "get_input_embeddings", None) or getattr(self.model, "get_input_embeddings", None)
        if callable(getter):
            emb = getter()
            if callable(emb):
                return emb
        return None

    def embed_tokens(self, input_ids, rc_invariant=False):
        if self._embed is None:
            raise RuntimeError("HyenaAdapter could not find input embedding layer.")
        return self._embed(input_ids)

class Evo2Adapter(torch.nn.Module):
    def __init__(self, evo2_model, layer_name=None):
        super().__init__()
        self.model = evo2_model
        self.tokenizer = evo2_model.tokenizer
        self.pad_token_id = getattr(self.tokenizer, "pad_id", 0)
        inner = getattr(evo2_model, "model", None)
        self.max_position_embeddings = getattr(inner, "max_position_embeddings", None)
        self._embed = self._resolve_embed_fn(evo2_model)
        self._layer_name = layer_name or self._infer_layer_name(inner or evo2_model)

    def _resolve_embed_fn(self, evo2_model):
        for obj in (evo2_model, getattr(evo2_model, "model", None)):
            if obj is None: continue
            for attr in ("embed_tokens", "tok_embeddings"):
                fn = getattr(obj, attr, None)
                if callable(fn): return fn
            getter = getattr(obj, "get_input_embeddings", None)
            if callable(getter):
                emb = getter()
                if callable(emb): return emb
            try:
                for name, module in obj.named_modules():
                    if "embed" in name and "token" in name and callable(module):
                        return module
            except Exception: pass
        return None

    def _infer_layer_name(self, obj):
        try:
            keys = list(obj.state_dict().keys())
        except Exception: return None
        for k in keys:
            if "embed" in k and k.endswith("weight"):
                return k.rsplit(".weight", 1)[0]
        return None

    def embed_tokens(self, input_ids, rc_invariant=False):
        if self._embed is not None:
            return self._embed(input_ids)
        if not self._layer_name:
            raise RuntimeError("Could not find Evo2 embedding layer.")
        _, embeds = self.model(input_ids, return_embeddings=True, layer_names=[self._layer_name])
        out = embeds[self._layer_name]
        return out.to(input_ids.device) if out.device != input_ids.device else out

# --- Helpers ---

def make_random_seqs(b: int, T: int, seed: int = 0) -> list[str]:
    rng = np.random.default_rng(seed)
    alphabet = np.array(list("ACGT"))
    return ["".join(rng.choice(alphabet, size=T)) for _ in range(b)]

def find_model_path(model_name, env_var):
    env_override = os.environ.get(env_var)
    if env_override:
        p = pathlib.Path(env_override).expanduser().resolve()
        return p if p.exists() else None
    
    env_home = os.environ.get("HF_HOME") or os.environ.get("HF_HUB_CACHE")
    if env_home:
        base = pathlib.Path(env_home).expanduser()
        # Check standard locations
        candidates = [
            base / model_name,
            base / "models" / model_name,
            base / "hub" / f"models--{model_name.replace('/', '--')}" / "snapshots"
        ]
        for candidate in candidates:
            if candidate.exists():
                if candidate.name == "snapshots":
                    snaps = sorted(candidate.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
                    if snaps: return snaps[0]
                return candidate
    
    repo_fallback = (ROOT / "hf-cache" / "models" / model_name).resolve()
    return repo_fallback if repo_fallback.exists() else None

# --- Benchmark Logic ---

def run_benchmark_scenario(name, adapter, batch_size, seq_len, device, results_list):
    # Setup DNATok first to detect k-mer structure
    tok_helper = DNATok(adapter)
    tok_helper.ids_max_tokens_per_call = batch_size * seq_len
    tok_helper.force_fp32_outputs = False  # favor throughput for fair speedup comparisons
    tok_helper.discover()
    
    # Adjust seq_len if k-mer structure detected
    if tok_helper.kmer_k is not None:
        k = tok_helper.kmer_k
        if seq_len % k != 0:
            old_len = seq_len
            seq_len = (seq_len // k) * k
            print(f"   [Adjusted seq_len {old_len}->{seq_len} for k={k}]")
    
    if tok_helper.kmer_lut is not None:
        print(f"   [DNATok K-mer LUT: min={tok_helper.kmer_lut.min()}, max={tok_helper.kmer_lut.max()}]")
        # Try to resolve embedding weight for check
        emb_layer = adapter._embed
        if hasattr(emb_layer, "weight"):
             vocab_size = emb_layer.weight.shape[0]
             if tok_helper.kmer_lut.max() >= vocab_size:
                 print(f"   [WARNING] DNATok LUT max {tok_helper.kmer_lut.max()} >= Embedding size {vocab_size}")
        else:
             print(f"   [INFO] Could not resolve embedding weight for size check.")

    print(f"   Running {name} (B={batch_size}, T={seq_len})...")
    seqs = make_random_seqs(batch_size, seq_len)
    
    # 1. Baseline (average of 3 runs for stability)
    def _baseline_once():
        t0 = time.perf_counter()
        # Tokenize (handle Evo2 which might not have encode)
        if hasattr(adapter.tokenizer, "encode"):
            tok_ids_list = [adapter.tokenizer.encode(s, add_special_tokens=False) for s in seqs]
        else:
            # Fallback for Evo2 or others
            tok_ids_list = []
            for s in seqs:
                out = adapter.tokenizer.tokenize(s)
                # Evo2 tokenize returns list of strings or ids?
                if isinstance(out, list) and len(out) > 0 and isinstance(out[0], str):
                    out = adapter.tokenizer.convert_tokens_to_ids(out)
                tok_ids_list.append(out)
        tok_ids = torch.tensor(tok_ids_list, dtype=torch.long)
        hf_ids = tok_ids.to(device)
        hf_emb = adapter.embed_tokens(hf_ids)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        return t1 - t0, hf_ids

    baseline_times: list[float] = []
    hf_ids = None
    tok_len = None
    for _ in range(3):
        t, ids_ref = _baseline_once()
        baseline_times.append(t)
        hf_ids = ids_ref
        tok_len = ids_ref.shape[1]
    baseline_time = sum(baseline_times) / len(baseline_times)
    if hf_ids is not None and hf_ids.numel() > 0:
        print(f"   [Baseline IDs: min={hf_ids.min().item()}, max={hf_ids.max().item()}]")
    print(f"   [Baseline (avg of 3) finished in {baseline_time:.4f}s]")
    
    # Set token_len for DNATok consistency
    if tok_helper.use_ids_path and tok_len is not None:
        tok_helper.token_len = tok_len

    # 2. DNATok (try multiple paths and pick best)
    def _run_path(path: str):
        times: list[float] = []
        for _ in range(3):
            t_start = time.perf_counter()
            chunks = list(tok_helper.embed_from_strings(seqs, emb_batch=batch_size, device=device, path=path))
            _ = torch.cat(chunks, dim=0)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_end = time.perf_counter()
            times.append(t_end - t_start)
        return sum(times) / len(times)

    candidate_paths = ["bytes", "ids"] if tok_helper.use_ids_path else ["ids"]
    timings = {}
    for path in candidate_paths:
        try:
            timings[path] = _run_path(path)
            print(f"   [DNATok {path} path finished in {timings[path]:.4f}s]")
        except Exception as e:
            print(f"   [DNATok {path} path failed: {e}]")
    if not timings:
        print("   [DNATok: no successful paths]")
        return

    best_path = min(timings, key=timings.get)
    dnatok_time = timings[best_path]
    speedup = baseline_time / dnatok_time if dnatok_time > 0 else 0
    print(f"   [DNATok best: {best_path} in {dnatok_time:.4f}s; speedup {speedup:.2f}x]")

    results_list.append({
        "Model": adapter.__class__.__name__.replace("Adapter", ""),
        "Scenario": name,
        "Batch": batch_size,
        "SeqLen": seq_len,
        "Path": best_path,
        "Baseline (s)": baseline_time,
        "DNATok (s)": dnatok_time,
        "Speedup": speedup
    })

def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Running benchmarks on {device}...")
    
    results = []
    
    # 1. Nucleotide Transformer
    nt_name = "nucleotide-transformer-2.5b-1000g"
    nt_path = find_model_path(nt_name, "NT_MODEL_PATH")
    if nt_path:
        print(f"\nLoading Nucleotide Transformer from {nt_path}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(nt_path), trust_remote_code=True, local_files_only=True)
            model = AutoModelForMaskedLM.from_pretrained(str(nt_path), trust_remote_code=True, local_files_only=True, dtype=torch.float16 if "cuda" in device else None).to(device)
            
            # Resize embeddings if needed (fix for CUDA device-side assert)
            # Calculate true max ID
            max_id = len(tokenizer) - 1
            if hasattr(tokenizer, "vocab"):
                max_id = max(max_id, max(tokenizer.vocab.values()))
            elif hasattr(tokenizer, "get_vocab"):
                max_id = max(max_id, max(tokenizer.get_vocab().values()))
            
            vocab_size = max_id + 1
            print(f"   [Tokenizer: len={len(tokenizer)}, max_id={max_id}, computed_vocab_size={vocab_size}]")

            if vocab_size > model.get_input_embeddings().num_embeddings:
                print(f"   [Resizing embeddings {model.get_input_embeddings().num_embeddings} -> {vocab_size}]")
                model.resize_token_embeddings(vocab_size)
            
            adapter = NTAdapter(model, tokenizer).to(device)
            
            run_benchmark_scenario("Latency", adapter, 1, 1024, device, results)
            run_benchmark_scenario("Throughput", adapter, 32, 1024, device, results)
            run_benchmark_scenario("LongSeq", adapter, 1, 4096, device, results)
            
            del model, adapter, tokenizer
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Failed to load/run NT: {e}")
    else:
        print(f"\nSkipping Nucleotide Transformer (not found).")

    # 2. HyenaDNA
    hyena_name = "hyenadna-small-32k-seqlen-hf"
    hyena_path = find_model_path(hyena_name, "HYENA_MODEL_PATH")
    if hyena_path:
        print(f"\nLoading HyenaDNA from {hyena_path}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(hyena_path), trust_remote_code=True, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(str(hyena_path), trust_remote_code=True, local_files_only=True, dtype=torch.float16 if "cuda" in device else None).to(device)
            
            # Resize embeddings if needed
            max_id = len(tokenizer) - 1
            if hasattr(tokenizer, "vocab"):
                max_id = max(max_id, max(tokenizer.vocab.values()))
            elif hasattr(tokenizer, "get_vocab"):
                max_id = max(max_id, max(tokenizer.get_vocab().values()))
            vocab_size = max_id + 1
            
            if vocab_size > model.get_input_embeddings().num_embeddings:
                print(f"   [Resizing embeddings {model.get_input_embeddings().num_embeddings} -> {vocab_size}]")
                model.resize_token_embeddings(vocab_size)

            adapter = HyenaAdapter(model, tokenizer).to(device)
            
            run_benchmark_scenario("Latency", adapter, 1, 1024, device, results)
            run_benchmark_scenario("Throughput", adapter, 64, 1024, device, results)
            run_benchmark_scenario("LongSeq", adapter, 1, 8192, device, results)
            
            del model, adapter, tokenizer
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Failed to load/run Hyena: {e}")
    else:
        print(f"\nSkipping HyenaDNA (not found).")

    # 3. Evo2
    # Evo2 requires 'evo2' package and model name
    try:
        from evo2 import Evo2
        evo2_name = "evo2_7b"
        print(f"\nLoading Evo2 ({evo2_name})...")
        # Evo2 loading might fail if weights aren't present, usually it checks ~/.cache/evo2 or similar
        # We assume it works if the package is there, or we catch error
        try:
            evo2_model = Evo2(evo2_name)
            adapter = Evo2Adapter(evo2_model).to(device)
            
            run_benchmark_scenario("Latency", adapter, 1, 1024, device, results)
            run_benchmark_scenario("Throughput", adapter, 16, 1024, device, results) # Smaller batch for 7B model
            run_benchmark_scenario("LongSeq", adapter, 1, 4096, device, results)
            
            del evo2_model, adapter
            torch.cuda.empty_cache()
        except Exception as e:
             print(f"Failed to load/run Evo2: {e}")
    except ImportError:
        print(f"\nSkipping Evo2 (package not installed).")

    print("\n" + "="*80)
    print("REAL MODEL BENCHMARK SUMMARY")
    print("="*80)
    saved_paths = []
    if results:
        df = pd.DataFrame(results)
        df_display = df.copy()
        df_display["Baseline (s)"] = df_display["Baseline (s)"].map(lambda x: f"{x:.4f}")
        df_display["DNATok (s)"] = df_display["DNATok (s)"].map(lambda x: f"{x:.4f}")
        df_display["Speedup"] = df_display["Speedup"].map(lambda x: f"{x:.2f}x")
        print(df_display.to_string(index=False))

        csv_path = RESULTS_DIR / f"real_model_benchmark_{stamp}.csv"
        json_path = RESULTS_DIR / f"real_model_benchmark_{stamp}.json"
        df.to_csv(csv_path, index=False)
        df.to_json(json_path, orient="records", indent=2)
        saved_paths.extend([csv_path, json_path])
    else:
        print("No models were successfully benchmarked.")
    print("="*80)
    if saved_paths:
        print("Saved results to:")
        for p in saved_paths:
            print(f" - {p}")

if __name__ == "__main__":
    main()
