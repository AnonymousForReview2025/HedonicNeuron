#!/usr/bin/env python
"""
lora_weight_ablation_synergy.py  —  LOCAL‑LOGIT version
=======================================================
Computes three coalition‑level synergy metrics for a LoRA‑tuned transformer
layer.  Synergy is measured on **local logits**:

    local_logit(x,L) = score(  h_L(x)[:, -1, :]  )

where `h_L` is the hidden state *after* layer L and `score` is the model’s
classification head.  Ablation sets the requested LoRA‑B rows to zero, then the
drop in local logits is collected.

Metrics reported per coalition S (|S|=8 by default):

  • pair_avg_synergy(S)
  • total_ablation_ratio(S)
  • excess_loss(S)

Everything else (data loading, mixing, command‑line flags) is unchanged.
"""
import argparse, json, pickle, pathlib, random, sys, os, warnings
from typing import List, Dict, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftConfig, PeftModel
from huggingface_hub import login


torch.set_grad_enabled(False)

# ────────────────────────────────  Data  ────────────────────────────────────
def load_ms_marco_data(n_queries: int,
                       n_docs: int,
                       file_path: str) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for ln in f:
            _, _, q, d = ln.rstrip("\n").split("\t")[:4]
            if q not in out:
                if len(out) >= n_queries:
                    continue
                out[q] = [d]
            elif len(out[q]) < n_docs:
                out[q].append(d)
    return out

def load_dl19_data(n_queries: int,
                         n_docs: int,
                         file_path: str = "rerank_input.repllama.psd.dl19.jsonl") -> Dict[str, List[str]]:
    """Load DL19 data from a JSONL file - this file contains 43 queries with 200 docs each retreived from repllama"""
    out_qd: Dict[str, List[str]] = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            query = data["query"]
            doc = data["text"]

            if query not in out_qd:
                if len(out_qd) >= n_queries:
                    continue
                out_qd[query] = [doc]
            elif len(out_qd[query]) < n_docs:
                out_qd[query].append(doc)

    return out_qd

def tokenize_dataset(q2d: Dict[str, List[str]],
                     tok: AutoTokenizer,
                     batch_size: int,
                     max_length: int,
                     device: str):
    txt = [f"{q} </s> {d}" for q, ds in q2d.items() for d in ds]
    enc = tok(txt, padding=True, truncation=True,
              max_length=max_length, return_tensors="pt")
    return [{k: v[i:i+batch_size].to(device)
             for k, v in enc.items()}
            for i in range(0, enc["input_ids"].size(0), batch_size)]

# ───────────────────────────  Coalition helpers  ───────────────────────────
def load_coalitions(p: pathlib.Path):
    if p.suffix.lower() == ".json":
        return json.loads(p.read_text())
    if p.suffix.lower() in {".pkl", ".pickle"}:
        return pickle.loads(p.read_bytes())
    raise ValueError("Unsupported coalition file type")

# ─────────────────────────  Coalition helpers  ───────────────────────────
def mix_coalitions(
        coals: List[List[int]],
        max_neurons: int = 14_336,
        rng: random.Random | None = None
) -> List[List[int]]:
    """
    Replace the old “shuffle‑the‑same‑neurons” behaviour with fresh,
    non‑overlapping random coalitions that respect the original sizes.

    Parameters
    ----------
    coals : List[List[int]]
        Original coalitions (only their lengths are used).
    max_neurons : int, default 14 336
        Number of neurons available in the layer.
    rng : random.Random | None
        Optional dedicated RNG (helps with reproducibility).

    Returns
    -------
    List[List[int]]
        New coalitions with the same cardinalities as `coals`.
        Every neuron ID is < max_neurons and appears at most once.
    """
    rng = rng or random

    sizes = [len(g) for g in coals]
    needed = sum(sizes)
    if needed > max_neurons:
        raise ValueError(
            f"Requested {needed} distinct neurons but only {max_neurons} "
            "are available. Either raise --max_neurons or trim coalitions."
        )

    pool = list(range(max_neurons))
    rng.shuffle(pool)

    out, idx = [], 0
    for sz in sizes:
        out.append(pool[idx:idx + sz])
        idx += sz
    return out

# ─────────────────────────  LoRA masking utilities  ────────────────────────
def mask_rows_merged(layer, rows):
    saved = []
    with torch.no_grad():
        w = layer.mlp.gate_proj.weight# merged gate_proj is (4h,h)
        saved.append((w, rows, w[rows].clone()))
        w[rows] = 0
    return saved

def restore_lora_rows(saved):
    with torch.no_grad():
        for w, rows, orig in saved:
            w[rows] = orig

# ────────────────────  Local‑logit forward helper  ─────────────────────────
def run_model_local(model, layer, score_layer, batches):
    """Return concatenated local logits for all batches."""
    collected = []

    def hook(_, __, out):
        if isinstance(out, tuple):
            out = out[0]  # take the hidden states from (hidden, residual)

        # out: (B, seq_len, hidden)
        if out.ndim == 3:
            rep = out[:, -1, :]  # (B, H)
        elif out.ndim == 2:
            rep = out            # already (B, H)
        else:
            raise ValueError(f"Unexpected output shape from hook: {out.shape}")
                 # last token, GPT style
        logits = torch.nn.functional.linear(rep, score_layer.weight,
                                            score_layer.bias).squeeze(-1)
        collected.append(logits.detach().float().cpu())

    h = layer.register_forward_hook(hook)
    with torch.no_grad():
        for b in batches:
            model(**b)
    h.remove()
    return torch.cat(collected)

# ─────────────────────────────  Metrics (patched)  ───────────────────────────
def make_phi_fn(model, layer, score, batches, base_local):
    """
    Returns ψ‑style functions:
        φ([i])   ≈  E[ℓ - ℓ_{‑i}]
        φ([i,j]) ≈  E[ℓ - ℓ_{‑{i,j}}]
    We keep a tiny cache so repeated calls are cheap.
    """
    single_cache: Dict[int, float] = {}
    pair_cache:   Dict[Tuple[int, int], float] = {}

    def phi(neurons: Tuple[int, ...]) -> float:
        key = tuple(sorted(neurons))
        if len(key) == 1 and key[0] in single_cache:
            return single_cache[key[0]]
        if len(key) == 2 and key in pair_cache:
            return pair_cache[key]

        saved = mask_rows_merged(layer, list(key))          # zero rows
        logits = run_model_local(model, layer, score, batches)
        restore_lora_rows(saved)

        # signed contribution (no absolute!)
        val = (base_local - logits).mean().item()
        if len(key) == 1:
            single_cache[key[0]] = val
        elif len(key) == 2:
            pair_cache[key] = val
        return val
    return phi

def pair_avg_synergy(S: List[int], φ) -> float:
    """Eq. Pair(C): mean ψ(i,j) over ordered pairs."""
    pairs = [(i, j) for idx, i in enumerate(S) for j in S[idx+1:]]
    return sum(φ((i, j)) - φ((i,)) - φ((j,)) for i, j in pairs) / len(pairs)

def ratio_synergy(S: List[int], φ) -> float:
    """Eq. Ratio(C): Σ ψ(i,j) / Σ ψ(i)"""
    singles = [φ((i,)) for i in S]
    pair_sum = sum(
        φ((i, j)) - φ((i,)) - φ((j,))      # ψ(i,j)
        for idx, i in enumerate(S) for j in S[idx+1:]
    )
    return pair_sum / (sum(singles) + 1e-9)

def excess_loss(S, phi):
    phi_S = phi(S)
    best_sub = max(phi([j for j in S if j != k]) for k in S)
    return phi_S - best_sub

# ─────────────────────────────────  Main  ──────────────────────────────────
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--lora_path",  default="AnonymousForReview2/finegrained_checkpoint_experiment_rankmistral_r8_mlp_only")
    pa.add_argument("--tok_path",  default="mistralai/Mistral-7B-v0.1")
    pa.add_argument("--layer_idx", type=int, default=12)
    pa.add_argument("--coalitions_file", default="../partition.pkl")
    pa.add_argument("--n_queries", type=int, default=50)
    pa.add_argument("--n_docs",    type=int, default=10)
    pa.add_argument("--data_file", default="../dataset/top1000.dev")
    pa.add_argument("--batch_size", type=int, default=8)
    pa.add_argument("--max_length", type=int, default=256)
    pa.add_argument("--device", choices=["cpu", "cuda"], default=None)
    pa.add_argument("--mix", action="store_true")
    args = pa.parse_args()
    login_key = "Enter LLaMa access key"
    login(login_key)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ---- model ------------------------------------------------------------
    cfg  = PeftConfig.from_pretrained(args.lora_path)
    base = AutoModelForSequenceClassification.from_pretrained(
            cfg.base_model_name_or_path,
            num_labels=1,
            torch_dtype=torch.float16 if device=="cuda" else torch.float32,
            device_map=device if device=="cuda" else None,
            low_cpu_mem_usage=True)
    tok = AutoTokenizer.from_pretrained(args.tok_path)
    tok.pad_token = tok.eos_token
    base.config.pad_token_id = tok.pad_token_id
    model = PeftModel.from_pretrained(base, args.lora_path,
                                      device_map=device if device=="cuda" else None,
                                      num_labels=1).eval()
    model = model.merge_and_unload()       


    layer = model.model.layers[args.layer_idx]
    score = model.score
    # ---- data -------------------------------------------------------------
    data   = load_ms_marco_data(args.n_queries, args.n_docs, args.data_file)
    batches = tokenize_dataset(data, tok, args.batch_size, args.max_length, device)

    base_local = run_model_local(model, layer, score, batches)   # (N,)
    phi = make_phi_fn(model, layer, score, batches, base_local)

    coals = load_coalitions(pathlib.Path(args.coalitions_file))
    if args.mix:
        coals = mix_coalitions(coals)

    res = []
    for cid, S in enumerate(coals):
        if (len(S) < 7) or (len(S) > 7):
            continue
        pavg  = pair_avg_synergy(S, phi)
        #ratio = ratio_synergy(S, phi)
        res.append((cid, pavg))
        print(f"Coalition {cid:3d} | {S} | pairwise={pavg:.3f}")
   

    if res:
        mp = sum(r[1] for r in res)/len(res)
        # mr = sum(r[2] for r in res)/len(res)
        # me = sum(r[3] for r in res)/len(res)
        print("────────────────────────────────────────────────────────")
        print(f"Means   pair_avg={mp:.6f}") 
    else:
        print("No size‑8 coalitions found.", file=sys.stderr)

if __name__ == "__main__":
    main()