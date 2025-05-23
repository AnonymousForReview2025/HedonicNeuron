#!/usr/bin/env python
"""
baseline_clustering_rankmistral.py  —  Weight‑Space K‑Means baseline
-------------------------------------------------------------------
Simple non‑game‑theoretic baseline that clusters MLP gate projection
weights with spherical k‑means.
*   Uses **the same model, data loader, and CLI defaults** as
    `partition_rankmistral.py` so it can be swapped into existing
    pipelines.
*   No φ calculation, no PAC‑Top‑Cover – just weight‑space similarity.
*   All heavy tensors stay on‑GPU; multi‑GPU setups run only on rank 0.
*   Outputs `baseline_clusters.pkl` → list[list[int]] of neuron ids.
"""
from __future__ import annotations
import argparse, math, os, pickle
from pathlib import Path
from typing import List, Tuple

import torch, torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftConfig, PeftModel

# ─────────────────────────  Distributed helper  ────────────────────────────

def init_dist() -> Tuple[int, int]:
    world = int(os.getenv("WORLD_SIZE", "1"))
    if world > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank  = dist.get_rank()  if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    return rank, world

# ─────────────────────────────  K‑Means  ───────────────────────────────────

def spherical_kmeans(x: torch.Tensor, k: int, iters: int = 25, seed: int = 42) -> torch.Tensor:
    """Return cluster assignments for rows of *x* using cosine k‑means."""
    g = torch.Generator().manual_seed(seed)    
    n, _ = x.shape
    # k random distinct initial centres
    centres = x[torch.randperm(n, generator=g)[:k]]
    for _ in range(iters):
        # cosine distance = 1‑cos sim  (x and centres are ℓ2‑normalised)
        sim   = x @ centres.T                                # (n,k)
        assign = sim.argmax(dim=1)                           # (n,)
        for j in range(k):
            mask = assign == j
            if mask.any():
                centres[j] = torch.nn.functional.normalize(x[mask].mean(dim=0, keepdim=True), dim=1)
    return assign

# ───────────────────────────────  Main  ────────────────────────────────────

def main() -> None:
    arg = argparse.ArgumentParser()
    # Same CLI defaults as partition_rankmistral.py
    arg.add_argument("--lora_path", default="AnonymousForReview2/finegrained_checkpoint_experiment_rankmistral_r8_mlp_only")
    arg.add_argument("--layer",     type=int, default=12)
    arg.add_argument("--fraction",  type=float, default=1.0, help="0<φ≤1 rows kept")
    arg.add_argument("--k",         type=int, default=None, help="Number of clusters (default=⌈√K⌉)")
    arg.add_argument("--iters",     type=int, default=25,  help="K‑means iterations")
    arg.add_argument("--seed",      type=int, default=42)
    arg.add_argument("--output",    default="baseline_clusters.pkl")
    cfg = arg.parse_args()

    if not (0. < cfg.fraction <= 1.):
        raise ValueError("--fraction must be in (0,1]")

    rank, world = init_dist()
    dev = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # Only rank‑0 performs clustering; spare ranks exit early
    if rank != 0:
        if dist.is_initialized():
            dist.barrier(); dist.destroy_process_group()
        return

    # ───────────────  Load model (weights only)  ────────────────
    cfg_peft = PeftConfig.from_pretrained(cfg.lora_path)
    base = AutoModelForSequenceClassification.from_pretrained(
        cfg_peft.base_model_name_or_path,
        num_labels=1,
        torch_dtype=torch.float16,
        device_map={"": dev.index})
    model = PeftModel.from_pretrained(base, cfg.lora_path).eval().to(dev).merge_and_unload()

    # ───────────────  Select neurons  ────────────────
    gate_W = model.model.layers[cfg.layer].mlp.gate_proj.weight      # (d_ff, d_model)
    full   = gate_W.size(0)
    keep   = max(1, int(round(full * cfg.fraction)))
    W      = gate_W[:keep].to(torch.float32).to(dev)                # (K,d_model)
    W      = torch.nn.functional.normalize(W, dim=1)                # ℓ2 normalise rows

    # ───────────────  K‑means  ────────────────
    k = cfg.k or int(math.ceil(math.sqrt(keep)))
    assign = spherical_kmeans(W, k=k, iters=cfg.iters, seed=cfg.seed)   # (K,)

    # ───────────────  Build clusters  ────────────────
    parts: List[List[int]] = []
    for c in range(k):
        idx = torch.nonzero(assign == c, as_tuple=False).squeeze(1)
        if idx.numel():
            parts.append(idx.tolist())

    # Largest‑first for readability
    parts.sort(key=len, reverse=True)

    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(parts, open(cfg.output, "wb"))

    print(f"✓ {len(parts)} clusters saved  →  {cfg.output}")
    sizes = [len(p) for p in parts]
    print(f"  Cluster sizes: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)/len(sizes):.2f}")
    print(f"  First 10 sizes: {sizes[:10]}…")

    if dist.is_initialized():
        dist.barrier(); dist.destroy_process_group()

if __name__ == "__main__":
    main()
