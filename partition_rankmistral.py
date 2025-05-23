#!/usr/bin/env python
"""
partition_rankmistral.py  —  PAC‑stable partitioning of RankMistral MLP neurons
================================================================================
〉 Dataset  : MS‑MARCO dev top‑1000  (`dataset/top1000.dev` 4‑col TSV)
〉 Model    : LoRA‑fine‑tuned RankMistral seq‑classification head (1 logit)
〉 Metrics  : --metric {oca,pas}
              · OCA : φ(i,j) = (1 − |cos(W_i,W_j)|) · Cov[a_i,a_j]
              · PAS : φ(i,j) ≈ −E[(∂ℓ/∂a_i · a_i)(∂ℓ/∂a_j · a_j)]
〉 Output   : pickled list[list[int]] of ε‑PAC‑stable coalitions
--------------------------------------------------------------------------------
"""
from __future__ import annotations
import argparse, math, os, pickle
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftConfig, PeftModel
from tqdm import tqdm

# ─────────────────────────────  MS‑MARCO loader  ──────────────────────────────
def load_ms_marco_data(n_queries: int,
                       n_docs: int,
                       file_path: str = 'dataset/top1000.dev') -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    with open(file_path, 'r', encoding='utf‑8') as f:
        for line in f:
            _, _, q, doc = line.rstrip('\n').split('\t')[:4]
            if q not in out and len(out) >= n_queries:
                continue
            out.setdefault(q, []).append(doc)
            if len(out[q]) == n_docs:
                continue
    return out

# ─────────────────────────────  Model loading  ────────────────────────────────
def load_rankmistral(lora_path: str, device: torch.device):
    cfg  = PeftConfig.from_pretrained(lora_path)
    base = AutoModelForSequenceClassification.from_pretrained(
        cfg.base_model_name_or_path,
        num_labels=1,
        torch_dtype=torch.float16,
        device_map={'': device.index})
    tok  = AutoTokenizer.from_pretrained(cfg.base_model_name_or_path)
    tok.pad_token = tok.eos_token
    base.config.pad_token_id = tok.pad_token_id
    model = PeftModel.from_pretrained(base, lora_path).eval().to(device)
    model = model.merge_and_unload()          # drop LoRA adapters → single fp16 module
    return model, tok

def pair_text(query: str, doc: str) -> str:
    return f"query: {query} document: {doc}"

# ───────────────────  O C A   (weights ⊥ + activation cov)  ───────────────────
def collect_oca_phi(model, tok, data, layer, device, fraction,
                    batch=4, max_len=256, dtype=torch.float16):
    """
    φ(i,j) = (1 − |cos(W_i,W_j)|) · Cov[a_i,a_j]
    """
    assert 0. < fraction <= 1.
    mlp = model.model.layers[layer].mlp               # rank‑mistral layer
    W   = mlp.down_proj.weight.t().to(torch.float32)  # (H,d_model)
    n_keep = max(1, int(round(W.size(0)*fraction)))
    W   = W[:n_keep]

    W_unit = W / W.norm(dim=1, keepdim=True).clamp_min(1e-6)
    cos_W  = (W_unit @ W_unit.T).abs()                # (n,n)

    run_sum  = torch.zeros(n_keep,           device=device)
    run_2nd  = torch.zeros(n_keep, n_keep,   device=device)
    total    = 0
    cache: dict = {}

    def fwd_hook(_, __, hidden):                      # hidden shape (B,L,H)
        act = hidden.detach().to(torch.float32)
        if act.ndim == 3:
            act = act.view(-1, act.size(-1))
        cache['a'] = act[:, :n_keep]

    h = mlp.gate_proj.register_forward_hook(fwd_hook)
    try:
        pairs = [pair_text(q, d) for q, docs in data.items() for d in docs]
        for i in tqdm(range(0, len(pairs), batch),
                      desc=f'Layer‑{layer} activ', disable=device.index not in {None,0}):
            enc = tok(pairs[i:i+batch], return_tensors='pt',
                      padding=True, truncation=True, max_length=max_len).to(device)
            with torch.no_grad():
                model(**enc)
            a = cache.pop('a')
            run_sum += a.sum(0)
            run_2nd += a.T @ a
            total   += a.size(0)
    finally:
        h.remove()

    mu   = run_sum / total
    cov  = run_2nd / total - mu.unsqueeze(1) @ mu.unsqueeze(0)
    phi  = (1.0 - cos_W) * cov
    phi.fill_diagonal_(0)
    return phi.to(dtype=dtype)

# ─────────────────────  P A S   (fast gradient interaction)  ───────────────────
@torch.no_grad()
def _layer_local_logit(model, hidden, layer):
    """One‑layer‑ahead linear head (clone of final head)."""
    head_weight = model.score.weight.squeeze()        # (d_model,)
    head_bias   = model.score.bias.squeeze()          # scalar
    return (hidden @ head_weight) + head_bias

def collect_pas_phi(
    model,
    tok,
    data,
    layer: int,
    device,
    fraction: float,
    batch: int = 4,
    max_len: int = 256,
    dtype: torch.dtype = torch.float16,
):
    """
    Estimate the PAS matrix

        φ(i,j) ≈ -E_x[(∂ℓ/∂a_i · a_i)(∂ℓ/∂a_j · a_j)]

    using *one* gradient pass per mini‑batch and keeping memory flat.
    """

    # --------------------------- setup ----------------------------------
    assert 0.0 < fraction <= 1.0, "`fraction` must be in (0,1]"
    mlp      = model.model.layers[layer].mlp
    H_full   = mlp.gate_proj.out_features
    n_keep   = max(1, int(round(H_full * fraction)))

    vv_accum  = torch.zeros(n_keep, n_keep, device=device, dtype=torch.float32)
    token_cnt = 0

    model.eval()
    model.requires_grad_(False)            # params stay fixed

    cache: dict = {}

    # -------- hook: save activation slice that *requires grad* ----------
    def fwd_hook(_, __, out):
        """
        out: (B, seq_len, H)  or  (tokens, H)
        We keep only the slice of interest and mark it for gradients.
        """
        act = out.view(-1, H_full)[:, :n_keep]   # view, no copy
        act.retain_grad()                       # so autograd keeps ∂ℓ/∂a
        cache["act"] = act

    hook = mlp.gate_proj.register_forward_hook(fwd_hook)

    pair_texts = [f"{q} </s> {d}" for q, docs in data.items() for d in docs]

    try:
        with torch.enable_grad():               # grad mode only here
            for start in tqdm(
                range(0, len(pair_texts), batch),
                desc=f"Layer‑{layer} PAS",
                disable=device.index not in {None, 0},
            ):
                enc = tok(
                    pair_texts[start : start + batch],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_len,
                ).to(device)

                # ---- forward ----
                logits = model(**enc).logits.squeeze(-1).sum()

                # ---- gradient just w.r.t. cached slice ----
                grads = torch.autograd.grad(
                    logits,                       # scalar output
                    cache["act"],                 # (tokens, n_keep)
                    retain_graph=False,
                    create_graph=False,
                )[0]

                v = (grads * cache["act"]).float()   # Hadamard vector
                vv_accum += v.T @ v
                token_cnt += v.size(0)

                # free tensors for next iteration
                cache.clear()
                del grads, v

    finally:
        hook.remove()
        torch.set_grad_enabled(False)            # restore default

    phi = -(vv_accum / max(token_cnt, 1)).to(dtype=dtype)
    phi.fill_diagonal_(0)
    return phi

# ─────────────────────────  Coalition sampling helper  ─────────────────────────
def sample_coalitions(pool: torch.Tensor, m: int, g: torch.Generator,
                      min_k=2, max_k=6) -> List[Tuple[int,...]]:
    n = pool.numel()
    if n == 0:
        return []
    max_k = min(max_k, n)
    min_k = min(min_k, max_k)

    k_sizes = torch.randint(min_k, max_k+1, (m,), generator=g, device=pool.device)
    k_max   = int(k_sizes.max())
    perms   = torch.stack([torch.randperm(n, generator=g, device=pool.device)[:k_max]
                           for _ in range(m)])
    cand    = pool[perms]
    mask    = torch.arange(k_max, device=pool.device).expand(m, k_max) < k_sizes.unsqueeze(1)
    return [tuple(cand[i, mask[i]].tolist()) for i in range(m)]

# ─────────────────────────  F A S T   T O P ‑ C O V E R  ───────────────────────
def fast_top_cover(phi: torch.Tensor, rng: torch.Generator,
                   ω=32, m=2048, max_k=6) -> List[List[int]]:
    n, device = phi.size(0), phi.device
    R = torch.arange(n, device=device)
    reservoir = sample_coalitions(R, m, rng, 2, max_k)
    parts: List[List[int]] = []

    while R.numel():
        picks = [T for T in reservoir if set(T).issubset(R.tolist())][:ω]

        # build best‑mate mask
        M = torch.zeros(len(R), len(R), dtype=torch.bool, device=device)
        for T in picks:
            T = list(T)
            for idx, i in enumerate(T):
                j = max((phi[i,j], j) for j in T if j!=i)[1]
                pos_i, pos_j = (R==i).nonzero()[0,0], (R==j).nonzero()[0,0]
                M[pos_i,pos_j] = True
        M |= torch.eye(len(R), dtype=torch.bool, device=device)

        # BFS SCC extraction
        unseen = torch.ones(len(R), dtype=torch.bool, device=device)
        while unseen.any():
            seed = unseen.nonzero()[0,0]
            q, comp = [seed.item()], []
            unseen[seed]=False
            while q:
                u=q.pop()
                comp.append(u)
                nbrs = M[u].nonzero().squeeze(1)
                for v in nbrs:
                    if unseen[v]:
                        unseen[v]=False; q.append(v.item())
            parts.append(R[comp].tolist())
        R = torch.tensor([i for i in range(n) if all(i not in p for p in parts)],
                         device=device)
    return parts

# ────────────────────────────────  Main  ───────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
        default='AnonymousForReview2/watereddown_reranker_mistral_cqtr_mlp_only')
    parser.add_argument('--ms_file', default='dataset/top1000.dev')
    parser.add_argument('--n_queries', type=int, default=50)
    parser.add_argument('--n_docs',    type=int, default=100)
    parser.add_argument('--layer',     type=int, default=12)
    parser.add_argument('--fraction',  type=float, default=1.0)
    parser.add_argument('--metric',    choices=['oca','pas'], default='oca')
    parser.add_argument('--m', type=int, default=50000)
    parser.add_argument('--ω', type=int, default=2000)
    parser.add_argument('--max_k', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_len',    type=int, default=256)
    parser.add_argument('--output',     default='partition.pkl')
    args = parser.parse_args()

    if not (0. < args.fraction <= 1.):
        raise ValueError('--fraction ∉ (0,1]')

    rank  = int(os.getenv('RANK', 0))
    world = int(os.getenv('WORLD_SIZE', 1))
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

    if world > 1 and not dist.is_initialized():
        dist.init_process_group(backend='nccl', rank=rank, world_size=world)

    model, tok = load_rankmistral(args.model, device)
    data       = load_ms_marco_data(args.n_queries, args.n_docs, args.ms_file)

    if world > 1:                                  # simple query‑level sharding
        qs   = list(data.items())
        data = dict(qs[rank::world])

    if args.metric == 'oca':
        phi = collect_oca_phi(model, tok, data,
                              layer=args.layer, device=device,
                              fraction=args.fraction, batch=args.batch_size,
                              max_len=args.max_len)
    else:
        torch.set_grad_enabled(True)
        phi = collect_pas_phi(model, tok, data,
                              layer=args.layer, device=device,
                              fraction=args.fraction, batch=args.batch_size,
                              max_len=args.max_len)
        torch.set_grad_enabled(False)

    rng = torch.Generator(device=phi.device).manual_seed(42+rank)
    partition = fast_top_cover(phi, rng, ω=args.ω, m=args.m, max_k=args.max_k)

    if rank == 0:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'wb') as f:
            pickle.dump(partition, f)
        print(f'✓ Saved {len(partition)} coalitions → {args.output}')

    if world > 1:
        dist.barrier(); dist.destroy_process_group()

if __name__ == '__main__':
    main()