# bipartite.py

from __future__ import annotations
import torch, numpy as np
from typing import List, Tuple, Dict, Any
from scipy.optimize import linear_sum_assignment

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftConfig, PeftModel
from tqdm import tqdm
import os
import pickle
import plotly.graph_objects as go
from claude_sankey import sankey_coalitions, sankey_neurons
# ─────────────────────────────────────────────────────────────
# 0. Utils
# ─────────────────────────────────────────────────────────────

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
    model = model.merge_and_unload()
    model.eval()
    return model, tok

def read_pkl(file_path: str):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    data = [lst for lst in data if len(lst) > 10]
    data = [sorted(lst) for lst in data]
    data = sorted(data, key=len, reverse=True)

    # Print the coalitions
    # for i, coalition in enumerate(data, 1):
    #     print(f"Coalition {i}: {coalition}")
    return data

# ─────────────────────────────────────────────────────────────
# 0‑bis.  Helper to get the 14 336 → 14 336 transition matrix
# ─────────────────────────────────────────────────────────────
def get_intermediate_to_intermediate_W(model, layer_id: int,  # ℓ  → ℓ+1
                                       dtype=torch.float32,
                                       device: torch.device | None = None
                                      ) -> torch.Tensor:
    """
    Returns W = gate_{ℓ+1} · down_{ℓ},  shape (14336, 14336)

    ‑  down_{ℓ}.weight : (4096 , 14336)
    ‑  gate_{ℓ+1}.weight: (14336, 4096)

    Their product gives the direct linear influence of
    *layer‑ℓ* intermediate neurons on *layer‑(ℓ+1)* intermediates.
    """
    layer_L   = model.model.layers[layer_id]          # ℓ
    layer_Lp1 = model.model.layers[layer_id + 1]      # ℓ+1

    down = layer_L.mlp.down_proj.weight      # (4096 , 14336)
    gate = layer_Lp1.mlp.gate_proj.weight    # (14336, 4096)

    W = gate @ down                          # (14336, 14336)

    if device is not None:
        W = W.to(device, dtype=dtype)
    else:
        W = W.to(dtype=dtype)

    return W

# ─────────────────────────────────────────────────────────────
# 1. Coalition‑level influence matrix
# ─────────────────────────────────────────────────────────────

def coalition_influence_matrix(
    W: torch.Tensor,              # (n_next, n_prev)  weight matrix (float32)
    A: torch.Tensor,              # (n_prev,)         mean activation in layer ℓ
    C_prev: List[List[int]],      # coalitions in layer ℓ   (≤ 100)
    C_next: List[List[int]],      # coalitions in layer ℓ+1 (≤ 100)
    normalise: bool = True,
    cpu_threshold: int = 5000,    # move to CPU if n_prev > threshold
) -> np.ndarray:
    """
    Returns S[i,j] = Σ_{p∈C_prev[i]} Σ_{q∈C_next[j]} |W[q,p]| * A[p]

    If `normalise` is True, divides by |C_prev[i]|·|C_next[j]|.
    """
     # ‑‑ Ensure A is a tensor living on the same device as W
    if isinstance(A, np.ndarray):
        A = torch.from_numpy(A)
    A = A.to(W.device).float()


    n_prev = W.size(1)

    # Optional CPU fallback for very wide layers
    if W.is_cuda and n_prev > cpu_threshold:
        W = W.cpu()
        A = A.cpu()
        print(f"[INFO] Offloading influence calc to CPU for {n_prev}×{n_prev} matrix")
        
        
    impact = torch.abs(W) * A.view(1, -1)          # (n_next, n_prev)

    S = torch.zeros(len(C_prev), len(C_next), dtype=torch.float32)
    for i, S_src in enumerate(C_prev):
        cols = torch.tensor(S_src, dtype=torch.long, device=impact.device)
        col_sum = impact.index_select(1, cols).sum(dim=1)        # (n_next,)

        for j, S_tgt in enumerate(C_next):
            rows = torch.tensor(S_tgt, dtype=torch.long, device=impact.device)
            val = col_sum.index_select(0, rows).sum()
            if normalise:
                val = val / (len(S_src) * len(S_tgt) + 1e-6)
            S[i, j] = val

    return S.cpu().numpy()

# ─────────────────────────────────────────────────────────────
# 2. Maximum‑weight matching
# ─────────────────────────────────────────────────────────────

def match_coalitions(
    S: np.ndarray,
    min_score: float = None
) -> List[Tuple[int, int, float]]:
    """
    Hungarian assignment on −S (maximisation).
    Returns list of (i_prev, j_next, score).
    If min_score is set, filters edges below that value.
    """
    row_ind, col_ind = linear_sum_assignment(-S)
    matches = []
    for i, j in zip(row_ind, col_ind):
        if min_score is None or S[i, j] >= min_score:
            matches.append((i, j, float(S[i, j])))
    return matches

# ─────────────────────────────────────────────────────────────
# 3. Event classification
# ─────────────────────────────────────────────────────────────

def classify_events(
    C_prev: List[List[int]],
    C_next: List[List[int]],
    S: np.ndarray,
    tau: float = 0.30,
) -> Dict[str, Any]:
    """
    Classify how coalitions evolve from layer ℓ (C_prev) to layer ℓ+1 (C_next).

    ─────────────
    Terminology
    ─────────────
    • persist  — strong 1‑to‑1 continuation (primary influence in *both* directions)
    • split    — one parent distributes significant mass (≥τ·row‑max) to ≥2 children
    • merge    — ≥2 parents contribute significant mass (≥τ·col‑max) to one child
    • vanish   — parent coalition has no significant child
    • emerge   — child coalition has no significant parent
    """
    K_prev, K_next = len(C_prev), len(C_next)

    best_row = S.max(axis=1)          # shape (K_prev,)
    best_col = S.max(axis=0)          # shape (K_next,)

    used_prev: set[int] = set()
    used_next: set[int] = set()

    events: Dict[str, Any] = dict(
        persist=[],
        split=[],
        merge=[],
        vanish=[],
        emerge=[],
    )

    # ── 1. Persist (one‑to‑one, strongest both ways) ───────────────────────
    for i in range(K_prev):
        j = int(S[i].argmax())
        if (
            S[i, j] >= tau * best_row[i]      # strong for this parent
            and S[i, j] == best_col[j]        # also strongest for that child
        ):
            events["persist"].append(
                {"prev": i, "next": j, "score": float(S[i, j])}
            )
            used_prev.add(i)
            used_next.add(j)

    # ── 2. Splits (one parent → many children) ─────────────────────────────
    for i in range(K_prev):
        if i in used_prev:
            continue
        # children whose share exceeds threshold relative to this parent
        children = [
            j for j in range(K_next)
            if S[i, j] >= tau * best_row[i]
        ]
        if len(children) >= 2:
            events["split"].append({"prev": i, "children": children})
            used_prev.add(i)
            used_next.update(children)

    # ── 3. Merges (many parents → one child) ───────────────────────────────
    for j in range(K_next):
        if j in used_next:
            continue
        # parents whose share exceeds threshold relative to this child
        parents = [
            i for i in range(K_prev)
            if S[i, j] >= tau * best_col[j] and i not in used_prev
        ]
        if len(parents) >= 2:
            events["merge"].append({"next": j, "parents": parents})
            used_next.add(j)
            used_prev.update(parents)

    # ── 4. Vanish / Emerge (unmatched coalitions) ──────────────────────────
    for i in range(K_prev):
        if i not in used_prev:
            events["vanish"].append(i)
    for j in range(K_next):
        if j not in used_next:
            events["emerge"].append(j)

    return events

# ─────────────────────────────────────────────────────────────
# 4. Example driver
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Dummy shapes for demo
    n_prev = n_next = 14336
    d_model = 4096
    layer_id=8
    n_queries = 50
    n_docs = 100
    torch.manual_seed(0)
    activations = np.zeros((n_queries*n_docs, d_model), dtype=float)
    model, tok = load_rankmistral('/gypsum/work1/allan/anijasure/multi_stage_retrieval/model_outputs/finegrained_checkpoint_experiment_llama3_r8_lora_mlp', torch.device(f'cuda:{int(os.getenv('RANK', 0))}'))
      
    W = model.model.layers[layer_id].mlp.down_proj.weight.detach().T.float()
    #print(activations.shape)
    d_int = model.model.layers[0].mlp.gate_proj.weight.size(0)   # 14 336
    activations_int = np.zeros((n_queries * n_docs, d_int), dtype=np.float32)

    for i in range(n_queries):
        for j in range(n_docs):
            path = f"activations/llama_relevance/q{i}/d{j}/layer_{layer_id}_intermediate.pt"
            if not os.path.exists(path):
                print(f"File not found: {path}")
                break
            act = torch.load(path, map_location="cpu").float()   # (14 336,)
            activations_int[i * n_docs + j] = act.numpy()
            #print(activations_int[i * n_docs + j].max())
    # Mean activation of every neuron in layer ℓ (14 336,)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = torch.from_numpy(activations_int.mean(axis=0)).to(device)

    # ---------- 14 336 → 14 336 weight ----------
    W = get_intermediate_to_intermediate_W(model, layer_id, dtype=torch.float32, device=device)

    print("W shape :", tuple(W.shape))           # (14336, 14336)
    print("A shape :", tuple(A.shape))           # (14336,)
    print("A max :", A.max().item())          # 0.0001
    print("W max :", W.max().item())            # 0.0001

    # ----------  Load coalitions ----------
    C_prev = read_pkl("../partitions/covariate/llama_8_relevance.pkl")   # indices in [0,14 336)
    C_next = read_pkl("../partitions/covariate/llama_9_relevance.pkl")   # indices in [0,14 336)
    # print(C_prev)
    # print(C_next)
    # ----------  Influence matrix ----------
    S = coalition_influence_matrix(W, A, C_prev, C_next,
                                   normalise=True, cpu_threshold=20000)

    print("S shape :", tuple(S.shape))           # (14336,)

    # ----------  Matching ----------
    matches = match_coalitions(S, min_score=0.01)
    # print("Top‑5 matches:", matches[:5])

    # ----------  Events ----------
    events = classify_events(C_prev, C_next, S, tau=0.30)
    # for k, v in events.items():
    #     print(f"{k:<9} → {v[:3]} … ({len(v)} total)")
      
    # ----- 5. Quick percentages --------------------------------------------------
    tot_prev = len(C_prev)            # coalitions that existed in layer ℓ
    tot_next = len(C_next)            # coalitions that exist in layer ℓ+1

    pct_emerge   = 100.0 * len(events["emerge"])     / tot_next if tot_next else 0.0
    pct_merge    = 100.0 * len(events["merge"])      / tot_next if tot_next else 0.0
    pct_split    = 100.0 * len(events["split"])      / tot_prev if tot_prev else 0.0
    pct_vanish   = 100.0 * len(events["vanish"])  / tot_prev if tot_prev else 0.0
    pct_persist  = 100.0 * len(events["persist"])    / tot_prev if tot_prev else 0.0

    print(f"{layer_id}→{layer_id+1} | "
        f"Emerge: {pct_emerge:.1f}%   "
        f"Merge: {pct_merge:.1f}%   "
        f"Split: {pct_split:.1f}%   "
        f"Vanish: {pct_vanish:.1f}%"
        f"Persist: {pct_persist:.1f}%")       
    
    
    #Visualisation
    filtered_dict = {key: events[key] for key in {'persist','split'} if key in events}

    sankey_coalitions(C_prev, C_next, S, filtered_dict,
                        thr=0.3,                       #  show stronger links only
                        html="coalitions_sankey_claude_0.3_merge_split_persist_layer8.html")
    
    sankey_neurons(W, A,
                   top_k=2,                          #  2 strongest targets / neuron
                   thr=0.2,                        #  drop very weak links
                   html="neurons_sankey_claude.html")