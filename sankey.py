# Demo script to test the fixed Sankey visualization

import numpy as np
import torch
import plotly.graph_objects as go
from typing import List, Dict, Any

def sankey_coalitions(C_prev: List[List[int]], C_next: List[List[int]], S: np.ndarray, 
                     events: Dict[str, Any] | None = None, thr: float = 0.01, 
                     html: str = "coalitions_sankey.html", return_fig: bool = True):
    """
    Create a Sankey diagram showing coalition relationships.
    """
    # The edge colour encodes:
    # • green   = persist / strengthen
    # • orange  = split child
    # • purple  = merge parent
    # • grey    = other active connection
    K_prev, K_next = len(C_prev), len(C_next)
    labels = [f"L{i} ({len(C_prev[i])})" for i in range(K_prev)] + \
            [f"R{j} ({len(C_next[j])})" for j in range(K_next)]

    src, tgt, val, col = [], [], [], []

    max_S = S.max()
    
    thr_val = thr * max_S

    def colour(i, j):
        if not events:
            return "rgba(160,160,160,0.5)"
        # persist?
        if "persist" in events:
            for e in events["persist"]:
                if e["prev"] == i and e["next"] == j:
                    return "rgba(0,180,0,0.7)"
        # split child?
        if "split" in events:
            for e in events["split"]:
                if e["prev"] == i and j in e["children"]:
                    return "rgba(255,140,0,0.7)"
        # merge parent?
        if "merge" in events:
            for e in events["merge"]:
                if e["next"] == j and i in e["parents"]:
                    return "rgba(150,0,200,0.7)"
        return "rgba(160,160,160,0.5)"

    for i in range(K_prev):
        for j in range(K_next):
            w = float(S[i, j])
            if w < thr_val:
                continue
            src.append(i)
            tgt.append(K_prev + j)      # right-side offset
            val.append(w)
            col.append(colour(i, j))
    
    # Ensure we have data to display
    if not src:
        print("Warning: No links meet the threshold criteria. Try lowering the threshold.")
        # Add at least one minimal connection to prevent empty diagram
        if K_prev > 0 and K_next > 0:
            src.append(0)
            tgt.append(K_prev)
            val.append(float(S[0, 0]))
            col.append("rgba(160,160,160,0.5)")

    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            label=labels, 
            pad=6, 
            thickness=8,
            color=["#4c78a8"]*K_prev + ["#f58518"]*K_next
        ),
        link=dict(
            source=src, 
            target=tgt, 
            value=val, 
            color=col
        )
    )])
    
    fig.update_layout(
        title_text="Coalition flow ℓ → ℓ+1", 
        font_size=11,
        width=900,  # Specify width explicitly
        height=700  # Specify height explicitly
    )
    
    # Make sure we write the complete HTML file with proper headers
    fig.write_html(
        html,
        include_plotlyjs=True,
        full_html=True
    )
    
    if return_fig:
        return fig

def sankey_neurons(W: torch.Tensor, A: torch.Tensor | None = None, top_k: int = 3,
                  thr: float = 0.02, html: str = "neurons_sankey.html", return_fig: bool = True):
    """
    Create a Sankey diagram showing neuron-level connections.
    """
    with torch.no_grad():
        impact = torch.abs(W)
        if A is not None:
            impact = impact * A.view(1, -1)

        impact = impact.cpu().numpy()

    N = impact.shape[0]                 # typically 14336
    max_link = impact.max()
    thr_val = thr * max_link

    src, tgt, val = [], [], []
    for p in range(N):
        # indices of the k strongest outgoing edges
        q_idx = np.argpartition(impact[:, p], -top_k)[-top_k:]
        for q in q_idx:
            w = impact[q, p]
            if w < thr_val:
                continue
            src.append(p)
            tgt.append(N + q)           # right side offset
            val.append(float(w))
    
    # Ensure we have data to display
    if not src:
        print("Warning: No links meet the threshold criteria. Try lowering the threshold.")
        # Add at least one minimal connection to prevent empty diagram
        if N > 0:
            src.append(0)
            tgt.append(N)
            val.append(float(impact[0, 0]))

    # Create labels
    if N > 100:
        labels = [f"L{n}" for n in range(N)] + [f"R{n}" for n in range(N)]
    else:
        labels = [f"L{n}" for n in range(N)] + [f"R{n}" for n in range(N)]

    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            label=labels, 
            pad=2, 
            thickness=6,
            color=["#6baed6"]*N + ["#fd8d3c"]*N
        ),
        link=dict(
            source=src, 
            target=tgt, 
            value=val,
            color="rgba(120,120,120,0.4)"
        )
    )])
    
    fig.update_layout(
        title_text="Neuron-level flow ℓ → ℓ+1 (top-k edges)",
        font_size=9,
        width=900,  # Specify width
        height=700  # Specify height
    )
    
    # Make sure we write the complete HTML file with proper headers
    fig.write_html(
        html,
        include_plotlyjs=True,
        full_html=True
    )
    
    if return_fig:
        return fig

