#!/usr/bin/env python3

import argparse
import os
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from sam2act.mvt.spatial_graph_memory import (
    SpatialEMAMemoryBank,
    batched_spatial_ema_update,
    spatial_masked_scaled_dot_product_attention,
    spatial_graph_contrastive_loss,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(0)

    B, Seq = 4, 16

    # --- Task 1: Spatial-Aware Memory Aggregation (Node Coarsening) ---
    banks = [SpatialEMAMemoryBank(eps=0.05, alpha=0.4) for _ in range(B)]

    centers = torch.tensor(
        [
            [0.10, 0.10, 0.10],
            [0.30, 0.10, 0.10],
            [0.30, 0.30, 0.10],
        ],
        device=device,
        dtype=torch.float32,
    )

    feat_dim = 32
    for t in range(Seq):
        cluster_ids = torch.randint(low=0, high=centers.shape[0], size=(B,), device=device)
        new_pos = centers[cluster_ids] + 0.01 * torch.randn((B, 3), device=device)
        new_feat = torch.randn((B, feat_dim), device=device, requires_grad=True)
        batched_spatial_ema_update(banks, new_feat, new_pos)

    bank_sizes = [len(b) for b in banks]
    print(f"[Task1] bank sizes after Seq={Seq}: {bank_sizes}")

    # --- Task 2: Sparse Adjacency Masking Attention ---
    Lq, Lk, Dk, Dv = 16, 32, 64, 128
    q = torch.randn((B, Lq, Dk), device=device, requires_grad=True)
    k = torch.randn((B, Lk, Dk), device=device, requires_grad=True)
    v = torch.randn((B, Lk, Dv), device=device, requires_grad=True)
    pos_q = centers[torch.randint(0, centers.shape[0], (B, Lq), device=device)] + 0.01 * torch.randn(
        (B, Lq, 3), device=device
    )
    pos_k = centers[torch.randint(0, centers.shape[0], (B, Lk), device=device)] + 0.01 * torch.randn(
        (B, Lk, 3), device=device
    )

    attn_out = spatial_masked_scaled_dot_product_attention(
        q,
        k,
        v,
        pos_q=pos_q,
        pos_k=pos_k,
        eps=0.06,
        dropout_p=0.0,
        training=True,
    )
    print(f"[Task2] attn_out shape={tuple(attn_out.shape)} device={attn_out.device}")
    attn_loss = attn_out.square().mean()
    attn_loss.backward()
    print(f"[Task2] loss={float(attn_loss.item()):.6f} grad(q)={q.grad is not None} grad(k)={k.grad is not None}")

    # --- Task 3: Spatial Graph Contrastive Loss ---
    D = 96
    h = torch.randn((B, Seq, D), device=device, requires_grad=True)
    pos = centers[torch.randint(0, centers.shape[0], (B, Seq), device=device)] + 0.01 * torch.randn(
        (B, Seq, 3), device=device
    )
    aux_loss = spatial_graph_contrastive_loss(h, pos, eps=0.06, tau=0.1, reduction="mean")
    aux_loss.backward()
    print(f"[Task3] aux_loss={float(aux_loss.item()):.6f} grad(h)={h.grad is not None}")


if __name__ == "__main__":
    main()
