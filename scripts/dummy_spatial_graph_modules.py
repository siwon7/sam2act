import torch

from sam2act.utils.spatial_graph_memory import (
    SpatialGraphMemoryBank,
    build_summary_attention_bias,
    spatial_graph_contrastive_loss,
)
from sam2act.mvt.sam2_train.modeling.sam.transformer import Attention


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bsz, seq, dim = 4, 16, 128
    tokens_per_node = 32

    bank = SpatialGraphMemoryBank(
        distance_threshold=0.08,
        ema_alpha=0.5,
        grip_threshold=0.35,
        feature_similarity_threshold=-1.0,
    )
    memory_bank = []
    transition_counts = {}
    prev_idx = None
    for idx in range(seq):
        feat = torch.randn(tokens_per_node, 1, dim, device=device, requires_grad=True)
        pos_enc = torch.randn(tokens_per_node, 1, dim, device=device, requires_grad=True)
        pos3d = torch.randn(1, 3, device=device)
        grip = torch.rand(1, 1, device=device)
        memory_bank, _, node_idx = bank.update(memory_bank, feat, pos_enc, pos3d, grip, idx)
        if prev_idx is not None:
            transition_counts[(prev_idx, node_idx)] = transition_counts.get((prev_idx, node_idx), 0) + 1
        prev_idx = node_idx

    query_pos = torch.randn(1, 3, device=device)
    bias = build_summary_attention_bias(
        summary_nodes=memory_bank[: min(len(memory_bank), 8)],
        prev_summary_idx=prev_idx,
        transition_counts=transition_counts,
        current_grip_state=torch.rand(1, 1, device=device),
        query_pos=query_pos,
        distance_threshold=0.08,
        transition_bias_scale=1.5,
        grip_bias_scale=0.5,
        spatial_bias_scale=0.25,
        device=device,
        dtype=torch.float32,
    )

    q = torch.randn(bsz, seq, dim, device=device, requires_grad=True)
    k = torch.randn(bsz, seq * 2, dim, device=device, requires_grad=True)
    v = torch.randn(bsz, seq * 2, dim, device=device, requires_grad=True)
    attn = Attention(embedding_dim=dim, num_heads=8).to(device)
    attn_mask = None
    if bias is not None:
        attn_mask = bias.expand(bsz, 1, seq, bias.shape[-1])
        if attn_mask.shape[-1] != k.shape[1]:
            attn_mask = torch.zeros(bsz, 1, seq, k.shape[1], device=device)
    out = attn(q, k, v, attn_mask=attn_mask)

    hidden = torch.randn(bsz, seq, dim, device=device, requires_grad=True)
    pos = torch.randn(bsz, seq, 3, device=device)
    loss_ctr, acc = spatial_graph_contrastive_loss(
        hidden_states=hidden,
        positions=pos,
        temperature=0.1,
        distance_threshold=0.5,
    )

    total = out.mean() + loss_ctr
    total.backward()
    print(
        {
            "device": str(device),
            "num_nodes": len(memory_bank),
            "attn_out": tuple(out.shape),
            "contrastive_loss": float(loss_ctr.item()),
            "contrastive_top3_acc": float(acc.item()),
        }
    )


if __name__ == "__main__":
    main()
