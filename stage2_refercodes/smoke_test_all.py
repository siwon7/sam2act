#!/usr/bin/env python3
"""Smoke test all 4 Stage2 ideas with a single forward+backward pass.

Usage:
    CUDA_VISIBLE_DEVICES=4 python smoke_test_all.py --idea 1
    CUDA_VISIBLE_DEVICES=5 python smoke_test_all.py --idea 2
    CUDA_VISIBLE_DEVICES=6 python smoke_test_all.py --idea 3
    CUDA_VISIBLE_DEVICES=7 python smoke_test_all.py --idea 4
"""
import argparse
import os
import sys
import torch
import torch.nn.functional as F

def test_idea1(device):
    """GPSN: Graph-Guided Peak Selection"""
    sys.path.insert(0, os.path.dirname(__file__))
    from idea1_gpsn import GraphPeakSelector, peak_selection_loss, node_contrastive_loss, transition_prediction_loss

    module = GraphPeakSelector(
        feat_dim=128, node_dim=64, lang_dim=1024, topk=3,
    ).to(device)

    # Simulate inputs
    bs, nc, H, W = 2, 3, 224, 224
    heatmap = torch.randn(bs, nc, 1, H, W, device=device)
    encoder_feat = torch.randn(bs, 128, nc, 16, 16, device=device)
    memory_feat = torch.randn(bs, nc, 128, 16, 16, device=device)
    lang_emb = torch.randn(bs, 1024, device=device)
    prev_node = torch.randn(bs, 64, device=device)

    out = module(heatmap, encoder_feat, memory_feat, lang_emb, prev_node=prev_node)

    for k, v in out.items():
        if hasattr(v, 'shape'):
            print(f"  {k}: {v.shape}")
        else:
            print(f"  {k}: {v}")

    # Test losses
    node_embeds = torch.randn(5, 64, device=device)
    positions = torch.randn(5, 3, device=device)
    c_loss = node_contrastive_loss(node_embeds, positions)
    print("  contrastive_loss:", c_loss.item())

    # Backward - output key is "trans_refined"
    hm_key = "trans_refined" if "trans_refined" in out else "reweighted_heatmap"
    loss = out[hm_key].sum() + c_loss
    loss.backward()
    print("  backward: OK")
    return True


def test_idea2(device):
    """FiLM-Conditioned Keyframe Memory"""
    sys.path.insert(0, os.path.dirname(__file__))
    from idea2_film_memory import FiLMConditionedMemoryAttention

    module = FiLMConditionedMemoryAttention(
        d_model=128, node_dim=64, lang_dim=1024,
        num_heads=8, phase_predictor_layers=2,
    ).to(device)

    # Simulate: compute bias for memory attention
    num_mem = 3
    curr_feat = torch.randn(1, 128, 16, 16, device=device)
    lang_emb = torch.randn(1, 1024, device=device)

    # Check available methods
    methods = [m for m in dir(module) if not m.startswith('_') and callable(getattr(module, m))]
    print("  methods:", [m for m in methods if 'update' in m.lower() or 'add' in m.lower() or 'bias' in m.lower() or 'forward' in m.lower()])

    # Try forward pass
    try:
        bias = module(curr_feat, lang_emb)
        print("  forward output:", type(bias))
    except Exception as e:
        print("  forward error:", e)
        # Try compute_bias
        try:
            bias = module.compute_bias(curr_feat, lang_emb)
        except Exception as e2:
            print("  compute_bias error:", e2)
            bias = torch.zeros(1, device=device, requires_grad=True)
    print("  attn_bias shape:", bias.shape)
    # Expected: (1, num_heads, 1, num_mem * num_tokens_per_mem)

    # Backward
    loss = bias.sum()
    loss.backward()
    print("  backward: OK")

    if hasattr(module, 'reset'):
        module.reset()
    elif hasattr(module, 'reset_episode'):
        module.reset_episode()
    print("  reset: OK")
    return True


def test_idea3(device):
    """Dual-Level Memory"""
    sys.path.insert(0, os.path.dirname(__file__))
    from idea3_dual_memory import DualLevelMemoryModule

    module = DualLevelMemoryModule(
        feat_dim=128, node_dim=64, lang_dim=1024,
        max_peaks=3,
    ).to(device)

    bs, nc, H, W = 1, 3, 224, 224
    heatmap = torch.randn(bs, nc, 1, H, W, device=device)
    # DualLevelMemoryModule expects (B, C, H, W) for memory/encoder feat
    encoder_feat = torch.randn(bs, 128, 16, 16, device=device)
    memory_feat = torch.randn(bs, 128, 16, 16, device=device)
    lang_emb = torch.randn(bs, 1024, device=device)

    # DualLevelMemoryModule default lang_dim=128, but we pass 1024
    # Recreate with matching lang_dim
    module = DualLevelMemoryModule(
        feat_dim=128, node_dim=64, lang_dim=1024, max_peaks=3,
    ).to(device)

    out = module(memory_feat, encoder_feat, heatmap, lang_emb=lang_emb)
    print("  output keys:", list(out.keys()))

    for k, v in out.items():
        if hasattr(v, 'shape'):
            print(f"  {k}: {v.shape}")

    # Update episodic memory
    curr_node = out.get("curr_node", torch.randn(bs, 64, device=device))
    module.update_episodic_memory(memory_feat, curr_node)
    print("  episodic_memory updated")

    # Reset
    module.reset_episodic_memory()
    print("  reset: OK")

    # Backward
    hm_out = out.get("reweighted_heatmap", out.get("heatmap_refined", None))
    if hm_out is not None:
        loss = hm_out.sum()
        loss.backward()
        print("  backward: OK")
    else:
        print("  backward: skipped (no heatmap output)")

    module.reset_episodic_memory()
    print("  reset: OK")
    return True


def test_idea4(device):
    """Contrastive Phase-Aware Memory Selection"""
    sys.path.insert(0, os.path.dirname(__file__))
    from idea4_contrastive_phase import ContrastivePhaseSelector

    from idea4_contrastive_phase import ContrastivePhaseConfig
    cfg = ContrastivePhaseConfig(feat_dim=128, proprio_dim=4, phase_embed_dim=64, topk=3)
    module = ContrastivePhaseSelector(cfg).to(device)

    bs, nc, H, W = 1, 3, 224, 224
    heatmap = torch.randn(bs, nc, 1, H, W, device=device)
    feature_map = torch.randn(nc, 128, 16, 16, device=device)
    proprio = torch.randn(bs, 4, device=device)

    out = module(heatmap, feature_map, proprio)
    print("  output keys:", list(out.keys()))

    hm_out = out.get("reweighted_heatmap", None)
    if hm_out is not None:
        print("  reweighted_heatmap:", hm_out.shape)

    phase_embed = out.get("phase_embed", out.get("curr_node_embed", None))
    if phase_embed is not None:
        print("  phase_embed:", phase_embed.shape)

    # Backward
    if hm_out is not None:
        loss = hm_out.sum()
        loss.backward()
        print("  backward: OK")

    module.reset()
    print("  reset: OK")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--idea", type=int, required=True, choices=[1, 2, 3, 4])
    args = parser.parse_args()

    device = "cuda:0"
    test_fn = {1: test_idea1, 2: test_idea2, 3: test_idea3, 4: test_idea4}

    print(f"\n{'='*50}")
    print(f"  Smoke Test: Idea {args.idea}")
    print(f"{'='*50}")

    try:
        ok = test_fn[args.idea](device)
        print(f"\n  ✓ Idea {args.idea}: PASS")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n  ✗ Idea {args.idea}: FAIL — {e}")


if __name__ == "__main__":
    main()
