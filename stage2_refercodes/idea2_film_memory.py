"""
Idea 2: FiLM-Conditioned Keyframe Memory

Core insight: Instead of generic SAM2 memory cross-attention, use FiLM
(Feature-wise Linear Modulation) to dynamically modulate WHICH memory entries
are attended, conditioned on:
  1. Language instruction (task-specific continuous embedding)
  2. Predicted phase (from transition history of node embeddings)

Key difference from KC-VLA:
  - KC-VLA uses FiLM for binary keyframe detection (discrete task_id)
  - We use FiLM for continuous memory attention modulation (continuous lang embedding)
  - KC-VLA: task_id -> FiLM -> modulate phase_emb -> cross-attn with visual
  - Ours:   lang_emb + phase_pred -> FiLM -> additive bias on memory QK^T scores

Integration point: SAM2's MemoryAttentionLayer._forward_ca(), where
cross-attention between current frame and memory bank occurs. We inject
a FiLM-generated attention bias into the attn_mask argument of
F.scaled_dot_product_attention.

Architecture:
  ┌──────────────┐     ┌───────────────┐
  │ lang_emb     │     │ node_history  │
  │ [B, D_lang]  │     │ [B, T, D_node]│
  └──────┬───────┘     └──────┬────────┘
         │                    │
         │              PhasePredictor
         │                    │
         │              ┌─────▼──────┐
         │              │ phase_emb  │
         │              │ [B, D_node]│
         │              └─────┬──────┘
         │                    │
         └────────┬───────────┘
                  │
           FiLMBiasGenerator
                  │
           ┌──────▼──────┐
           │ attn_bias   │
           │ [B, H, 1, M]│  (H=num_heads, M=num_memory_tokens)
           └──────┬──────┘
                  │
                  ▼
    score = QK^T/sqrt(d) + attn_bias   <-- injected into RoPEAttention
                  │
                  ▼
           softmax -> weighted V -> memory-conditioned output
"""

import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# =============================================================================
# 1. Phase Predictor: node embedding history -> current phase estimate
# =============================================================================

class PhasePredictor(nn.Module):
    """
    Predict the current task phase from the sequence of node embeddings
    accumulated during episode execution.

    Unlike KC-VLA's discrete phase counter (monotonic, hardcoded max),
    this predicts a continuous phase embedding that can represent
    non-monotonic transitions (e.g., revisit ambiguity in put_block_back).

    Input:  node_history [B, T_hist, D_node]  (variable length, padded)
            history_mask [B, T_hist]           (True = valid)
    Output: phase_emb    [B, D_node]
    """

    def __init__(
        self,
        node_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        max_history: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.max_history = max_history

        # Learnable temporal position embeddings for history steps
        self.temporal_pos = nn.Parameter(
            torch.randn(1, max_history, node_dim) * 0.02
        )

        # Transformer encoder to aggregate history
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=node_dim,
            nhead=num_heads,
            dim_feedforward=node_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            enable_nested_tensor=False,
        )

        # Learnable query to read out phase embedding
        self.phase_query = nn.Parameter(torch.randn(1, 1, node_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=node_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(node_dim)

        # Fallback for empty history (first step)
        self.empty_phase = nn.Parameter(torch.randn(node_dim) * 0.02)

    def forward(
        self,
        node_history: Tensor,           # [B, T, D_node]
        history_mask: Optional[Tensor],  # [B, T] bool, True=valid
    ) -> Tensor:
        """Returns phase_emb [B, D_node]."""
        B, T, D = node_history.shape

        if T == 0:
            # First step: no history yet
            return self.empty_phase.unsqueeze(0).expand(B, -1)

        # Add temporal position encoding
        pos = self.temporal_pos[:, :T, :]
        h = node_history + pos

        # Build causal-style key_padding_mask for transformer encoder
        # True = ignore in PyTorch convention
        if history_mask is not None:
            src_key_padding_mask = ~history_mask  # [B, T]
        else:
            src_key_padding_mask = None

        # Skip self-attention when T=1 (single token, self-attn is identity)
        # Also avoids PyTorch MHA ambiguity when B==T for small sequences
        if T > 1:
            h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)
        # When T=1, the single node embedding passes through directly to cross-attn

        # Cross-attend with learnable query
        # Only pass key_padding_mask when there are actual padded entries
        # to avoid PyTorch MHA shape ambiguity with small batch/seq combos
        query = self.phase_query.expand(B, -1, -1)  # [B, 1, D]
        cross_mask = src_key_padding_mask
        if cross_mask is not None and not cross_mask.any():
            cross_mask = None  # all valid, no need for mask
        phase_emb, _ = self.cross_attn(
            query=query,
            key=h,
            value=h,
            key_padding_mask=cross_mask,
        )
        phase_emb = self.norm(phase_emb.squeeze(1))  # [B, D]

        return phase_emb


# =============================================================================
# 2. FiLM Bias Generator: lang_emb + phase_emb -> attention bias
# =============================================================================

class FiLMBiasGenerator(nn.Module):
    """
    Generate per-head, per-memory-token attention bias using FiLM conditioning.

    Inspired by KC-VLA's FiLM generator, but fundamentally different:
      KC-VLA:  task_id (discrete) -> FiLM -> modulate phase_emb -> binary classifier
      Ours:    lang_emb (continuous) -> FiLM -> modulate phase_emb -> attention bias

    The bias is additive to the QK^T/sqrt(d) scores in memory cross-attention,
    allowing the model to up/down-weight specific memory entries based on
    the current task instruction and predicted phase.

    Input:
      lang_emb   [B, D_lang]   - language instruction embedding
      phase_emb  [B, D_node]   - predicted phase from PhasePredictor
      num_memory_tokens: int   - number of memory tokens (varies per step)

    Output:
      attn_bias  [B, num_heads, 1, M]  - additive bias for cross-attention
    """

    def __init__(
        self,
        lang_dim: int = 128,
        node_dim: int = 64,
        num_heads: int = 1,
        bias_hidden_dim: int = 128,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.node_dim = node_dim

        # FiLM: language -> (gamma, beta) to modulate phase embedding
        # This is the core KC-VLA-inspired component
        self.film_generator = nn.Sequential(
            nn.Linear(lang_dim, bias_hidden_dim),
            nn.GELU(),
            nn.Linear(bias_hidden_dim, node_dim * 2),  # gamma + beta
        )

        # Modulated phase -> per-head bias projection
        # Each head gets its own bias pattern over memory tokens
        self.bias_key_proj = nn.Linear(node_dim, num_heads)

        # Memory token projector: project memory features to node_dim
        # so we can compute compatibility with FiLM-conditioned phase
        self.memory_proj = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.GELU(),
            nn.Linear(node_dim, num_heads),
        )

        # Learnable temperature per head
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

    def forward(
        self,
        lang_emb: Tensor,       # [B, D_lang]
        phase_emb: Tensor,      # [B, D_node]
        memory_node_embeds: Tensor,  # [B, M, D_node] - node embeds stored with each memory
    ) -> Tensor:
        """Returns attn_bias [B, num_heads, 1, M]."""
        B, M, D = memory_node_embeds.shape

        # FiLM conditioning: lang -> (gamma, beta)
        film_params = self.film_generator(lang_emb)      # [B, D_node*2]
        gamma, beta = film_params.chunk(2, dim=-1)       # each [B, D_node]

        # Modulate phase embedding with language FiLM
        # This is where task instruction controls phase interpretation
        conditioned_phase = gamma * phase_emb + beta     # [B, D_node]

        # Compute per-head query bias from conditioned phase
        phase_bias = self.bias_key_proj(conditioned_phase)  # [B, num_heads]
        phase_bias = phase_bias.unsqueeze(-1).unsqueeze(-1)  # [B, H, 1, 1]

        # Compute per-head, per-memory-token compatibility
        mem_scores = self.memory_proj(memory_node_embeds)   # [B, M, num_heads]
        mem_scores = mem_scores.permute(0, 2, 1).unsqueeze(2)  # [B, H, 1, M]

        # Final bias = temperature * (phase_bias + memory_compatibility)
        attn_bias = self.temperature * (phase_bias + mem_scores)  # [B, H, 1, M]

        return attn_bias


# =============================================================================
# 3. Node Embedding Head: extract node embedding from memory features
# =============================================================================

class NodeEmbedHead(nn.Module):
    """
    Extract a compact node embedding from the memory-enhanced feature map.
    This embedding is stored in the memory bank alongside spatial features
    and used by PhasePredictor for phase estimation.

    The node embedding captures "where am I in the task graph" information.
    """

    def __init__(self, feat_dim: int = 256, node_dim: int = 64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.GELU(),
            nn.Linear(feat_dim // 2, node_dim),
        )

    def forward(self, memory_feat: Tensor) -> Tensor:
        """
        memory_feat: [B, N_tokens, D] - output of SAM2 memory attention
        Returns: [B, D_node] - L2-normalized node embedding
        """
        # Global average pool over spatial tokens
        pooled = memory_feat.mean(dim=1)  # [B, D]  (or dim=0 if seq-first)
        node_emb = self.proj(pooled)
        return F.normalize(node_emb, dim=-1)


# =============================================================================
# 4. FiLM-Conditioned Memory Attention Module (wraps existing SAM2 components)
# =============================================================================

class FiLMConditionedMemoryAttention(nn.Module):
    """
    Wrapper that augments SAM2's MemoryAttention with FiLM-based conditioning.

    This module:
      1. Maintains a history of node embeddings (per-episode state)
      2. Predicts current phase from node history
      3. Generates FiLM attention bias from (language, phase)
      4. Injects bias into SAM2's memory cross-attention via attn_mask

    Integration strategy:
      - Does NOT replace SAM2 memory attention (minimal code change)
      - Generates an additive attn_mask that is passed to existing
        MemoryAttention.forward() as the memory_attn_mask argument
      - SAM2's RoPEAttention already supports attn_mask in
        F.scaled_dot_product_attention

    Usage in mvt_sam2_single.py:
      film_module = FiLMConditionedMemoryAttention(...)
      # During forward:
      attn_bias = film_module.compute_bias(lang_emb, memory_node_embeds)
      output = sam2_memory_attention(
          curr, memory, curr_pos, memory_pos,
          memory_attn_mask=existing_mask + attn_bias,  # <-- only change
          num_obj_ptr_tokens=...
      )
      node_emb = film_module.update_history(memory_output)
    """

    def __init__(
        self,
        d_model: int = 256,
        lang_dim: int = 128,
        node_dim: int = 64,
        num_heads: int = 1,
        phase_predictor_layers: int = 2,
        phase_predictor_heads: int = 4,
        max_history: int = 16,
        film_hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.node_dim = node_dim
        self.num_heads = num_heads
        self.max_history = max_history

        # Sub-modules
        self.node_embed_head = NodeEmbedHead(
            feat_dim=d_model, node_dim=node_dim
        )
        self.phase_predictor = PhasePredictor(
            node_dim=node_dim,
            num_heads=phase_predictor_heads,
            num_layers=phase_predictor_layers,
            max_history=max_history,
            dropout=dropout,
        )
        self.film_bias_gen = FiLMBiasGenerator(
            lang_dim=lang_dim,
            node_dim=node_dim,
            num_heads=num_heads,
            bias_hidden_dim=film_hidden_dim,
        )

        # Language projection (if lang_emb dim != lang_dim)
        self.lang_proj = nn.Linear(lang_dim, lang_dim)  # identity-init possible

        # Episode state (not parameters, managed externally)
        self._node_history: List[Tensor] = []
        self._memory_node_embeds: List[Tensor] = []

    def reset_episode(self):
        """Call at the start of each episode to clear history."""
        self._node_history = []
        self._memory_node_embeds = []

    def compute_bias(
        self,
        lang_emb: Tensor,               # [B, D_lang]
        memory_node_embeds: Tensor,      # [B, M, D_node]
        node_history: Optional[Tensor] = None,   # [B, T, D_node]
        history_mask: Optional[Tensor] = None,    # [B, T]
    ) -> Tensor:
        """
        Compute FiLM attention bias for memory cross-attention.

        Args:
            lang_emb: language instruction embedding
            memory_node_embeds: node embeddings stored with each memory entry
            node_history: override internal history (for batch training)
            history_mask: mask for padded history entries

        Returns:
            attn_bias: [B, num_heads, 1, M] additive bias for attention scores
        """
        B = lang_emb.shape[0]

        # Use provided history or internal state
        if node_history is None:
            if len(self._node_history) > 0:
                node_history = torch.stack(self._node_history, dim=1)  # [B, T, D]
                history_mask = torch.ones(
                    B, node_history.shape[1],
                    dtype=torch.bool, device=lang_emb.device
                )
            else:
                node_history = torch.zeros(
                    B, 0, self.node_dim, device=lang_emb.device
                )
                history_mask = None

        # 1. Predict current phase
        phase_emb = self.phase_predictor(node_history, history_mask)  # [B, D_node]

        # 2. Project language
        lang_proj = self.lang_proj(lang_emb)  # [B, D_lang]

        # 3. Generate FiLM attention bias
        attn_bias = self.film_bias_gen(
            lang_proj, phase_emb, memory_node_embeds
        )  # [B, H, 1, M]

        return attn_bias

    def extract_node_embed(
        self,
        memory_output: Tensor,
        batch_first: bool = True,
    ) -> Tensor:
        """
        Extract node embedding from memory attention output.
        Call after SAM2 memory attention forward pass.

        Args:
            memory_output: [B, N, D] if batch_first else [N, B, D]
            batch_first: if False, transpose to [B, N, D] first.
                SAM2 MemoryAttention returns seq-first [N, B, D] after
                its internal transpose, so pass batch_first=False when
                using SAM2's raw output.

        Returns:
            node_emb: [B, D_node]
        """
        if not batch_first:
            memory_output = memory_output.transpose(0, 1)

        return self.node_embed_head(memory_output)

    def update_history(self, node_emb: Tensor):
        """
        Add current node embedding to episode history.
        Call after extract_node_embed.

        Args:
            node_emb: [B, D_node] - detached for history tracking
        """
        self._node_history.append(node_emb.detach())
        if len(self._node_history) > self.max_history:
            self._node_history = self._node_history[-self.max_history:]

    def add_memory_node_embed(self, node_emb: Tensor):
        """Store node embedding alongside memory bank entry."""
        self._memory_node_embeds.append(node_emb.detach())

    def get_memory_node_embeds(self) -> Optional[Tensor]:
        """Get stacked memory node embeddings [B, M, D_node]."""
        if len(self._memory_node_embeds) == 0:
            return None
        return torch.stack(self._memory_node_embeds, dim=1)


# =============================================================================
# 5. Training losses
# =============================================================================

class FiLMMemoryLosses(nn.Module):
    """
    Auxiliary losses for training the FiLM-conditioned memory module.

    L_node_contrastive: Ensure node embeddings at similar 3D positions
        cluster together, and different positions separate.
    L_phase_consistency: Phase predictor should produce similar outputs
        for similar task-progress states.
    L_bias_regularization: Prevent attention bias from dominating;
        keep it as a gentle modulation.
    """

    def __init__(
        self,
        node_dim: int = 64,
        position_threshold: float = 0.05,  # 5cm in normalized coords
        bias_reg_weight: float = 0.01,
    ):
        super().__init__()
        self.position_threshold = position_threshold
        self.bias_reg_weight = bias_reg_weight

    def node_contrastive_loss(
        self,
        node_embeds: Tensor,    # [B, T, D_node] - all node embeds in episode
        positions_3d: Tensor,   # [B, T, 3] - corresponding 3D positions
        mask: Optional[Tensor] = None,  # [B, T] valid mask
    ) -> Tensor:
        """
        InfoNCE-style contrastive loss on node embeddings.
        Pairs with 3D distance < threshold are positives.
        """
        B, T, D = node_embeds.shape
        if T < 2:
            return torch.tensor(0.0, device=node_embeds.device)

        # Pairwise distances in 3D space
        pos_diff = positions_3d.unsqueeze(2) - positions_3d.unsqueeze(1)  # [B, T, T, 3]
        pos_dist = pos_diff.norm(dim=-1)  # [B, T, T]

        # Positive pairs: close in 3D
        pos_mask = (pos_dist < self.position_threshold).float()
        # Remove diagonal
        eye = torch.eye(T, device=node_embeds.device).unsqueeze(0)
        pos_mask = pos_mask * (1 - eye)

        # Cosine similarity of node embeddings
        node_norm = F.normalize(node_embeds, dim=-1)
        sim = torch.bmm(node_norm, node_norm.transpose(1, 2))  # [B, T, T]
        sim = sim / 0.1  # temperature

        # Apply valid mask if provided
        if mask is not None:
            valid_pairs = mask.unsqueeze(2) * mask.unsqueeze(1)  # [B, T, T]
            sim = sim * valid_pairs
            pos_mask = pos_mask * valid_pairs

        # InfoNCE: for each anchor, log_softmax over all pairs, take positive
        log_prob = F.log_softmax(sim, dim=-1)
        loss = -(log_prob * pos_mask).sum() / (pos_mask.sum() + 1e-8)

        return loss

    def bias_regularization(self, attn_bias: Tensor) -> Tensor:
        """
        L2 regularization on attention bias to keep it small.
        The bias should gently modulate, not overpower QK^T scores.
        """
        return self.bias_reg_weight * (attn_bias ** 2).mean()

    def forward(
        self,
        node_embeds: Tensor,
        positions_3d: Tensor,
        attn_bias: Tensor,
        mask: Optional[Tensor] = None,
    ) -> dict:
        """Compute all auxiliary losses."""
        l_contrastive = self.node_contrastive_loss(
            node_embeds, positions_3d, mask
        )
        l_bias_reg = self.bias_regularization(attn_bias)

        return {
            "loss_node_contrastive": l_contrastive,
            "loss_bias_reg": l_bias_reg,
            "loss_total_aux": l_contrastive + l_bias_reg,
        }


# =============================================================================
# 6. Convenience: build full module from config
# =============================================================================

def build_film_memory_module(config: dict) -> FiLMConditionedMemoryAttention:
    """
    Build FiLMConditionedMemoryAttention from a config dict.

    Expected config keys (with defaults):
        d_model: 256
        lang_dim: 128
        node_dim: 64
        num_heads: 1
        phase_predictor_layers: 2
        phase_predictor_heads: 4
        max_history: 16
        film_hidden_dim: 128
        dropout: 0.1
    """
    defaults = dict(
        d_model=256,
        lang_dim=128,
        node_dim=64,
        num_heads=1,
        phase_predictor_layers=2,
        phase_predictor_heads=4,
        max_history=16,
        film_hidden_dim=128,
        dropout=0.1,
    )
    defaults.update(config)
    return FiLMConditionedMemoryAttention(**defaults)


def count_parameters(module: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


# =============================================================================
# 7. Example usage / smoke test
# =============================================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cpu"

    # Config
    B, N_tokens, D_model = 2, 64, 256
    D_lang, D_node = 128, 64
    M = 5  # number of memory entries
    H = 1  # num_heads for bias

    # Build module
    film_module = FiLMConditionedMemoryAttention(
        d_model=D_model,
        lang_dim=D_lang,
        node_dim=D_node,
        num_heads=H,
        phase_predictor_layers=2,
        phase_predictor_heads=4,
        max_history=16,
        film_hidden_dim=128,
    ).to(device)

    print(f"Total parameters: {count_parameters(film_module):,}")
    print(f"  NodeEmbedHead:    {count_parameters(film_module.node_embed_head):,}")
    print(f"  PhasePredictor:   {count_parameters(film_module.phase_predictor):,}")
    print(f"  FiLMBiasGenerator:{count_parameters(film_module.film_bias_gen):,}")

    # Simulate episode
    film_module.reset_episode()
    lang_emb = torch.randn(B, D_lang, device=device)

    for step in range(3):
        # Simulate memory entries accumulated so far
        num_mem = step + 1
        memory_node_embeds = torch.randn(B, num_mem, D_node, device=device)

        # Compute FiLM bias
        attn_bias = film_module.compute_bias(lang_emb, memory_node_embeds)
        print(f"Step {step}: attn_bias shape = {attn_bias.shape}")
        # Expected: [B, H, 1, num_mem]

        # Simulate SAM2 memory attention output
        memory_output = torch.randn(B, N_tokens, D_model, device=device)

        # Extract and store node embedding
        node_emb = film_module.extract_node_embed(memory_output)
        film_module.update_history(node_emb)
        film_module.add_memory_node_embed(node_emb)

    # Test losses
    loss_module = FiLMMemoryLosses(node_dim=D_node)
    node_embeds = torch.randn(B, 3, D_node)
    positions_3d = torch.randn(B, 3, 3)
    attn_bias_dummy = torch.randn(B, H, 1, 3)

    losses = loss_module(node_embeds, positions_3d, attn_bias_dummy)
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")

    print("\nSmoke test passed.")
