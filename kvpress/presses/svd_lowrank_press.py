# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from kvpress.presses.scorer_press import ScorerPress


@torch.no_grad()
def lowrank_svd_for_scoring(
    x: torch.Tensor,
    rank: Optional[int] = None,
    niter: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Low-rank SVD using torch.svd_lowrank for KV cache scoring.
    
    This method uses PyTorch's built-in low-rank SVD approximation which is
    faster than full SVD for low-rank approximations.
    
    - Input: assumed to be in low precision (BF16/FP16), used directly
    - SVD: performed in FP32 because CUDA backend doesn't support FP16 QR
    - Output: converted back to input dtype
    
    Parameters
    ----------
    x : torch.Tensor
        Input tensor with shape (bs, nh, sl, hd) where:
        - bs: batch size
        - nh: number of heads
        - sl: sequence length
        - hd: head dimension
    rank : int, optional
        Target rank for approximation. If None, uses min(128, sl, nh*hd).
    niter : int, default=2
        Number of subspace iterations for improved accuracy.
    
    Returns
    -------
    U : torch.Tensor
        Left singular vectors with shape (bs, sl, r)
    S : torch.Tensor
        Singular values with shape (bs, r)
    """
    input_dtype = x.dtype
    bs, nh, sl, hd = x.shape
    
    # Reshape: (bs, nh, sl, hd) -> (bs, sl, nh*hd)
    x2d = x.transpose(1, 2).reshape(bs, sl, nh * hd)
    m, n = sl, nh * hd
    
    # Determine rank (default to 128 for efficiency)
    if rank is None:
        r = min(m, n)
    else:
        r = min(rank, m, n)
    
    # Force FP32 for numerical stability and backend support
    x2d_fp32 = x2d.to(torch.float32)
    U, S, V = torch.svd_lowrank(x2d_fp32, q=r, niter=niter)
    del x2d_fp32, V
    
    # Convert back to input dtype
    U = U.to(input_dtype)
    S = S.to(input_dtype)
    
    return U, S


@dataclass
class SVDLowrankPress(ScorerPress):
    """
    Low-rank SVD-based KV cache compression using torch.svd_lowrank.
    
    This press uses PyTorch's built-in torch.svd_lowrank for computing importance
    scores. It provides a middle ground between full SVD (accurate but slow) and
    randomized SVD (fast but may be less accurate).
    
    The method:
    1. Reshapes keys from (bs, nh, sl, hd) to (bs, sl, nh*hd)
    2. Computes low-rank SVD approximation in FP32
    3. Uses U and S to compute importance scores
    
    Parameters
    ----------
    compression_ratio : float, default=0.0
        Fraction of key-value pairs to remove during compression.
        Must be between 0 and 1.
    rank : int, optional
        Target rank for SVD approximation. If None, defaults to min(128, seq_len).
    niter : int, default=2
        Number of subspace iterations. More iterations improve accuracy.
    svd_method : str, default="weighted"
        Scoring method:
        - "weighted": Weight by singular values (recommended)
        - "unweighted": Uniform weighting 
    normalize : bool, default=True
        Whether to normalize scores across sequence dimension.
    
    Examples
    --------
    >>> from kvpress.presses.svd_lowrank_press import SVDLowrankPress
    >>> from transformers import pipeline
    >>> 
    >>> pipe = pipeline("kv-press-text-generation", model="meta-llama/Llama-3.1-8B-Instruct")
    >>> 
    >>> # Low-rank SVD with 50% compression
    >>> press = SVDLowrankPress(compression_ratio=0.5)
    >>> result = pipe(context="Long text...", question="Question?", press=press)
    
    Notes
    -----
    - Faster than full SVD for low-rank approximations
    - Uses torch.svd_lowrank which is optimized for this use case
    - May be slower than fully randomized methods for very long sequences
    """
    
    compression_ratio: float = 0.0
    rank: Optional[int] = None
    niter: int = 2
    svd_method: str = "weighted"
    normalize: bool = True
    
    def __post_init__(self):
        """Validate parameters after initialization."""
        super().__post_init__()
        
        assert self.svd_method in ["weighted", "unweighted"], \
            f"svd_method must be 'weighted' or 'unweighted', got '{self.svd_method}'"
        
        if self.rank is not None:
            assert self.rank > 0, f"rank must be positive, got {self.rank}"
        
        assert self.niter >= 0, f"niter must be non-negative, got {self.niter}"
    
    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        """
        Compute importance scores using low-rank SVD.
        
        Parameters
        ----------
        module : nn.Module
            The transformer attention layer.
        hidden_states : torch.Tensor
            Input embeddings with shape (batch_size, seq_len, hidden_dim).
        keys : torch.Tensor
            Key tensors with shape (batch_size, num_kv_heads, seq_len, head_dim).
        values : torch.Tensor
            Value tensors with shape (batch_size, num_kv_heads, seq_len, head_dim).
        attentions : torch.Tensor
            Attention weights (may be None).
        kwargs : dict
            Additional arguments from the forward pass.
        
        Returns
        -------
        torch.Tensor
            Importance scores with shape (batch_size, num_kv_heads, seq_len).
        """
        bsz, num_kv_heads, seq_len, head_dim = keys.shape
        
        # Dynamic rank selection based on sequence length
        # If rank is None, automatically choose based on seq_len
        if self.rank is None:
            if seq_len < 50000:
                effective_rank = min(128, seq_len)
            else:
                effective_rank = min(256, seq_len)
        else:
            effective_rank = min(self.rank, seq_len)
        
        # Use low-rank SVD
        U, S = lowrank_svd_for_scoring(
            keys, 
            rank=effective_rank, 
            niter=self.niter
        )
        
        # DEBUG: Collect all layer shapes for alignment verification
        if not hasattr(self, '_layer_shapes'):
            self._layer_shapes = []
            self._context_count = 0
        
        layer_idx = len(self._layer_shapes)
        self._layer_shapes.append((tuple(keys.shape), effective_rank, tuple(U.shape), tuple(S.shape)))
        
        # Print each layer (only for first context to avoid spam)
        if self._context_count == 0:
            print(f"[SVDLowrank] Layer {layer_idx:2d}: keys={tuple(keys.shape)}, rank={effective_rank}, U={tuple(U.shape)}, S={tuple(S.shape)}")
        
        # Print summary after all 32 layers (Llama-3.1-8B has 32 layers)
        if len(self._layer_shapes) == 32:
            if self._context_count == 0:
                unique_U = set(s[2] for s in self._layer_shapes)
                unique_S = set(s[3] for s in self._layer_shapes)
                unique_rank = set(s[1] for s in self._layer_shapes)
                print(f"[SVDLowrank] ===== SUMMARY (32 layers) =====")
                print(f"[SVDLowrank]   Unique U shapes: {unique_U}")
                print(f"[SVDLowrank]   Unique S shapes: {unique_S}")
                print(f"[SVDLowrank]   Unique ranks: {unique_rank}")
                if len(unique_U) == 1 and len(unique_S) == 1:
                    print(f"[SVDLowrank]   All 32 layers ALIGNED!")
                else:
                    print(f"[SVDLowrank]   WARNING: Layers NOT aligned!")
                print(f"[SVDLowrank] ================================")
            self._layer_shapes = []  # Reset for next context
            self._context_count += 1
        
        # U shape: (bsz, seq_len, r)
        # S shape: (bsz, r)
        
        # Compute scores
        if self.svd_method == "weighted":
            # Weighted by singular values: score_i = sum_j(|U_ij| * S_j)
            scores_flat = (U.abs() * S.unsqueeze(1)).sum(dim=-1)  # (bsz, seq_len)
        else:  # "unweighted"
            scores_flat = U.abs().sum(dim=-1)  # (bsz, seq_len)
        
        # Replicate scores across all heads
        # Shape: (bsz, seq_len) -> (bsz, num_kv_heads, seq_len)
        scores = scores_flat.unsqueeze(1).expand(bsz, num_kv_heads, seq_len)
        
        # Optionally normalize scores
        if self.normalize:
            mean = scores.mean(dim=-1, keepdim=True)
            std = scores.std(dim=-1, keepdim=True).clamp_min(1e-6)
            scores = (scores - mean) / std
        
        # CRITICAL: Negate scores
        # SVD contribution high = common pattern = should be REMOVED
        # SVD contribution low = unique/important info = should be KEPT
        return -scores

