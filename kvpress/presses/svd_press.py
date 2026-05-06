# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from kvpress.presses.scorer_press import ScorerPress


@torch.no_grad()
def _orthonormalize(Y: torch.Tensor, reg_eps: float = 1e-4, cond_threshold: float = 1e6) -> torch.Tensor:
    """
    Orthonormalize a batched tall-skinny sketch matrix.

    Cholesky QR is fast for well-conditioned sketches. Standard QR is used as a
    fallback when the sketch is ill-conditioned or rank deficient.
    """
    Y_fp32 = Y.to(torch.float32)
    batch_size, _, sketch_size = Y_fp32.shape
    eye = torch.eye(sketch_size, device=Y.device, dtype=torch.float32).expand(batch_size, -1, -1)
    gram = torch.bmm(Y_fp32.transpose(1, 2), Y_fp32) + reg_eps * eye
    diag = torch.diagonal(gram, dim1=-2, dim2=-1).abs()
    cond = diag.max(dim=-1).values / diag.min(dim=-1).values.clamp(min=reg_eps)

    try:
        if not bool(torch.all(cond < cond_threshold)):
            raise RuntimeError("ill-conditioned sketch")
        R = torch.linalg.cholesky(gram, upper=True)
        Q = torch.linalg.solve_triangular(R, Y_fp32, upper=True, left=False)
    except RuntimeError:
        Q, _ = torch.linalg.qr(Y_fp32, mode="reduced")

    return Q.to(Y.dtype)


@torch.no_grad()
def randomized_svd_scores(
    keys: torch.Tensor,
    rank: int,
    oversample: int = 8,
    n_iter: int = 2,
    chol_reg_eps: float = 1e-4,
    chol_cond_threshold: float = 1e6,
) -> torch.Tensor:
    """
    Compute token scores from a randomized SVD approximation of the key cache.

    Keys are flattened across KV heads and head dimension, producing one score
    per token position. Higher scores mean the token contributes more to the
    low-rank key subspace.
    """
    batch_size, num_kv_heads, seq_len, head_dim = keys.shape
    matrix_width = num_kv_heads * head_dim
    effective_rank = min(rank, seq_len, matrix_width)
    sketch_size = min(effective_rank + oversample, seq_len, matrix_width)
    sketch_dtype = torch.bfloat16 if keys.is_cuda else torch.float32

    X = keys.transpose(1, 2).reshape(batch_size, seq_len, matrix_width).contiguous().to(sketch_dtype)
    X_t = X.transpose(1, 2)
    omega = torch.randn(batch_size, matrix_width, sketch_size, device=keys.device, dtype=sketch_dtype)
    Y = torch.bmm(X, omega)

    for _ in range(n_iter):
        Y = torch.bmm(X, torch.bmm(X_t, Y))
        norms = torch.linalg.vector_norm(Y, dim=1, keepdim=True, dtype=torch.float32).clamp(min=1e-5)
        Y = Y / norms.to(Y.dtype)

    Q = _orthonormalize(Y, reg_eps=chol_reg_eps, cond_threshold=chol_cond_threshold)
    B = torch.bmm(Q.transpose(1, 2), X).to(torch.float32)
    U_hat, singular_values, _ = torch.linalg.svd(B, full_matrices=False)
    U = torch.bmm(Q.to(torch.float32), U_hat)[:, :, :effective_rank]
    singular_values = singular_values[:, :effective_rank]

    return (U.abs() * singular_values.unsqueeze(1)).sum(dim=-1)


@dataclass
class FastSVDPress(ScorerPress):
    """
    SVD-based KV cache compression using randomized low-rank scoring.

    The press flattens the key cache across KV heads and head dimension, runs a
    randomized SVD approximation, and scores each token by its weighted
    contribution to the low-rank key subspace. Tokens with larger low-rank
    contribution are treated as more redundant, so the returned score is negated
    before the common ScorerPress top-k pruning logic is applied.

    Parameters
    ----------
    compression_ratio : float, default=0.0
        Fraction of key-value pairs to remove during compression.
    rank : int, optional
        Target rank for the randomized SVD approximation. If unset, the rank is
        selected from the sequence length.
    oversample : int, default=8
        Additional random projection dimensions used by randomized SVD.
    n_iter : int, default=2
        Number of power iterations used to improve the approximation.
    normalize : bool, default=True
        Whether to normalize token scores before pruning.
    """

    compression_ratio: float = 0.0
    rank: Optional[int] = None
    oversample: int = 8
    n_iter: int = 2
    normalize: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.rank is not None:
            assert self.rank > 0, f"rank must be positive, got {self.rank}"
        assert self.oversample >= 0, f"oversample must be non-negative, got {self.oversample}"
        assert self.n_iter >= 0, f"n_iter must be non-negative, got {self.n_iter}"

    def _get_rank(self, seq_len: int, head_width: int) -> int:
        if self.rank is not None:
            return min(self.rank, seq_len, head_width)
        return min(128 if seq_len < 50_000 else 256, seq_len, head_width)

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        batch_size, num_kv_heads, seq_len, head_dim = keys.shape
        rank = self._get_rank(seq_len, num_kv_heads * head_dim)
        scores = randomized_svd_scores(
            keys,
            rank=rank,
            oversample=self.oversample,
            n_iter=self.n_iter,
        )

        if self.normalize:
            mean = scores.mean(dim=-1, keepdim=True)
            std = scores.std(dim=-1, keepdim=True).clamp_min(1e-6)
            scores = (scores - mean) / std

        return -scores.unsqueeze(1).expand(batch_size, num_kv_heads, seq_len)
