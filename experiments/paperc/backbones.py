"""FT-style transformer backbone over per-feature tokens (Paper C).

The FT-Transformer of Gorishniy et al. (NeurIPS 2021) embeds each feature
as one token and attends across features with a CLS readout. This module
reproduces that architecture but accepts ARBITRARY per-feature token
vectors (raw scalars, PLE encodings, or OT-layer assignments), so every
tokenizer arm of experiment C3 runs under the identical backbone -- the
designated camera-ready comparison of the Paper C draft.
"""

# Cesar Claros <cesar.claros@outlook.com>
# Copyright (C) 2026

from __future__ import annotations

import torch
from torch import Tensor, nn


class FeatureTokenTransformer(nn.Module):
    """Per-feature linear embeddings + transformer encoder + CLS head.

    Parameters
    ----------
    n_features:
        Number of feature tokens.
    token_dim:
        Dimension of each incoming per-feature token vector.
    d_model:
        Embedding width (FT-Transformer's token dimension).
    n_layers:
        Number of encoder blocks.
    n_heads:
        Attention heads per block.
    dropout:
        Dropout inside the encoder blocks.

    Notes
    -----
    Embedding weights are per feature, matching FT-Transformer's
    continuous-feature tokenizer (for ``token_dim=1`` this is exactly
    ``w_f * x + b_f``). Pre-norm blocks with GELU and a CLS readout
    through LayerNorm -> ReLU -> Linear follow the reference
    implementation (rtdl); torch-native so no extra dependency and no
    coupling to rtdl's raw-float feature tokenizer, which would bypass
    the tokenizers under study.
    """

    def __init__(self, n_features: int, token_dim: int, d_model: int = 64,
                 n_layers: int = 2, n_heads: int = 4,
                 dropout: float = 0.1, n_out: int = 1) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads "
                f"({n_heads}).")
        self.weight = nn.Parameter(
            torch.empty(n_features, token_dim, d_model))
        self.bias = nn.Parameter(torch.zeros(n_features, d_model))
        nn.init.xavier_uniform_(self.weight)
        self.cls = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=2 * d_model,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True)
        # nested-tensor fast path never applies here (pre-norm, no
        # padding mask); disable explicitly to keep construction quiet.
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers,
                                             enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.ReLU(),
                                  nn.Linear(d_model, n_out))

    def forward(self, tokens: Tensor) -> Tensor:
        """Logits of shape ``(batch,)`` from per-feature tokens of shape
        ``(batch, n_features, token_dim)``."""
        h = torch.einsum("bft,fth->bfh", tokens, self.weight) + self.bias
        cls = self.cls.expand(len(tokens), 1, -1)
        h = self.encoder(torch.cat([cls, h], dim=1))
        return self.head(h[:, 0]).squeeze(-1)
