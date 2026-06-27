"""Section III: transformer encoder.

A thin wrapper over torch.nn.TransformerEncoder. It consumes the embedded token
sequence from Embedding and returns contextualised features for every position.
"""

import torch.nn as nn
from torch import Tensor


class Encoder(nn.Module):
    """Multi-head self-attention encoder stack.

    Args:
        d_model: hidden size (matches Embedding).
        n_heads: number of attention heads.
        n_layers: number of stacked encoder layers.
        dropout: dropout inside the encoder layers.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        # A standard TransformerEncoderLayer = multi-head self-attention + a
        # feedforward MLP, each wrapped in a residual connection and LayerNorm.
        #
        # This `dropout` is NOT the same one as in the embedding layer. The
        # embedding applies dropout once, on the assembled input sequence. Here
        # the single value is reused at several points INSIDE every layer:
        #   - on the attention weights (the softmax output)
        #   - on the attention output, before the residual add
        #   - inside the feedforward MLP (between its two linears)
        #   - on the feedforward output, before the residual add
        # and all of this repeats for each of the n_layers stacked layers.
        # So it regularises the internal attention/MLP computations, whereas the
        # embedding dropout only regularises the input. Different locations, not
        # a duplicate.
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )

        # Stack n_layers identical layers into the full encoder.
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, tokens: Tensor, masks: Tensor) -> Tensor:
        """Contextualise the token sequence.

        Args:
            tokens: (B, L, d_model) from Embedding.
            masks: (B, L) bool, True = ignore (padded slots).

        Returns:
            (B, L, d_model) encoded sequence (same length as input).
        """
        return self.encoder(tokens, src_key_padding_mask=masks)
