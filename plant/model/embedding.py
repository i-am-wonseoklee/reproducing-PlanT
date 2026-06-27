"""Section II: token embedding.

Turns the raw 6D obstacle and route feature vectors into a single transformer
input sequence:

    [CLS] + obstacle tokens + route tokens

Each step:
    1. linear projection   6 -> d_model
    2. token-type embedding (obstacle vs route) added to the projected tokens
    3. a learned [CLS] token prepended to the front
    4. a key_padding_mask built so the encoder ignores padded obstacle slots

The [CLS] token is its own learned parameter, so it already carries a unique
identity; it gets no token-type embedding (that table only distinguishes
obstacle vs route).

The corresponding paper section is "Object-level Input Representation".
"""

import torch
import torch.nn as nn
from torch import Tensor


class Embedding(nn.Module):
    """Project 6D tokens to d_model and assemble the encoder input sequence.

    Args:
        d_model: transformer hidden size.
        dropout: dropout applied to the assembled sequence (0 to disable).
    """

    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()

        # This linear projection is shared for all token types (obstacle, route).
        # The type distinction is made by the learned token-type embedding below.
        self.projection = nn.Linear(in_features=6, out_features=d_model, bias=True)

        # 0 is for obstacle, 1 is for route.
        self.type_embedding = nn.Embedding(num_embeddings=2, embedding_dim=d_model)

        # In almost every PyTorch module, the first axis is the batch dimension.
        # The expected shape (without batch) of Embedding layer outputs is (L, d_model).
        # Considering the batch dimension, the output shape is (B, L, d_model).
        # To expand the [CLS] token to the batch, the shape is (1, 1, d_model).
        self.token_cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Dropout masks some of the input tokens during training to prevent overfitting.
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        feature_obstacles: Tensor,
        feature_routes: Tensor,
        mask_obstacles: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Assemble the encoder input.

        Args:
            feature_obstacles: (B, N, 6) obstacle feature (N is varying for each batch).
            feature_routes: (B, R, 6) route segment feature (R = 2).
            mask_obstacles: (B, N) bool, True = valid obstacle slot.

        Returns:
            tokens: (B, 1 + N + R, d_model) input sequence.
            masks: (B, 1 + N + R) bool, True = position to IGNORE.
        """
        B = feature_obstacles.shape[0]

        # Retrieve the learned token-type embeddings for obstacles and routes.
        type_embedding_obstacles = self.type_embedding.weight[0]
        type_embedding_routes = self.type_embedding.weight[1]

        # Embed the obstacles and routes.
        token_obstacles = self.projection(feature_obstacles) + type_embedding_obstacles
        token_routes = self.projection(feature_routes) + type_embedding_routes

        # `expand` does not allocate new memory.
        # It just creates a view of the original tensor with the specified shape.
        token_cls = self.token_cls.expand(B, -1, -1)

        # Concatenate all the tokens.
        tokens = torch.cat([token_cls, token_obstacles, token_routes], dim=1)

        # Apply dropout to the tokens.
        tokens = self.dropout(tokens)

        # CLS, route tokens are always kept, so their mask is all False.
        mask_cls = torch.zeros(
            B,
            1,
            dtype=torch.bool,
            device=feature_obstacles.device,
        )
        mask_routes = torch.zeros(
            B,
            feature_routes.shape[1],
            dtype=torch.bool,
            device=feature_obstacles.device,
        )
        masks = torch.cat([mask_cls, ~mask_obstacles, mask_routes], dim=1)

        return tokens, masks
