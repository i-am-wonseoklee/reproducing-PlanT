"""Section IV: auxiliary prediction heads."""

import torch.nn as nn
from torch import Tensor


class Head(nn.Module):
    """Per-obstacle classification of next-step attributes (auxiliary task).

    Args:
        d_model: size of each per-obstacle encoder feature.
        n_position_bins: discretisation bins for future position.
        n_velocity_bins: bins for future velocity.
        n_orientation_bins: bins for future orientation.
    """

    def __init__(
        self,
        d_model: int,
        n_position_bins: int = 128,
        n_velocity_bins: int = 4,
        n_orientation_bins: int = 32,
    ):
        super().__init__()
        # One independent linear classifier per attribute. Each maps a per-obstacle
        # feature to logits over that attribute's discretisation bins.
        self.position_x = nn.Linear(
            in_features=d_model, out_features=n_position_bins, bias=True
        )
        self.position_y = nn.Linear(
            in_features=d_model, out_features=n_position_bins, bias=True
        )
        self.velocity = nn.Linear(
            in_features=d_model, out_features=n_velocity_bins, bias=True
        )
        self.orientation = nn.Linear(
            in_features=d_model, out_features=n_orientation_bins, bias=True
        )

    def forward(self, obstacle_feats: Tensor) -> dict[str, Tensor]:
        """Classify each obstacle's future attributes.

        Args:
            obstacle_feats: (B, N, d_model) per-obstacle encoder features.

        Returns:
            dict of logits:
                "position_x":  (B, N, n_position_bins)
                "position_y":  (B, N, n_position_bins)
                "velocity":    (B, N, n_velocity_bins)
                "orientation": (B, N, n_orientation_bins)
        """
        # Each Linear acts on the last axis only, so (B, N, d_model) maps to
        # (B, N, n_bins) for every obstacle in one pass.
        return {
            "position_x": self.position_x(obstacle_feats),
            "position_y": self.position_y(obstacle_feats),
            "velocity": self.velocity(obstacle_feats),
            "orientation": self.orientation(obstacle_feats),
        }
