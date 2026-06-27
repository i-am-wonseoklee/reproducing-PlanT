"""Section IV: waypoint decoder."""

import torch
import torch.nn as nn
from torch import Tensor


class Decoder(nn.Module):
    """Autoregressive GRU that predicts n_predictions (x, y) waypoints.

    Args:
        d_model: size of the [CLS] feature.
        d_hidden: width the [CLS] feature is projected to (paper: 64).
        n_predictions: number of waypoints to roll out (4 for PlanT).
    """

    def __init__(self, d_model: int, d_hidden: int, n_predictions: int = 4):
        super().__init__()
        self.n_predictions = n_predictions

        # The traffic light flag is concatenated to the projected CLS feature.
        self.d_gru_hidden = d_hidden + 1

        # Project the [CLS] scene summary down to the GRU hidden width.
        self.projection = nn.Linear(
            in_features=d_model, out_features=d_hidden, bias=True
        )

        # One autoregressive step. Input each step is [prev_waypoint; target_point].
        self.cell = nn.GRUCell(input_size=4, hidden_size=self.d_gru_hidden)

        # Maps the hidden state to a per-step delta (dx, dy).
        self.output = nn.Linear(
            in_features=self.d_gru_hidden, out_features=2, bias=True
        )

    def forward(
        self,
        token_cls: Tensor,
        traffic_light: Tensor,
        target_point: Tensor,
    ) -> Tensor:
        """Roll out future waypoints.

        Args:
            token_cls: (B, d_model) encoder [CLS] output.
            traffic_light: (B, 1) red-light flag.
            target_point: (B, 2) goal point in ego frame.

        Returns:
            (B, n_predictions, 2) cumulative ego-frame waypoints.
        """
        B = token_cls.shape[0]

        # Initial hidden state: projected CLS summary + traffic light flag.
        # (B, hidden_size) cat (B, 1) -> (B, hidden_size + 1).
        hidden = torch.cat([self.projection(token_cls), traffic_light], dim=-1)

        # First "previous waypoint" w_0 is the origin (ego is at (0, 0)).
        waypoint = torch.zeros(B, 2, device=token_cls.device, dtype=token_cls.dtype)

        # Roll out n_predictions waypoints autoregressively.
        waypoints = []
        for _ in range(self.n_predictions):
            step_input = torch.cat([waypoint, target_point], dim=-1)
            hidden = self.cell(step_input, hidden)
            delta = self.output(hidden)
            waypoint = waypoint + delta
            waypoints.append(waypoint)

        # (B, n_predictions, 2): stack along a new step axis.
        return torch.stack(waypoints, dim=1)
