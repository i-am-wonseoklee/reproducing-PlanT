"""Full PlanT model"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from plant.model.decoder import Decoder
from plant.model.embedding import Embedding
from plant.model.encoder import Encoder
from plant.model.head import Head


class PlanT(nn.Module):
    """PlanT planning transformer.

    Args:
        config: dict of hyperparameters (see configs/plant_*.yaml).
    """

    def __init__(self, config: dict):
        """Constructor.

        Args:
            config: dict of hyperparameters (see configs/plant_*.yaml).
        """
        super().__init__()
        self.config = config

        self.embedding = Embedding(
            config["d_model"],
            dropout=config["dropout"],
        )
        self.encoder = Encoder(
            config["d_model"],
            n_heads=config["n_heads"],
            n_layers=config["n_layers"],
            dropout=config["dropout"],
        )
        self.decoder = Decoder(
            config["d_model"],
            d_hidden=config["gru_hidden_size"],
            n_predictions=config["n_predictions"],
        )
        self.head = Head(
            config["d_model"],
            n_position_bins=config["n_position_bins"],
            n_velocity_bins=config["n_velocity_bins"],
            n_orientation_bins=config["n_orientation_bins"],
        )

    def forward(self, batch: dict) -> dict:
        """Run the model on a collated batch.

        Args:
            batch: dict of tensors with the Dataset keys, batched (B, ...):
                feature_obstacles      (B, N, 6)
                mask_obstacles         (B, N) bool
                feature_route_segments (B, R, 6)
                feature_traffic_light  (B, 1)
                feature_target_point   (B, 2)

        Returns:
            dict:
                "waypoints":     (B, n_predictions, 2)
                "future_logits": {"position_x", "position_y", "velocity",
                                  "orientation"}
        """
        # 0. The input is a batch containing multiple frames.
        # Each frame has a variable number of obstacles, but the batch dimension should
        # be fixed. Therefore, the upstream collate function pads the obstacle features
        # to a fixed size N, and provides a mask to indicate which obstacles are valid.
        N = batch["feature_obstacles"].shape[1]

        # 1. embed tokens and build the padding mask (True = IGNORE):
        # - tokens: (B, 1 + N + R, d_model) = [[CLS, obstacles..., routes...], ...]
        # - masks: (B, 1 + N + R) = [[F(CLS), T/F per obstacle, F(routes...)], ...]
        tokens, masks = self.embedding(
            batch["feature_obstacles"],
            batch["feature_route_segments"],
            batch["mask_obstacles"],
        )

        # 2. contextualise with self-attention.
        # - encoded: (B, 1 + N + R, d_model) = [[CLS, obstacles..., routes...], ...]
        encoded = self.encoder(tokens, masks)

        # 3. split the sequence.
        # - feature_cls: (B, d_model) = CLS summary token.
        # - feature_obstacles: (B, N, d_model) = contextualised obstacle tokens.
        feature_cls = encoded[:, 0]
        feature_obstacles = encoded[:, 1 : 1 + N]

        # 4. main task: roll out waypoints from the CLS summary.
        # - waypoints: (B, n_predictions, 2) = predicted future positions.
        waypoints = self.decoder(
            feature_cls,
            batch["feature_traffic_light"],
            batch["feature_target_point"],
        )

        # 5. auxiliary task: classify each obstacle's next-step attributes.
        # - future_logits: dict of (B, N, n_bins) for each attribute.
        future_logits = self.head(feature_obstacles)

        return {"waypoints": waypoints, "future_logits": future_logits}

    def compute_loss(self, pred: dict, batch: dict) -> tuple[Tensor, dict]:
        """Compute the combined planning + auxiliary loss.

        Args:
            pred: output of forward().
            batch: same batch passed to forward(); supplies the targets
                feature_waypoints        (B, P, 2)
                feature_future_obstacles (B, N, 6)
                mask_future_obstacles    (B, N)

        Returns:
            (total_loss, logs) where logs is a dict of detached scalars for
            monitoring.
        """
        # 1. waypoint L1 loss (main task).
        loss_wp = F.l1_loss(pred["waypoints"], batch["feature_waypoints"])

        # 2. discretise the ground-truth future attributes into class indices.
        # - targets: dict of (B, N) LongTensors for each attribute.
        #   e.g. targets["position_x"][b, n] = bin index of obstacle n's x at t+1.
        targets = _discretize_future(
            batch["feature_future_obstacles"],
            self.config["n_position_bins"],
            self.config["n_velocity_bins"],
            self.config["n_orientation_bins"],
            self.config["range_max"],
            self.config["speed_max"],
        )

        # 3. cross-entropy over valid obstacles only, averaged across attributes.
        mask = batch["mask_future_obstacles"]
        logits = pred["future_logits"]
        if mask.any():
            loss_aux = torch.stack(
                [
                    F.cross_entropy(logits[attr][mask], targets[attr][mask])
                    for attr in ("position_x", "position_y", "velocity", "orientation")
                ]
            ).mean()
        else:
            loss_aux = loss_wp.new_zeros(())

        # 4. weighted sum.
        total = loss_wp + self.config["aux_loss_weight"] * loss_aux

        logs = {
            "loss": total.detach(),
            "loss_wp": loss_wp.detach(),
            "loss_aux": loss_aux.detach(),
        }
        return total, logs


def _discretize_future(
    feature_future_obstacles: Tensor,
    n_position_bins: int,
    n_velocity_bins: int,
    n_orientation_bins: int,
    range_max: float,
    speed_max: float,
) -> dict[str, Tensor]:
    """Bin each future attribute into class indices (B, N) for CE loss."""

    def bin_uniform(value: Tensor, lo: float, hi: float, n_bins: int) -> Tensor:
        edges = torch.linspace(
            lo, hi, n_bins + 1, device=value.device, dtype=value.dtype
        )
        return torch.bucketize(value.contiguous(), edges[1:-1])

    speed = feature_future_obstacles[..., 0]
    x = feature_future_obstacles[..., 1]
    y = feature_future_obstacles[..., 2]
    yaw = feature_future_obstacles[..., 3]

    return {
        "position_x": bin_uniform(x, -range_max, range_max, n_position_bins),
        "position_y": bin_uniform(y, -range_max, range_max, n_position_bins),
        "velocity": bin_uniform(speed, 0.0, speed_max, n_velocity_bins),
        "orientation": bin_uniform(yaw, -math.pi, math.pi, n_orientation_bins),
    }
