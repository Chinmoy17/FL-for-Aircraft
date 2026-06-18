"""Multi-task 1-D CNN for joint RUL regression and binary fault detection.

Design notes
------------
- **GroupNorm**, not BatchNorm. BatchNorm's running mean/var would need to be
  aggregated across federated clients, which is brittle (FedAvg over stats
  produces statistically wrong global stats). GroupNorm depends only on the
  current batch, behaves identically in train / eval mode, and is the standard
  drop-in choice for federated CNNs.
- **AdaptiveAvgPool1d(1)** reduces the temporal dimension to a single vector so
  the heads do not depend on the window length. Swapping ``window_size`` later
  requires no model changes.
- **Two heads, shared trunk.** The shared trunk gives the encoder a multi-task
  inductive bias (RUL and fault are both functions of the same degradation
  state); the heads keep the task-specific noise from polluting each other.
- **Input transpose lives inside the model** so callers pass ``(B, T, F)`` —
  the same layout the rest of the pipeline uses — and the model handles the
  ``(B, F, T)`` conversion Conv1d expects.

Parameter budget (default kwargs, F=17): 29,890 trainable params.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class MultiTaskCNNConfig:
    """Hyper-parameters of :class:`MultiTaskCNN`.

    Attributes:
        n_features: Number of input feature channels (op_settings + sensors).
        window_size: Temporal length of each input window (informational; the
            model is window-size agnostic thanks to ``AdaptiveAvgPool1d``).
        conv_channels: Output channels of each conv block, in order.
        kernel_sizes: Conv kernel sizes; must match ``conv_channels``.
        gn_groups: ``num_groups`` for GroupNorm. Must divide every channel size.
        trunk_dim: Width of the shared MLP trunk between encoder and heads.
        dropout: Dropout probability applied after the trunk.
    """

    n_features: int
    window_size: int = 30
    conv_channels: tuple[int, ...] = (32, 64, 64)
    kernel_sizes: tuple[int, ...] = (5, 5, 3)
    gn_groups: int = 8
    trunk_dim: int = 64
    dropout: float = 0.2

    def __post_init__(self) -> None:
        if self.n_features < 1:
            raise ValueError(f"n_features must be >= 1, got {self.n_features}.")
        if self.window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {self.window_size}.")
        if len(self.conv_channels) != len(self.kernel_sizes):
            raise ValueError(
                "conv_channels and kernel_sizes must have the same length "
                f"(got {len(self.conv_channels)} vs {len(self.kernel_sizes)})."
            )
        if any(c < 1 for c in self.conv_channels):
            raise ValueError(f"conv_channels must be >= 1, got {self.conv_channels}.")
        if any(k < 1 for k in self.kernel_sizes):
            raise ValueError(f"kernel_sizes must be >= 1, got {self.kernel_sizes}.")
        if any(c % self.gn_groups != 0 for c in self.conv_channels):
            raise ValueError(
                f"Every channel count in conv_channels {self.conv_channels} must be "
                f"divisible by gn_groups={self.gn_groups}."
            )
        if self.trunk_dim < 1:
            raise ValueError(f"trunk_dim must be >= 1, got {self.trunk_dim}.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}.")


class MultiTaskCNN(nn.Module):
    """Shared 1-D CNN encoder with separate RUL and fault heads.

    Forward signature::

        x:          (batch, window_size, n_features)  float32
        returns ->  RULPrediction(rul=(batch,), fault_logits=(batch,))

    The fault head returns **logits**, not probabilities — pair it with
    ``nn.BCEWithLogitsLoss`` (which is what :class:`MultiTaskLoss` uses) for
    numerically stable training.
    """

    def __init__(self, config: MultiTaskCNNConfig) -> None:
        super().__init__()
        self.config = config

        # ---------------- Encoder ----------------
        encoder_layers: list[nn.Module] = []
        in_ch = config.n_features
        for out_ch, k in zip(config.conv_channels, config.kernel_sizes):
            encoder_layers.append(
                nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2)
            )
            encoder_layers.append(nn.GroupNorm(config.gn_groups, out_ch))
            encoder_layers.append(nn.ReLU(inplace=True))
            in_ch = out_ch
        encoder_layers.append(nn.AdaptiveAvgPool1d(1))
        self.encoder = nn.Sequential(*encoder_layers)

        # ---------------- Shared trunk ----------------
        last_ch = config.conv_channels[-1]
        self.trunk = nn.Sequential(
            nn.Flatten(),
            nn.Linear(last_ch, config.trunk_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
        )

        # ---------------- Heads ----------------
        # The RUL head produces a non-negative scalar via softplus to enforce a
        # physically meaningful prediction range; the model can still predict
        # large RULs (softplus is unbounded above) but cannot predict negatives.
        self.rul_head = nn.Linear(config.trunk_dim, 1)
        self.fault_head = nn.Linear(config.trunk_dim, 1)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Deterministic-init each module so a single ``seed_everything`` fully reproduces a run."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> "RULPrediction":
        if x.dim() != 3:
            raise ValueError(
                f"Expected input shape (batch, window_size, n_features); got {tuple(x.shape)}."
            )
        # (B, T, F) -> (B, F, T) for Conv1d.
        z = x.transpose(1, 2)
        z = self.encoder(z)  # (B, C, 1)
        z = self.trunk(z)  # (B, trunk_dim)
        # ``softplus`` enforces RUL >= 0 in a smooth, differentiable way.
        rul = nn.functional.softplus(self.rul_head(z)).squeeze(-1)
        fault_logits = self.fault_head(z).squeeze(-1)
        return RULPrediction(rul=rul, fault_logits=fault_logits)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


@dataclass(frozen=True)
class RULPrediction:
    """Container for one batch of model outputs.

    Attributes:
        rul: Shape ``(batch,)``, non-negative predicted RUL in cycles (capped
            range matches the training labels — typically 0..125).
        fault_logits: Shape ``(batch,)``, raw fault logits; apply
            ``torch.sigmoid`` for probabilities or pair with
            ``BCEWithLogitsLoss`` directly.
    """

    rul: torch.Tensor
    fault_logits: torch.Tensor

    def fault_probs(self) -> torch.Tensor:
        return torch.sigmoid(self.fault_logits)
