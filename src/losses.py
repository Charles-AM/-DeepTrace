"""Focal loss (Task 3).

Focal loss down-weights easy examples so training concentrates on the hard,
borderline cases::

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

At ``gamma=0`` (and uniform alpha) this is exactly cross-entropy.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["FocalLoss"]


class FocalLoss(nn.Module):
    """Multi-class focal loss over raw logits.

    Args:
        gamma: focusing parameter. ``0`` -> cross-entropy. Paper default ``2``.
        alpha: class weighting.
            * ``None`` -> no weighting.
            * ``float`` -> weight for the positive class (index 1); the negative
              class gets ``1 - alpha``. Convenient for binary real/fake.
            * 1-D tensor / sequence -> explicit per-class weights.
        reduction: ``"mean"`` | ``"sum"`` | ``"none"``.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float | torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"invalid reduction: {reduction}")
        self.gamma = float(gamma)
        self.reduction = reduction

        if alpha is None:
            weight = None
        elif isinstance(alpha, (float, int)):
            if not 0.0 < float(alpha) < 1.0:
                raise ValueError("scalar alpha must be in (0, 1)")
            weight = torch.tensor([1.0 - float(alpha), float(alpha)])
        else:
            weight = torch.as_tensor(alpha, dtype=torch.float32)
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.dim() != 2:
            raise ValueError(f"expected (B, num_classes) logits, got shape {tuple(logits.shape)}")
        log_p = F.log_softmax(logits, dim=1)
        log_pt = log_p.gather(1, target.unsqueeze(1)).squeeze(1)   # log prob of true class
        pt = log_pt.exp()
        loss = -((1.0 - pt) ** self.gamma) * log_pt                # unweighted focal term

        if self.weight is not None:
            at = self.weight.to(logits.device).gather(0, target)
            loss = at * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
