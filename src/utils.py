"""Shared helpers: seeding, device selection, running averages."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

__all__ = [
    "seed_everything", "get_device", "AverageMeter", "count_parameters",
    "no_decay_param_groups",
]


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and Torch RNGs for reproducible runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(prefer: str | None = None) -> torch.device:
    """Pick the best available device: explicit `prefer`, else CUDA -> MPS -> CPU."""
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class AverageMeter:
    """Tracks a running mean (e.g. loss over an epoch)."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


def count_parameters(module: torch.nn.Module, trainable_only: bool = True) -> int:
    return sum(
        p.numel() for p in module.parameters() if p.requires_grad or not trainable_only
    )


def no_decay_param_groups(module: torch.nn.Module, weight_decay: float) -> list[dict]:
    """Split parameters into a decayed group (conv/linear weights) and a
    zero-weight-decay group (biases, norm params, and any 1-D/scalar parameter such
    as a learnable fusion gate). AdamW's decoupled weight decay otherwise pulls a
    scalar gate toward 0 every step regardless of its task gradient — a real bug we
    hit (``alpha_logit`` sitting at exactly sigmoid(0)=0.5 across every seed/config
    turned out to be consistent with weight decay dominating a weak task signal,
    not necessarily the model "choosing" not to use the frequency branch)."""
    decay, no_decay = [], []
    for p in module.parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 else decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
