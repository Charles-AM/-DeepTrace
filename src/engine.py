"""Train / evaluate loops (Task 4). Reused unchanged by Tasks 5-7."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from .metrics import compute_metrics
from .utils import AverageMeter

__all__ = ["train_one_epoch", "evaluate", "predict_scores"]


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    grad_clip: float | None = 1.0,
    scaler: "torch.cuda.amp.GradScaler | None" = None,
    log_every: int = 50,
    writer=None,
    epoch: int = 0,
) -> float:
    model.train()
    meter = AverageMeter()
    use_amp = scaler is not None

    for step, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)

        if use_amp:
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        meter.update(loss.item(), x.size(0))
        if writer is not None and step % log_every == 0:
            gstep = epoch * len(loader) + step
            writer.add_scalar("train/loss_step", loss.item(), gstep)

    return meter.avg


@torch.no_grad()
def predict_scores(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(y_true, y_score)`` where score = P(fake)."""
    model.eval()
    scores, targets = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        prob_fake = logits.softmax(dim=1)[:, 1]
        scores.append(prob_fake.float().cpu())
        targets.append(y.clone())
    return torch.cat(targets).numpy(), torch.cat(scores).numpy()


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: torch.nn.Module | None = None,
    threshold: float = 0.5,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Return ``(metrics_dict, y_true, y_score)``."""
    model.eval()
    scores, targets, losses = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        if criterion is not None:
            losses.append(criterion(logits, y).item())
        scores.append(logits.softmax(dim=1)[:, 1].float().cpu())
        targets.append(y.cpu())

    y_true = torch.cat(targets).numpy()
    y_score = torch.cat(scores).numpy()
    metrics = compute_metrics(y_true, y_score, threshold=threshold)
    if losses:
        metrics["loss"] = float(np.mean(losses))
    return metrics, y_true, y_score
