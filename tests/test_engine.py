"""Validation for the train/eval loops (Task 4).

Uses a tiny in-memory dataset with a real (if trivial) signal so we can check
that a training step actually reduces the loss.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.engine import evaluate, predict_scores, train_one_epoch
from src.losses import FocalLoss
from src.models import SpatialFrequencyDetector
from src.utils import seed_everything

SIZE = 32


class ToyFaces(Dataset):
    """Fake = bright-ish images, real = dark-ish. Separable but noisy."""

    def __init__(self, n=48, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.x, self.y = [], []
        for i in range(n):
            label = i % 2
            base = 0.6 if label == 1 else -0.6
            self.x.append(base + 0.4 * torch.randn(3, SIZE, SIZE, generator=g))
            self.y.append(label)
        self.x = torch.stack(self.x)
        self.y = torch.tensor(self.y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], int(self.y[i])


def _loader(n=48, bs=8, seed=0):
    return DataLoader(ToyFaces(n, seed), batch_size=bs, shuffle=True)


def test_train_one_epoch_returns_finite_loss():
    seed_everything(0)
    model = SpatialFrequencyDetector(image_size=SIZE, pretrained=False)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = train_one_epoch(model, _loader(), opt, FocalLoss(gamma=2.0), torch.device("cpu"))
    assert np.isfinite(loss)


def test_evaluate_returns_metrics_and_arrays():
    model = SpatialFrequencyDetector(image_size=SIZE, pretrained=False)
    metrics, y_true, y_score = evaluate(model, _loader(), torch.device("cpu"), FocalLoss())
    for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "eer", "loss"):
        assert k in metrics
    assert y_true.shape == y_score.shape == (48,)
    assert ((y_score >= 0) & (y_score <= 1)).all()


def test_predict_scores_shapes():
    model = SpatialFrequencyDetector(image_size=SIZE, pretrained=False)
    y_true, y_score = predict_scores(model, _loader(n=24), torch.device("cpu"))
    assert y_true.shape == (24,) and y_score.shape == (24,)


def test_training_reduces_loss_and_learns_signal():
    seed_everything(0)
    device = torch.device("cpu")
    model = SpatialFrequencyDetector(image_size=SIZE, pretrained=False, freeze_spatial_until=None)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    crit = FocalLoss(gamma=2.0)
    train, test = _loader(seed=1), _loader(seed=2)

    first = train_one_epoch(model, train, opt, crit, device)
    for _ in range(6):
        last = train_one_epoch(model, train, opt, crit, device)

    assert last < first
    acc = evaluate(model, test, device)[0]["accuracy"]
    assert acc > 0.75          # trivial signal -> should be well above chance


def test_amp_scaler_path_is_cpu_safe():
    # scaler=None on CPU is the normal path; just make sure autocast wrapper runs
    model = SpatialFrequencyDetector(image_size=SIZE, pretrained=False)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    out = train_one_epoch(model, _loader(n=16), opt, FocalLoss(), torch.device("cpu"), scaler=None)
    assert np.isfinite(out)
