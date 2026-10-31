"""NPR-inspired local pixel residual, with an Xception classifier.

Adapted from Tan et al., "Rethinking the Up-Sampling Operations in CNN-based
Generative Network for Generalizable Deepfake Detection", CVPR 2024. Verified
against `networks/resnet.py` in chuangchuangtan/NPR-DeepfakeDetection:

    def interpolate(self, img, factor):
        return F.interpolate(F.interpolate(img, scale_factor=factor,
            mode='nearest', recompute_scale_factor=True),
            scale_factor=1/factor, mode='nearest', recompute_scale_factor=True)

    NPR = x - self.interpolate(x, 0.5)
    x = self.conv1(NPR*2.0/3.0)

⚠️ **This is an adaptation, not a reproduction.** The original trains a ~1.44M
parameter custom CNN on ForenSynths with Adam at batch 32. We keep only the
transform and hand it to the same Xception backbone the rest of this project uses.
NPR reports **no matched FF++ comparison** of Xception against Xception+NPR, so
there is no published reference effect here — unlike FAD's +0.014.

🚨 **Placement matters.** The reference computes the residual *inside* the model's
forward, so **after** dataloader normalisation. Applying it to unnormalised pixels
gives a per-channel scaled version, not the same operation. This module is called
on the already-normalised tensor, matching that.

Properties, all verified in `tests/test_npr.py`:
  - zero learned parameters
  - output shape and channel count unchanged (3 channels in, 3 out)
  - no batch-structure requirement — each image is handled independently, unlike
    UCF, whose reconstruction loss needs balanced ordered batches
  - therefore the backbone needs no modification, not even `in_chans`
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

SCALE = 2.0 / 3.0


def npr_residual(x: torch.Tensor, factor: float = 0.5) -> torch.Tensor:
    """x minus its nearest-neighbour down/up round trip, scaled. Parameter-free.

    `recompute_scale_factor=True` reproduces the reference exactly; without it
    torch resolves the output size differently and the round trip can return a
    tensor whose shape does not match `x`.
    """
    down = F.interpolate(x, scale_factor=factor, mode="nearest",
                         recompute_scale_factor=True)
    up = F.interpolate(down, scale_factor=1.0 / factor, mode="nearest",
                       recompute_scale_factor=True)
    return (x - up) * SCALE


class NPRModule(nn.Module):
    """Stateless wrapper so the transform is visible in the module tree."""

    def __init__(self, factor: float = 0.5) -> None:
        super().__init__()
        self.factor = factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return npr_residual(x, self.factor)

    def extra_repr(self) -> str:
        return f"factor={self.factor}, scale={SCALE:.4f}, params=0"


class XceptionNPR(nn.Module):
    """NPR residual + Xception classifier.

    Deliberately mirrors `F3NetFAD` so the two supplementary arms differ only in
    which transform precedes an otherwise identical backbone.
    """

    def __init__(
        self, image_size: int = 128, num_classes: int = 2, pretrained: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        import timm

        self.image_size = image_size
        self.npr = NPRModule()
        self.backbone = timm.create_model(
            "legacy_xception", pretrained=pretrained, num_classes=num_classes,
            in_chans=3, drop_rate=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.npr(x))

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        f = self.backbone.forward_features(self.npr(x))
        return f.mean(dim=(-2, -1)) if f.dim() == 4 else f
