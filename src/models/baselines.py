"""Established baselines for the verification framework (XceptionNet, EfficientNet).

Wrapped in the same call/interface as ``SpatialFrequencyDetector`` so the exact
same training and evaluation code (``src/train.py``, ``src/engine.py``) runs them
on identical splits — the fair comparison the verification plan requires.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["TimmClassifier", "TIMM_BASELINES"]

# friendly name -> timm model id
TIMM_BASELINES = {
    "xception": "legacy_xception",
    "efficientnet_b0": "efficientnet_b0",
    "efficientnet_b4": "efficientnet_b4",
    "efficientnet_b7": "tf_efficientnet_b7",
}


class TimmClassifier(nn.Module):
    """A pretrained timm backbone with a fresh ``num_classes`` head."""

    def __init__(
        self,
        backbone: str,
        num_classes: int = 2,
        pretrained: bool = True,
        image_size: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        import timm

        self.backbone_name = backbone
        self.net = timm.create_model(
            backbone, pretrained=pretrained, num_classes=num_classes, drop_rate=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.net.forward_features(x).mean(dim=(-2, -1)) if x.dim() == 4 else self.net.forward_features(x)

    # -- shims so the shared training loop treats it like the hybrid model ----
    @torch.no_grad()
    def get_alpha(self):
        return None

    @torch.no_grad()
    def get_frequency_mask(self):
        return None
