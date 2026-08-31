"""Task 3 — the complete hybrid spatial-frequency deepfake detector.

Architecture:

    RGB image ─┬─> spatial branch  : ResNet-18 backbone -> linear -> s  (embed_dim)
               └─> frequency branch: DCT -> mask -> CNN  -> linear -> f  (embed_dim)

               fusion:  gated   ->  a*s + (1-a)*f,   a = sigmoid(scalar) in (0,1)
                        concat  ->  Linear([s ; f])
               head:    Linear -> ReLU -> Dropout -> Linear  ->  logits (num_classes)

The ablation switches (`use_spatial`, `use_frequency`, `use_mask`, `fusion`) live
here so Task 5's configurations are one constructor call each.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision

from .frequency_branch import FrequencyBranch

__all__ = ["SpatialFrequencyDetector"]

_RESNET18_FREEZE_GROUPS = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"]


class _SpatialBranch(nn.Module):
    """ResNet-18 feature extractor + projection to ``embed_dim``."""

    def __init__(self, embed_dim: int, pretrained: bool, freeze_until: str | None) -> None:
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = torchvision.models.resnet18(weights=weights)
        self.feature_dim = net.fc.in_features          # 512
        net.fc = nn.Identity()
        self.backbone = net
        self.proj = nn.Linear(self.feature_dim, embed_dim)

        if freeze_until is not None:
            if freeze_until not in _RESNET18_FREEZE_GROUPS:
                raise ValueError(
                    f"freeze_until must be one of {_RESNET18_FREEZE_GROUPS} or None"
                )
            cutoff = _RESNET18_FREEZE_GROUPS.index(freeze_until)
            for name in _RESNET18_FREEZE_GROUPS[: cutoff + 1]:
                for p in getattr(self.backbone, name).parameters():
                    p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.backbone(x))


class SpatialFrequencyDetector(nn.Module):
    """Hybrid detector. See module docstring.

    Args:
        image_size: square input size (px).
        num_classes: 2 for real/fake.
        embed_dim: shared feature width of both branches.
        use_spatial / use_frequency: enable each branch (at least one required).
        use_mask: include the learnable frequency mask in the frequency branch.
        fusion: ``"gated"`` (learnable scalar) or ``"concat"``. Ignored unless both
            branches are enabled.
        pretrained: load ImageNet weights for the ResNet-18 backbone.
        freeze_spatial_until: freeze backbone groups up to and including this one
            (``"conv1" | "bn1" | "layer1" | ... | "layer4"``), or ``None`` to train
            the whole backbone. Default ``"layer1"``.
        mlp_hidden / dropout: classifier head.
    """

    def __init__(
        self,
        image_size: int = 128,
        num_classes: int = 2,
        embed_dim: int = 256,
        use_spatial: bool = True,
        use_frequency: bool = True,
        use_mask: bool = True,
        fusion: str = "gated",
        pretrained: bool = True,
        freeze_spatial_until: str | None = "layer1",
        mask_channels: int | None = None,
        mlp_hidden: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if not (use_spatial or use_frequency):
            raise ValueError("at least one of use_spatial / use_frequency must be True")
        if fusion not in {"gated", "concat"}:
            raise ValueError(f"fusion must be 'gated' or 'concat', got {fusion!r}")

        self.use_spatial = use_spatial
        self.use_frequency = use_frequency
        self.both = use_spatial and use_frequency
        self.fusion = fusion if self.both else "single"
        self.embed_dim = embed_dim

        self.spatial = (
            _SpatialBranch(embed_dim, pretrained, freeze_spatial_until) if use_spatial else None
        )
        self.frequency = (
            FrequencyBranch(image_size, embed_dim, use_mask=use_mask, mask_channels=mask_channels)
            if use_frequency
            else None
        )

        if self.fusion == "gated":
            # sigmoid(0) = 0.5 -> equal weighting at init
            self.alpha_logit = nn.Parameter(torch.zeros(()))
            fused_dim = embed_dim
        elif self.fusion == "concat":
            self.fuse_proj = nn.Linear(2 * embed_dim, embed_dim)
            fused_dim = embed_dim
        else:  # single branch
            fused_dim = embed_dim

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, num_classes),
        )

    # -- fusion ----------------------------------------------------------------
    @property
    def alpha(self) -> torch.Tensor | None:
        """Spatial-branch weight in (0, 1), or ``None`` when not using gated fusion."""
        if self.fusion != "gated":
            return None
        return torch.sigmoid(self.alpha_logit)

    def _fuse(self, s: torch.Tensor | None, f: torch.Tensor | None) -> torch.Tensor:
        if self.fusion == "single":
            return s if s is not None else f
        if self.fusion == "gated":
            a = torch.sigmoid(self.alpha_logit)
            return a * s + (1.0 - a) * f
        return self.fuse_proj(torch.cat([s, f], dim=1))

    # -- forward -------------------------------------------------------------- -
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Fused feature vector ``(B, embed_dim)`` — for t-SNE / UMAP (Task 8)."""
        s = self.spatial(x) if self.spatial is not None else None
        f = self.frequency(x) if self.frequency is not None else None
        return self._fuse(s, f)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))

    # -- introspection ------------------------------------------------------- -
    @torch.no_grad()
    def get_alpha(self) -> float | None:
        a = self.alpha
        return None if a is None else a.item()

    @torch.no_grad()
    def get_frequency_mask(self) -> torch.Tensor | None:
        if self.frequency is None or self.frequency.mask is None:
            return None
        return self.frequency.mask.get_mask()

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    def extra_repr(self) -> str:
        return (
            f"use_spatial={self.use_spatial}, use_frequency={self.use_frequency}, "
            f"fusion={self.fusion}"
        )
