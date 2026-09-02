"""Named model configurations for the ablation / baseline matrix (Task 5).

Each entry is kwargs for ``SpatialFrequencyDetector``. Keeps every experiment a
single lookup so training runs are reproducible and comparable.
"""

from __future__ import annotations

MODEL_CONFIGS: dict[str, dict] = {
    # proposed method (architecture only; SAS is a training-time flag, see src/train.py)
    "full": dict(use_spatial=True, use_frequency=True, use_mask=True, fusion="gated"),
    # proposed method + frequency-band dropout regulariser (Task 9)
    "full_banddrop": dict(
        use_spatial=True, use_frequency=True, use_mask=True, fusion="gated", band_dropout_p=0.5
    ),
    # ablations
    "no_mask": dict(use_spatial=True, use_frequency=True, use_mask=False, fusion="gated"),
    "no_dct": dict(use_spatial=True, use_frequency=False, use_mask=False, fusion="gated"),
    "no_spatial": dict(use_spatial=False, use_frequency=True, use_mask=True, fusion="gated"),
    "no_fusion_concat": dict(use_spatial=True, use_frequency=True, use_mask=True, fusion="concat"),
    "no_banddrop": dict(
        use_spatial=True, use_frequency=True, use_mask=True, fusion="gated", band_dropout_p=0.0
    ),
    # Task 5 headline baseline (spatial CNN only) == no_dct, kept under a clear name
    "baseline_spatial": dict(use_spatial=True, use_frequency=False, use_mask=False, fusion="gated"),
    "frequency_only": dict(use_spatial=False, use_frequency=True, use_mask=True, fusion="gated"),
}


def build_model(name: str, image_size: int = 128, pretrained: bool = True, **overrides):
    """Instantiate a named config.

    ``name`` is either a hybrid-model config from ``MODEL_CONFIGS`` or an
    established baseline from ``TIMM_BASELINES`` (``xception``, ``efficientnet_b0``,
    ``efficientnet_b4``, ``efficientnet_b7``).
    """
    from .models.baselines import TIMM_BASELINES

    if name in MODEL_CONFIGS:
        from .models import SpatialFrequencyDetector

        cfg = {**MODEL_CONFIGS[name], "image_size": image_size, "pretrained": pretrained, **overrides}
        return SpatialFrequencyDetector(**cfg)

    if name in TIMM_BASELINES:
        from .models.baselines import TimmClassifier

        return TimmClassifier(
            TIMM_BASELINES[name], pretrained=pretrained, image_size=image_size, **overrides
        )

    raise KeyError(
        f"unknown config {name!r}; choices: {sorted(MODEL_CONFIGS) + sorted(TIMM_BASELINES)}"
    )
