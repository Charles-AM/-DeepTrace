"""Named model configurations for the ablation / baseline matrix (Task 5).

Each entry is kwargs for ``SpatialFrequencyDetector``. Keeps every experiment a
single lookup so training runs are reproducible and comparable.
"""

from __future__ import annotations

MODEL_CONFIGS: dict[str, dict] = {
    # proposed method
    "full": dict(use_spatial=True, use_frequency=True, use_mask=True, fusion="gated"),
    # ablations
    "no_mask": dict(use_spatial=True, use_frequency=True, use_mask=False, fusion="gated"),
    "no_dct": dict(use_spatial=True, use_frequency=False, use_mask=False, fusion="gated"),
    "no_spatial": dict(use_spatial=False, use_frequency=True, use_mask=True, fusion="gated"),
    "no_fusion_concat": dict(use_spatial=True, use_frequency=True, use_mask=True, fusion="concat"),
    # Task 5 headline baseline (spatial CNN only) == no_dct, kept under a clear name
    "baseline_spatial": dict(use_spatial=True, use_frequency=False, use_mask=False, fusion="gated"),
    "frequency_only": dict(use_spatial=False, use_frequency=True, use_mask=True, fusion="gated"),
}


def build_model(name: str, **overrides):
    """Instantiate a named config. ``overrides`` (e.g. ``image_size``, ``pretrained``)
    are merged on top."""
    from .models import SpatialFrequencyDetector

    if name not in MODEL_CONFIGS:
        raise KeyError(f"unknown config {name!r}; choices: {sorted(MODEL_CONFIGS)}")
    return SpatialFrequencyDetector(**{**MODEL_CONFIGS[name], **overrides})
