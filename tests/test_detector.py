"""Validation for the hybrid detector, focal loss, and ablation configs (Task 3).

All tests use ``pretrained=False`` to stay offline and fast.
"""

import pytest
import torch

from src.config import MODEL_CONFIGS, build_model
from src.losses import FocalLoss
from src.models import SpatialFrequencyDetector

SIZE = 64


def make(**kw) -> SpatialFrequencyDetector:
    return SpatialFrequencyDetector(image_size=SIZE, pretrained=False, **kw)


def test_forward_shape():
    model = make()
    out = model(torch.randn(3, 3, SIZE, SIZE))
    assert out.shape == (3, 2)


@pytest.mark.parametrize("name", list(MODEL_CONFIGS))
def test_every_config_builds_and_runs(name):
    model = build_model(name, image_size=SIZE, pretrained=False)
    model.train()
    x = torch.randn(2, 3, SIZE, SIZE)
    logits = model(x)
    assert logits.shape == (2, 2)
    assert torch.isfinite(logits).all()
    logits.sum().backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)


def test_alpha_bounded_and_learnable():
    model = make(fusion="gated")
    a = model.get_alpha()
    assert 0.0 < a < 1.0
    x = torch.randn(2, 3, SIZE, SIZE)
    model(x).sum().backward()
    assert model.alpha_logit.grad is not None and model.alpha_logit.grad != 0


def test_concat_fusion_has_no_alpha():
    model = make(fusion="concat")
    assert model.get_alpha() is None
    assert model(torch.randn(2, 3, SIZE, SIZE)).shape == (2, 2)


def test_single_branch_ignores_fusion():
    for kw in (dict(use_frequency=False), dict(use_spatial=False)):
        model = make(**kw)
        assert model.fusion == "single"
        assert model(torch.randn(2, 3, SIZE, SIZE)).shape == (2, 2)


def test_requires_both_branches():
    with pytest.raises(ValueError):
        make(use_spatial=False, use_frequency=False)


def test_backbone_freezing():
    model = make(freeze_spatial_until="layer1")
    frozen = {n for n, p in model.spatial.backbone.named_parameters() if not p.requires_grad}
    assert any(n.startswith("conv1") for n in frozen)
    assert any(n.startswith("layer1") for n in frozen)
    assert not any(n.startswith("layer4") for n in frozen)

    model_full = make(freeze_spatial_until=None)
    assert all(p.requires_grad for p in model_full.spatial.backbone.parameters())


def test_no_mask_config_drops_the_mask():
    assert build_model("no_mask", image_size=SIZE, pretrained=False).frequency.mask is None
    assert build_model("full", image_size=SIZE, pretrained=False).frequency.mask is not None


def test_frequency_mask_receives_gradient_in_full_model():
    model = build_model("full", image_size=SIZE, pretrained=False)
    model(torch.randn(2, 3, SIZE, SIZE)).sum().backward()
    assert model.frequency.mask.logits.grad is not None
    assert model.frequency.mask.logits.grad.abs().sum() > 0


def test_forward_features_shape_and_mask_getter():
    model = make(embed_dim=128)
    feats = model.forward_features(torch.randn(4, 3, SIZE, SIZE))
    assert feats.shape == (4, 128)
    assert model.get_frequency_mask().shape == (1, SIZE, SIZE)


# --- focal loss -------------------------------------------------------------- -

def test_focal_reduces_to_cross_entropy_at_gamma0():
    torch.manual_seed(0)
    logits = torch.randn(16, 2)
    target = torch.randint(0, 2, (16,))
    fl = FocalLoss(gamma=0.0)(logits, target)
    ce = torch.nn.functional.cross_entropy(logits, target)
    assert torch.allclose(fl, ce, atol=1e-6)


def test_focal_downweights_easy_examples():
    # one confident-correct sample, one uncertain sample
    logits = torch.tensor([[5.0, -5.0], [0.2, 0.0]])
    target = torch.tensor([0, 0])
    ce = torch.nn.functional.cross_entropy(logits, target, reduction="none")
    fl = FocalLoss(gamma=2.0, reduction="none")(logits, target)
    easy_ratio = (fl[0] / ce[0]).item()
    hard_ratio = (fl[1] / ce[1]).item()
    assert easy_ratio < hard_ratio
    assert easy_ratio < 0.01          # (1 - ~0.9999)^2


def test_focal_scalar_alpha_expands_to_two_classes():
    loss = FocalLoss(gamma=2.0, alpha=0.75)
    assert torch.allclose(loss.weight, torch.tensor([0.25, 0.75]))
    out = loss(torch.randn(8, 2), torch.randint(0, 2, (8,)))
    assert out.ndim == 0 and torch.isfinite(out)


def test_focal_gradient_flows():
    logits = torch.randn(8, 2, requires_grad=True)
    FocalLoss(gamma=2.0)(logits, torch.randint(0, 2, (8,))).backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


# --- timm baselines (verification framework) ------------------------------- -

@pytest.mark.parametrize("name", ["efficientnet_b0", "xception"])
def test_timm_baseline_builds_and_matches_interface(name):
    model = build_model(name, image_size=128, pretrained=False)
    out = model(torch.randn(2, 3, 128, 128))
    assert out.shape == (2, 2)
    assert model.get_alpha() is None
    assert model.get_frequency_mask() is None
    out.sum().backward()


def test_build_model_rejects_unknown_name():
    with pytest.raises(KeyError):
        build_model("not_a_real_config")


# --- focal loss ------------------------------------------------------------- -

def test_focal_rejects_bad_inputs():
    with pytest.raises(ValueError):
        FocalLoss(reduction="banana")
    with pytest.raises(ValueError):
        FocalLoss(alpha=1.5)
    with pytest.raises(ValueError):
        FocalLoss()(torch.randn(4, 2, 2), torch.zeros(4, dtype=torch.long))
