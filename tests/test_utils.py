"""Regression test for the weight-decay confound found 2026-09-04: a scalar gate
(e.g. the gated-fusion ``alpha_logit``) must not get decoupled weight decay, or
AdamW pulls it toward 0 every step regardless of its task gradient.
"""

import torch

from src.models import SpatialFrequencyDetector
from src.utils import no_decay_param_groups


def test_scalar_and_bias_params_excluded_from_decay():
    model = SpatialFrequencyDetector(image_size=32, use_spatial=True, use_frequency=True,
                                      use_mask=True, fusion="gated", pretrained=False)
    groups = no_decay_param_groups(model, weight_decay=0.05)
    decay_params = set(id(p) for p in groups[0]["params"])
    no_decay_params = set(id(p) for p in groups[1]["params"])

    assert groups[0]["weight_decay"] == 0.05
    assert groups[1]["weight_decay"] == 0.0
    assert id(model.alpha_logit) in no_decay_params
    assert id(model.alpha_logit) not in decay_params

    # every >=2D weight lands in the decayed group, every bias/norm/scalar in the other
    for p in model.parameters():
        if not p.requires_grad:
            continue
        assert id(p) in (no_decay_params if p.ndim <= 1 else decay_params)


def test_no_decay_groups_cover_every_trainable_param_exactly_once():
    model = SpatialFrequencyDetector(image_size=32, use_spatial=True, use_frequency=True,
                                      use_mask=True, fusion="gated", pretrained=False)
    groups = no_decay_param_groups(model, weight_decay=0.05)
    all_ids = [id(p) for g in groups for p in g["params"]]
    trainable_ids = [id(p) for p in model.parameters() if p.requires_grad]
    assert sorted(all_ids) == sorted(trainable_ids)


def test_optimizer_actually_skips_decay_on_the_gate():
    model = SpatialFrequencyDetector(image_size=32, use_spatial=True, use_frequency=True,
                                      use_mask=True, fusion="gated", pretrained=False)
    opt = torch.optim.AdamW(no_decay_param_groups(model, weight_decay=0.05), lr=0.1)

    with torch.no_grad():
        model.alpha_logit.zero_()
    # zero gradient everywhere: pure weight-decay step should NOT move alpha_logit
    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.zeros_like(p)
    opt.step()
    assert model.alpha_logit.item() == 0.0
