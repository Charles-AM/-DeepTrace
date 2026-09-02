"""Validation for the perturbation battery and robustness harness (Task 12)."""

import numpy as np
import pytest
from PIL import Image

from src.transforms_perturb import PERTURBATIONS, SEVERITIES, identity, make_perturbation


@pytest.fixture
def img():
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, (80, 80, 3), dtype=np.uint8))


def test_identity_is_noop(img):
    assert np.array_equal(np.asarray(identity(img)), np.asarray(img.convert("RGB")))
    assert np.array_equal(np.asarray(make_perturbation("blur", 0)(img)), np.asarray(img.convert("RGB")))


@pytest.mark.parametrize("name", list(PERTURBATIONS))
@pytest.mark.parametrize("sev", SEVERITIES)
def test_perturbation_contract(img, name, sev):
    out = make_perturbation(name, sev)(img)
    assert isinstance(out, Image.Image)
    assert out.size == img.size
    assert out.mode == "RGB"
    assert not np.array_equal(np.asarray(out), np.asarray(img.convert("RGB")))


@pytest.mark.parametrize("name", ["jpeg", "blur", "resize", "contrast"])
def test_severity_is_monotone(name):
    # higher severity -> further from the clean image. Use a structured image
    # (gradient + block) rather than pure noise; noise is stochastic, skip it.
    yy, xx = np.mgrid[0:96, 0:96]
    base = ((xx * 2) % 256).astype(np.uint8)
    struct = np.stack([base, np.roll(base, 20, 0), 255 - base], axis=-1)
    struct[20:60, 20:60] = 200
    src = Image.fromarray(struct)
    ref = np.asarray(src, dtype=np.float32)
    dists = [
        np.abs(np.asarray(make_perturbation(name, s)(src), dtype=np.float32) - ref).mean()
        for s in SEVERITIES
    ]
    assert dists == sorted(dists), f"{name} severity not monotone: {dists}"


def test_unknown_perturbation_and_bad_severity(img):
    with pytest.raises(KeyError):
        make_perturbation("sepia", 1)
    with pytest.raises(ValueError):
        make_perturbation("blur", 9)


def test_eval_transform_shape():
    from src.robustness import _eval_transform

    t = _eval_transform(64, "jpeg", 3)
    x = t(Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8)))
    assert tuple(x.shape) == (3, 64, 64)
