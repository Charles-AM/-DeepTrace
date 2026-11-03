"""`src/band_ablation.py` — the repository's last uncovered module.

Written 2026-11-03 while closing gap-register item G3. The script had never been
executed and had no tests, so these cover the two pure pieces that can be checked
without a checkpoint: the radial band partition and the DCT zeroing round trip.

One of them pins a defect rather than a guarantee — see
`test_zeroing_the_lowest_band_clips_half_the_pixels`.
"""

import numpy as np
import pytest
from PIL import Image
from scipy.fft import dctn, idctn

from src.band_ablation import ZeroDCTBand, _transform, radial_band_mask

SIZE, N_BANDS = 32, 8


@pytest.fixture
def noise_rgb():
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 256, (SIZE, SIZE, 3), dtype=np.uint8))


def _band_energy(img, mask):
    a = np.asarray(img.convert("RGB"), dtype=np.float64)
    return sum(float((dctn(a[..., c], type=2, norm="ortho")[mask] ** 2).sum())
               for c in range(a.shape[2]))


# --- the radial partition ----------------------------------------------------

def test_bands_partition_the_coefficient_grid():
    """Every DCT coefficient belongs to exactly one band — no gaps, no overlap.

    The last band uses a closed upper edge (`r <= hi`) precisely so the corner,
    whose normalised radius is exactly 1.0, is not dropped.
    """
    stack = np.stack([radial_band_mask(SIZE, b, N_BANDS) for b in range(N_BANDS)])
    assert (stack.sum(axis=0) == 1).all()
    assert int(stack.sum()) == SIZE * SIZE


def test_dc_is_in_the_lowest_band_and_the_corner_in_the_highest():
    """Corner-origin convention: DC sits at [0, 0], not at the centre."""
    assert radial_band_mask(SIZE, 0, N_BANDS)[0, 0]
    assert radial_band_mask(SIZE, N_BANDS - 1, N_BANDS)[SIZE - 1, SIZE - 1]


def test_the_corner_radius_normalises_to_exactly_one():
    """If the normalisation drifted, the outermost coefficients would fall out."""
    u, v = np.indices((SIZE, SIZE))
    r = np.sqrt(u.astype(np.float64) ** 2 + v.astype(np.float64) ** 2)
    r /= np.sqrt(2.0) * (SIZE - 1)
    assert r[SIZE - 1, SIZE - 1] == pytest.approx(1.0, abs=1e-12)
    assert r.max() == pytest.approx(1.0, abs=1e-12)


def test_band_count_is_honoured():
    for n in (2, 4, 16):
        stack = np.stack([radial_band_mask(SIZE, b, n) for b in range(n)])
        assert (stack.sum(axis=0) == 1).all()


# --- the DCT round trip ------------------------------------------------------

def test_mask_none_is_the_identity_on_rgb(noise_rgb):
    out = ZeroDCTBand(None)(noise_rgb)
    assert np.array_equal(np.asarray(out), np.asarray(noise_rgb.convert("RGB")))


def test_output_preserves_size_and_mode(noise_rgb):
    out = ZeroDCTBand(radial_band_mask(SIZE, 3, N_BANDS))(noise_rgb)
    assert out.size == noise_rgb.size and out.mode == "RGB"


@pytest.mark.parametrize("band", range(1, N_BANDS))
def test_zeroing_a_band_removes_its_energy(band, noise_rgb):
    """Above the DC band, >99% of the band's energy is gone after the round trip.

    It is not exactly zero because the result is clipped to [0, 255] and cast to
    uint8, which puts a little broadband energy back.
    """
    mask = radial_band_mask(SIZE, band, N_BANDS)
    before = _band_energy(noise_rgb, mask)
    after = _band_energy(ZeroDCTBand(mask)(noise_rgb), mask)
    assert after / before < 0.01


def test_zeroing_the_lowest_band_clips_half_the_pixels():
    """⚠️ PINS A DEFECT, not a guarantee.

    Band 0 contains the DC coefficient, which carries the mean. Zeroing it centres
    the image on zero, so roughly half of every image's pixels go negative and are
    clamped to 0 by `np.clip(a, 0, 255)`. This is structural, not content
    dependent: it holds for noise and for a smooth gradient alike.

    The consequence is that band 0's delta_auc does not measure reliance on the
    lowest frequency band — it measures the model's response to a half-clamped
    image. Band 0 must not be interpreted as a spectral result until the DC
    coefficient is preserved.
    """
    rng = np.random.default_rng(0)
    y, x = np.indices((SIZE, SIZE))
    for name, plane in [
        ("noise", rng.integers(0, 256, (SIZE, SIZE)).astype(np.float64)),
        ("gradient", 120.0 + 60.0 * np.sin(2 * np.pi * x / SIZE)),
    ]:
        a = plane.copy()
        d = dctn(a, type=2, norm="ortho")
        d[radial_band_mask(SIZE, 0, N_BANDS)] = 0.0
        a = idctn(d, type=2, norm="ortho")
        clipped = float(((a < 0) | (a > 255)).mean())
        assert clipped > 0.4, f"{name}: expected ~half the pixels clipped, got {clipped}"

    # A mid band on the same gradient clips nothing, isolating DC as the cause.
    a = 120.0 + 60.0 * np.sin(2 * np.pi * x / SIZE)
    d = dctn(a, type=2, norm="ortho")
    d[radial_band_mask(SIZE, 2, N_BANDS)] = 0.0
    a = idctn(d, type=2, norm="ortho")
    assert float(((a < 0) | (a > 255)).mean()) == 0.0


# --- the ordering trap the docstring warns about -----------------------------

def test_ablation_is_applied_after_the_resize_and_crop():
    """The docstring's stated trap: ablating before the resize lets interpolation
    smear the zeroed band. The mask must sit after CenterCrop and before ToTensor.
    """
    names = [t.__class__.__name__ for t in _transform(SIZE, None).transforms]
    assert names.index("ZeroDCTBand") > names.index("CenterCrop")
    assert names.index("ZeroDCTBand") < names.index("ToTensor")
    assert names.index("Normalize") == len(names) - 1
