"""Crossed bootstrap: test content and training run resampled together.

The property that makes this analysis possible is that every model was scored on
the SAME test items. V8 exists to provide that, so these tests check it is
enforced rather than assumed, and that the crossing is real.
"""

import csv
from pathlib import Path

import numpy as np
import pytest

from src.crossed_boot import _seed_of, crossed_bootstrap, load_matched

MANIPS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]


def _write(path, groups, seed, shift=0.0, crops=4):
    """One prediction dump: `groups` target ids, each 1 real + 4 fakes.

    `shift` scales the fake-vs-real separation. It must NOT be added to every
    score: AUC is rank-based, so a constant offset applied to both classes leaves
    it exactly unchanged — which is how an earlier version of this file managed to
    ask for a large effect and generate none.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for g in groups:
        vids = [("real", g, "", 0)] + [(m, g, f"{int(g)+500:03d}", 1) for m in MANIPS]
        for manip, t, s, lab in vids:
            for f in range(crops):
                name = (f"/d/original-sequences-youtube-c40-videos-{t}_{f:05d}.jpg"
                        if lab == 0 else
                        f"/d/manipulated-sequences-{manip}-c40-videos-{t}-{s}_{f:05d}.jpg")
                rows.append({"path": name, "label": lab, "manipulation": manip,
                             "target_seq": t, "source_seq": s, "video_id": t,
                             "frame_idx": f,
                             "logit_margin": float(rng.normal(lab * (1.0 + shift), 1.0))})
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def _pair(tmp_path, n_seeds=4, n_groups=12, shift_b=0.4):
    groups = [f"{i:03d}" for i in range(n_groups)]
    a, b = [], []
    for s in range(n_seeds):
        pa = tmp_path / f"run_xception_seed{s}_test.csv"
        pb = tmp_path / f"run_xception_fad_seed{s}_test.csv"
        _write(pa, groups, seed=100 + s)
        _write(pb, groups, seed=200 + s, shift=shift_b)
        a.append(str(pa)); b.append(str(pb))
    return a, b


def test_seed_is_parsed_from_the_filename():
    assert _seed_of("/x/ffpp_c40_vid_xception_seed3_split0_test.csv") == 3
    with pytest.raises(ValueError, match="cannot read a seed"):
        _seed_of("/x/no_seed_here.csv")


def test_mismatched_test_sets_are_refused(tmp_path):
    """The failure V8's fixed split exists to prevent: under a varying-split
    protocol each seed has different test content, and a crossed bootstrap over
    that is undefined rather than merely noisy."""
    a, b = _pair(tmp_path, n_seeds=2)
    _write(Path(a[1]), [f"{i:03d}" for i in range(9)], seed=7)   # different split
    with pytest.raises(ValueError, match="same test items"):
        load_matched(a, b)


def test_load_matched_aligns_seeds_and_components(tmp_path):
    a, b = _pair(tmp_path, n_seeds=4, n_groups=12)
    labels, A, B, groups, seeds = load_matched(a, b)
    assert seeds == [0, 1, 2, 3]
    assert A.shape == B.shape == (4, len(labels))
    assert len(labels) == 12 * 5          # video-level: 12 groups x 5 videos
    assert len(set(groups)) == 12
    assert labels.sum() == 12 * 4         # 4 fakes per group


def test_crossed_interval_is_wider_than_either_conditional(tmp_path):
    """The whole point: resampling both sources must cost more than either alone."""
    a, b = _pair(tmp_path, n_seeds=5, n_groups=20)
    labels, A, B, groups, _ = load_matched(a, b)
    r = crossed_bootstrap(labels, A, B, groups, n_boot=600, seed=1)
    assert r["crossed"]["halfwidth"] > r["component_only"]["halfwidth"]
    assert r["crossed"]["halfwidth"] > r["seed_only"]["halfwidth"]


def test_point_estimate_is_the_mean_over_seeds(tmp_path):
    from src.resolution_curve import auc
    a, b = _pair(tmp_path, n_seeds=3, n_groups=10)
    labels, A, B, groups, _ = load_matched(a, b)
    r = crossed_bootstrap(labels, A, B, groups, n_boot=200, seed=0)
    manual = np.mean([auc(labels, B[s]) - auc(labels, A[s]) for s in range(3)])
    assert r["diff"] == pytest.approx(round(float(manual), 4), abs=1e-9)


def test_a_real_effect_is_recovered(tmp_path):
    a, b = _pair(tmp_path, n_seeds=5, n_groups=25, shift_b=1.2)
    labels, A, B, groups, _ = load_matched(a, b)
    r = crossed_bootstrap(labels, A, B, groups, n_boot=600, seed=2)
    assert r["diff"] > 0
    assert r["crossed"]["ci_lo"] > 0, "a large true effect should clear the interval"


def test_degenerate_component_count_is_refused(tmp_path):
    a, b = _pair(tmp_path, n_seeds=2, n_groups=1)
    labels, A, B, groups, _ = load_matched(a, b)
    with pytest.raises(ValueError, match="degenerate"):
        crossed_bootstrap(labels, A, B, groups, n_boot=50)
