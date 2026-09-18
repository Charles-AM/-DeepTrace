"""Tests for src/agg_space.py.

The property that matters most is PAIRING: both aggregation arms must consume the
same resampling draws, otherwise the contrast between them carries Monte Carlo
noise and the per-replicate difference statistics are meaningless.
"""
import csv
import numpy as np
import pytest

from src.agg_space import fast_auc, run, _components


def _row(vid, manip, label, tgt, src, logit, prob, i):
    return {"video_id": vid, "manipulation": manip, "label": str(label),
            "target_seq": tgt, "source_seq": src, "logit_margin": f"{logit}",
            "prob_fake": f"{prob}", "path": f"/x/videos-{vid}-{src or '000'}_{i:05d}.jpg"}


def _write(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _corpus(tmp_path, n_comp=6, offset=0.0):
    """Two paired runs over n_comp components, each a real + fake video pair."""
    rows_a, rows_b = [], []
    for c in range(n_comp):
        tgt, src = f"{c*2:03d}", f"{c*2+1:03d}"
        for manip, label in (("real", 0), ("Deepfakes", 1)):
            for i in range(3):
                base = 0.3 + 0.4 * label + 0.01 * c
                rows_a.append(_row(tgt, manip, label, tgt, "" if label == 0 else src,
                                   base, base, i))
                rows_b.append(_row(tgt, manip, label, tgt, "" if label == 0 else src,
                                   base + offset, base + offset, i))
    a, b = tmp_path / "a_seed0_test.csv", tmp_path / "b_seed0_test.csv"
    _write(a, rows_a); _write(b, rows_b)
    return str(a), str(b)


def test_fast_auc_perfect_and_chance():
    assert fast_auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert fast_auc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0
    assert fast_auc([0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1]) == 0.5


def test_fast_auc_rejects_single_class():
    with pytest.raises(ValueError, match="one class absent"):
        fast_auc([0.1, 0.2], [1, 1])


def test_components_union_find_pairs_source_and_target():
    rows = [{"target_seq": "001", "source_seq": "002"},
            {"target_seq": "002", "source_seq": "001"},
            {"target_seq": "003", "source_seq": ""}]
    comps = _components(rows)
    assert comps[0] == comps[1], "a source-target pair must share a component"
    assert comps[2] != comps[0], "an unconnected sequence is its own component"


def test_refuses_runs_with_different_test_items(tmp_path):
    a, b = _corpus(tmp_path)
    other = tmp_path / "b2_seed0_test.csv"
    rows = list(csv.DictReader(open(b)))
    rows[0]["path"] = "/x/DIFFERENT.jpg"
    _write(other, rows)
    with pytest.raises(SystemExit, match="identical test items"):
        run(a, str(other), n_boot=10, out_dir=None)


def test_both_arms_are_paired_on_identical_draws(tmp_path):
    """With zero difference between arms, a paired contrast must be exactly zero.

    If the two aggregation spaces drew independently, sampling noise would make
    the per-replicate difference non-zero even for identical inputs.
    """
    a, b = _corpus(tmp_path, offset=0.0)
    _, paired = run(a, b, n_boot=200, boot_seed=0, out_dir=None)
    assert paired["paired_diff_mean"] == 0.0
    assert paired["replicates_disagreeing_on_margin"] == 0.0


def test_reports_both_aggregation_spaces(tmp_path):
    a, b = _corpus(tmp_path)
    out, _ = run(a, b, n_boot=50, boot_seed=0, out_dir=None)
    assert [r["aggregation_space"] for r in out] == ["mean_logit", "mean_probability"]
    for r in out:
        assert r["ci_lo"] <= r["point_estimate"] <= r["ci_hi"] or r["n_boot"] < 100
        assert r["n_components"] == 6
