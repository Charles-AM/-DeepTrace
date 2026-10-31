"""The pre-analysis gate must FAIL on the conditions it exists to catch.

A gate that only ever passes is decoration. Each negative control below perturbs
exactly one prespecified condition and asserts the gate returns non-zero and names
that specific check.

The contamination control is the important one: it mirrors the real hazard, where a
glob reaches a second campaign directory whose dumps share the split, the test items
and the component structure. Every content check passes in that case, so containment
is the only thing standing between a stray glob and a silently mixed estimate.
"""
from __future__ import annotations

import csv
import json

import pytest

from src.campaign_gate import run

FIELDS = ["run", "config", "seed", "split_seed", "split", "compression", "manipulation",
          "label", "target_seq", "source_seq", "video_id", "frame_idx",
          "logit_margin", "prob_fake", "path"]
MANIPS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]


def _dump(path, config, seed, split_seed="0", n_vid=8, n_frames=4, drop=0):
    rows = []
    for v in range(n_vid):
        manip = MANIPS[v % len(MANIPS)]
        tgt, src = f"{v:03d}", f"{(v + 50):03d}"
        for f in range(n_frames):
            rows.append({
                "run": f"run_{config}_seed{seed}", "config": config, "seed": seed,
                "split_seed": split_seed, "split": "test", "compression": "c40",
                "manipulation": manip, "label": v % 2, "target_seq": tgt,
                "source_seq": src, "video_id": tgt, "frame_idx": f * 12,
                "logit_margin": 0.5 + 0.01 * v, "prob_fake": 0.6,
                "path": f"/crops/{manip}-{tgt}-{src}_{f:05d}.jpg"})
    if drop:
        rows = rows[:-drop]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def _campaign(d, seeds=(0, 1, 2, 3, 4), split_seed="0", b_config="xception_npr"):
    d.mkdir(parents=True, exist_ok=True)
    for s in seeds:
        _dump(d / f"ffpp_xception_seed{s}_test.csv", "xception", s, split_seed)
        _dump(d / f"ffpp_{b_config}_seed{s}_test.csv", b_config, s, split_seed)
    manifest = d / "runs.json"
    manifest.write_text(json.dumps(
        [{"seed": s, "session_id": "S-A", "model_order": "a_first"} for s in seeds]))
    return manifest


def _call(d, manifest, a="*_xception_seed*_test.csv", b="*_xception_npr_seed*_test.csv",
          **kw):
    return run(str(d / a), str(d / b), str(d), str(manifest) if manifest else None, **kw)


def test_clean_campaign_passes(tmp_path):
    d = tmp_path / "npr"
    m = _campaign(d)
    assert _call(d, m) == 0


def test_contamination_is_caught(tmp_path):
    """A second directory with identical split, items and components. Only
    containment distinguishes it — which is the whole reason the check exists."""
    d, other = tmp_path / "npr", tmp_path / "v8"
    m = _campaign(d)
    _campaign(other)
    rc = run(str(other / "*_xception_seed*_test.csv"),
             str(d / "*_xception_npr_seed*_test.csv"), str(d), str(m))
    assert rc == 1


def test_missing_seed_is_caught(tmp_path):
    d = tmp_path / "npr"
    m = _campaign(d)
    (d / "ffpp_xception_npr_seed3_test.csv").unlink()
    assert _call(d, m) == 1


def test_mixed_split_seed_is_caught(tmp_path):
    d = tmp_path / "npr"
    m = _campaign(d)
    _dump(d / "ffpp_xception_seed2_test.csv", "xception", 2, split_seed="1")
    assert _call(d, m) == 1


def test_swapped_arms_are_caught(tmp_path):
    d = tmp_path / "npr"
    m = _campaign(d)
    rc = run(str(d / "*_xception_npr_seed*_test.csv"),
             str(d / "*_xception_seed*_test.csv"), str(d), str(m))
    assert rc == 1


def test_item_membership_mismatch_is_caught(tmp_path):
    d = tmp_path / "npr"
    m = _campaign(d)
    _dump(d / "ffpp_xception_npr_seed1_test.csv", "xception_npr", 1, drop=3)
    assert _call(d, m) == 1


def test_component_mismatch_is_caught(tmp_path):
    d = tmp_path / "npr"
    m = _campaign(d)
    # Same row count, different source sequences -> different union-find components.
    _dump(d / "ffpp_xception_npr_seed4_test.csv", "xception_npr", 4, n_vid=8, n_frames=4)
    rows = list(csv.DictReader(open(d / "ffpp_xception_npr_seed4_test.csv")))
    for r in rows:
        r["source_seq"] = "999"
    with open(d / "ffpp_xception_npr_seed4_test.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    assert _call(d, m) == 1


def test_pair_split_across_sessions_is_caught(tmp_path):
    d = tmp_path / "npr"
    _campaign(d)
    bad = d / "split_sessions.json"
    recs = [{"seed": s, "session_id": "S-A", "model_order": "a_first"} for s in range(5)]
    recs.append({"seed": 3, "session_id": "S-B", "model_order": "b_first"})
    bad.write_text(json.dumps(recs))
    assert _call(d, bad) == 1


def test_missing_manifest_is_caught(tmp_path):
    """The session rule is unverifiable without the log, so absence must block."""
    d = tmp_path / "npr"
    _campaign(d)
    assert _call(d, None) == 1


def test_no_dumps_matched_is_caught(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    assert run(str(d / "nope*.csv"), str(d / "nope*.csv"), str(d), None) == 1
