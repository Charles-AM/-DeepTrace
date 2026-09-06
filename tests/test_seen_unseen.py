"""V2 split construction.

The comparison is only meaningful if the two evaluation sets are matched, so
these tests check the matching itself, not just that files appear.
"""

import csv
import json
from pathlib import Path

import pytest

from src.seen_unseen import _ranks, build

MANIPS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]


def _manifest(tmp_path, n_train_groups=240, n_test_groups=30, crops=20):
    """A synthetic FF++-shaped manifest: each target group is 1 real + 4 fakes,
    each video has `crops` frames."""
    rows = []
    gid = 0
    for split, n in (("train", n_train_groups), ("val", 30), ("test", n_test_groups)):
        for _ in range(n):
            t = f"{gid:03d}"; s = f"{(gid + 500):03d}"; gid += 1
            for f in range(crops):
                rows.append((f"/d/real/original-sequences-youtube-c40-videos-{t}_{f*12:05d}.jpg", 0, split))
            for m in MANIPS:
                for f in range(crops):
                    rows.append((f"/d/fake/manipulated-sequences-{m}-c40-videos-{t}-{s}_{f*12:05d}.jpg", 1, split))
    p = tmp_path / "src_manifest.csv"
    with p.open("w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["path", "label", "split"]); w.writerows(rows)
    return p


def test_ranks_are_evenly_spaced_and_in_range():
    assert _ranks(20, 5) == [0, 4, 8, 12, 16]
    assert _ranks(20, 1) == [0]
    assert max(_ranks(20, 20)) == 19
    assert len(set(_ranks(20, 20))) == 20


def test_ranks_refuse_to_oversample():
    with pytest.raises(ValueError, match="videos have"):
        _ranks(5, 10)


def test_the_two_eval_sets_are_matched(tmp_path):
    """The whole experiment rests on this: same videos, crops, groups, class
    balance and manipulation mix, differing only in seen-vs-unseen."""
    rep = build(_manifest(tmp_path), tmp_path, seed=0)
    a, b = rep["seen_eval"], rep["unseen_eval"]
    assert a["crops"] == b["crops"] == 30 * 5 * 5
    assert a["videos"] == b["videos"] == 150
    assert a["groups"] == b["groups"] == 30
    assert a["reals"] == b["reals"]
    assert a["by_manipulation"] == b["by_manipulation"]


def test_held_out_crops_are_removed_from_training(tmp_path):
    """If they stayed in train the 'seen' set would measure memorisation of the
    exact crops rather than of the video, and the estimate would be inflated."""
    rep = build(_manifest(tmp_path), tmp_path, seed=0)
    assert rep["train_after"] == rep["train_before"] - rep["seen_eval"]["crops"]

    seen_paths = {r["path"] for r in csv.DictReader(open(rep["manifest_seen"]))
                  if r["split"] == "test"}
    train_paths = {r["path"] for r in csv.DictReader(open(rep["manifest_seen"]))
                   if r["split"] == "train"}
    assert not (seen_paths & train_paths)


def test_seen_videos_do_still_appear_in_training(tmp_path):
    """The point of 'seen': the video is in training, the crop is not. If the
    whole video were held out this would just be a second unseen set."""
    rep = build(_manifest(tmp_path), tmp_path, seed=0)
    rows = list(csv.DictReader(open(rep["manifest_seen"])))
    import re
    vid = lambda p: re.search(r"videos-([0-9]+)", p).group(1)
    seen_vids = {vid(r["path"]) for r in rows if r["split"] == "test"}
    train_vids = {vid(r["path"]) for r in rows if r["split"] == "train"}
    assert seen_vids <= train_vids


def test_unseen_eval_videos_never_appear_in_training(tmp_path):
    rep = build(_manifest(tmp_path), tmp_path, seed=0)
    import re
    vid = lambda p: re.search(r"videos-([0-9]+)", p).group(1)
    unseen = {vid(r["path"]) for r in csv.DictReader(open(rep["manifest_unseen"]))
              if r["split"] == "test"}
    train = {vid(r["path"]) for r in csv.DictReader(open(rep["manifest_seen"]))
             if r["split"] == "train"}
    assert not (unseen & train)


def test_validation_split_is_untouched(tmp_path):
    """Checkpoint selection must be identical to the runs this is compared with."""
    src = _manifest(tmp_path)
    rep = build(src, tmp_path, seed=0)
    orig = {r["path"] for r in csv.DictReader(open(src)) if r["split"] == "val"}
    new = {r["path"] for r in csv.DictReader(open(rep["manifest_seen"]))
           if r["split"] == "val"}
    assert orig == new


def test_group_selection_is_seeded_and_reproducible(tmp_path):
    src = _manifest(tmp_path)
    a = build(src, tmp_path / "a", seed=0)["seen_groups"]
    b = build(src, tmp_path / "b", seed=0)["seen_groups"]
    c = build(src, tmp_path / "c", seed=1)["seen_groups"]
    assert a == b and a != c
