"""The training-command builder.

The --amp incident cost a 155-minute run. These pin the behaviours that would
let it happen again.
"""

import pytest

from src.reference_cmd import build_train_cmd, load_reference

BASE = dict(config="xception", seed=3, split_seed=0, data_root="/d", out_root="/o")


def _ref(**over):
    r = {"epochs": 15, "batch_size": 64, "lr": 0.0003, "weight_decay": 0.05,
         "image_size": 128, "num_workers": 2, "focal_gamma": 2.0,
         "focal_alpha": None, "grad_clip": 1.0, "group_by": "videos-([0-9]+)",
         "limit": None, "band_dropout_p": None, "amp": False,
         "no_pretrained": False, "sas": False, "dataset_name": "ffpp_c40_vid"}
    r.update(over)
    return r


def test_a_false_flag_is_never_emitted():
    assert "--amp" not in build_train_cmd(_ref(amp=False), **BASE)


def test_a_true_flag_is_emitted():
    assert "--amp" in build_train_cmd(_ref(amp=True), **BASE)


def test_none_valued_scalars_are_omitted_not_passed_as_none():
    cmd = build_train_cmd(_ref(), **BASE)
    assert "--focal-alpha" not in cmd and "None" not in cmd


def test_every_recorded_hyperparameter_reaches_the_command():
    cmd = " ".join(build_train_cmd(_ref(), **BASE))
    for frag in ("--epochs 15", "--batch-size 64", "--lr 0.0003",
                 "--weight-decay 0.05", "--image-size 128", "--num-workers 2",
                 "--focal-gamma 2.0", "--grad-clip 1.0",
                 "--group-by videos-([0-9]+)"):
        assert frag in cmd, frag


def test_an_unhandled_reference_key_raises_rather_than_being_dropped():
    with pytest.raises(ValueError, match="does not pass"):
        build_train_cmd(_ref(mystery_knob=7), **BASE)


def test_dataset_name_can_be_overridden_for_repeat_runs():
    cmd = build_train_cmd(_ref(), dataset_name="ffpp_c40_vid_rep1", **BASE)
    assert "ffpp_c40_vid_rep1" in cmd and "ffpp_c40_vid " not in " ".join(cmd) + " "


def test_the_committed_reference_file_builds_without_amp():
    """The file that actually drives the runs."""
    ref = load_reference()
    cmd = build_train_cmd(ref, config="xception", seed=0, split_seed=0,
                          data_root="/d", out_root="/o")
    assert "--amp" not in cmd
    assert ref["amp"] is False
