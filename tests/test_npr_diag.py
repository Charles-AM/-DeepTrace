"""Tests for the loss-geometry diagnostic (docs/npr-loss-geometry-prespecification.md).

The decision rule is prespecified, so it is tested rather than eyeballed in a log.
The command tests matter most: an intervention that never reaches the training
command line would produce three identical runs and a confident wrong conclusion.
"""
import csv
import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("npr_diag", REPO / "docs" / "npr_diag.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


diag = _load()


# -- the prespecified rule -------------------------------------------------

def test_constant_predictor_is_collapsed():
    assert diag.verdict(auc=0.5, sd=0.0) == "collapsed"


def test_discriminating_run_is_not_collapsed():
    assert diag.verdict(auc=0.7978, sd=0.21) == "escaped"


def test_chance_auc_with_spread_scores_is_not_a_collapse():
    """A model that ranks at chance but outputs a spread of scores is a different
    failure. Calling it a constant-prediction collapse would misreport it."""
    assert diag.verdict(auc=0.50, sd=0.25) == "escaped"


def test_constant_scores_at_high_auc_cannot_happen_but_is_not_collapsed():
    assert diag.verdict(auc=0.90, sd=0.0) == "escaped"


@pytest.mark.parametrize("auc,sd,want", [
    (0.55, 0.009, "collapsed"),    # rule is AUC <= 0.55
    (0.551, 0.009, "escaped"),
    (0.50, 0.01, "escaped"),       # rule is sd < 0.01, so equality escapes
    (0.50, 0.00999, "collapsed"),
])
def test_thresholds_are_the_documented_boundaries(auc, sd, want):
    assert diag.verdict(auc, sd) == want


def test_missing_values_are_unknown_not_collapsed():
    """A failed run with no dump must not be silently counted as a collapse."""
    assert diag.verdict(None, None) == "unknown"
    assert diag.verdict(0.5, None) == "unknown"


# -- reading a dump --------------------------------------------------------

def _write_dump(path, probs, labels):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "label", "prob_fake"])
        for p, y in zip(probs, labels):
            w.writerow(["r", y, p])


def test_dump_stats_reproduces_the_campaign_seed0_signature(tmp_path):
    """A constant dump must give exactly what the collapsed campaign run reported:
    AUC 0.5000 on an 80/20 test set."""
    p = tmp_path / "d.csv"
    labels = [0] * 600 + [1] * 2400
    _write_dump(p, [0.5] * 3000, labels)
    auc, sd, mean, n = diag.dump_stats(p)
    assert (round(auc, 4), round(sd, 6), round(mean, 4), n) == (0.5, 0.0, 0.5, 3000)
    assert diag.verdict(auc, sd) == "collapsed"


def test_dump_stats_on_a_separating_dump(tmp_path):
    p = tmp_path / "d.csv"
    labels = [0] * 600 + [1] * 2400
    _write_dump(p, [0.1] * 600 + [0.9] * 2400, labels)
    auc, sd, mean, n = diag.dump_stats(p)
    assert auc == 1.0 and sd > 0.01
    assert diag.verdict(auc, sd) == "escaped"


def test_single_class_dump_returns_no_auc(tmp_path):
    p = tmp_path / "d.csv"
    _write_dump(p, [0.5] * 10, [1] * 10)
    auc, sd, mean, n = diag.dump_stats(p)
    assert auc is None and diag.verdict(auc, sd) == "unknown"


# -- the conditions reach the training command ----------------------------

def test_control_is_command_identical_to_the_campaign():
    """The control must carry NO --focal-alpha, so alpha is derived from class
    balance exactly as the campaign's NPR runs derived it."""
    tag, alpha, gamma, p_star = diag.CONDITIONS[0]
    assert (tag, alpha, gamma) == ("ctrl", None, 2.0)
    assert p_star == 0.500


def test_the_three_conditions_change_one_factor_each():
    by = {t: (a, g) for t, a, g, _ in diag.CONDITIONS}
    assert by["ctrl"] == (None, 2.0)
    assert by["alpha"] == (0.5, 2.0)      # gamma unchanged
    assert by["gamma"] == (None, 0.0)     # alpha unchanged


def test_interventions_appear_on_the_built_command_line():
    from src.reference_cmd import build_train_cmd, load_reference
    args = load_reference(REPO / "results" / "reference" / "train_args_c40_vid.json")
    built = {}
    for tag, alpha, gamma, _ in diag.CONDITIONS:
        ov = dict(args, focal_alpha=alpha, focal_gamma=gamma,
                  dataset_name=f"{diag.DS}_{tag}")
        built[tag] = [str(x) for x in build_train_cmd(
            ov, config=diag.CFG, seed=0, split_seed=0,
            data_root="/d", out_root="/o")]

    assert "--focal-alpha" not in built["ctrl"]
    assert "--focal-gamma" in built["ctrl"]
    assert built["ctrl"][built["ctrl"].index("--focal-gamma") + 1] == "2.0"

    assert built["alpha"][built["alpha"].index("--focal-alpha") + 1] == "0.5"
    assert built["alpha"][built["alpha"].index("--focal-gamma") + 1] == "2.0"

    assert "--focal-alpha" not in built["gamma"]
    assert built["gamma"][built["gamma"].index("--focal-gamma") + 1] == "0.0"

    # All three must be distinguishable, or two conditions would share a
    # checkpoint directory and the second would be skipped as already done.
    assert len({tuple(c) for c in built.values()}) == 3


def test_conditions_write_to_distinct_run_directories():
    names = {f"{diag.DS}_{t}_{diag.CFG}_seed{diag.SEED}" for t, *_ in diag.CONDITIONS}
    assert len(names) == len(diag.CONDITIONS)


def test_diagnostic_output_cannot_be_caught_by_a_campaign_glob():
    """results_diag/ and preds_diag/ exist so no analysis glob reaches these."""
    assert "preds_npr" not in str(diag.PRED) and "diag" in str(diag.PRED)
    assert str(diag.OUT).rstrip("/").endswith("results_diag")
