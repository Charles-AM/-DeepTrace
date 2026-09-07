"""The canonical results block must not drift from its source artifact.

Every table, figure and paragraph draws from results/canonical.json. If that file
and the CSV it was generated from ever disagree, a number in the paper is wrong
and nothing else would catch it.
"""

import csv
import hashlib
import json
import pathlib

import pytest

CANON = json.loads(pathlib.Path("results/canonical.json").read_text())
ROW = next(csv.DictReader(open(CANON["source_csv"])))


def test_source_csv_is_unchanged():
    """If the CSV is regenerated, canonical.json must be regenerated with it."""
    digest = hashlib.sha256(open(CANON["source_csv"], "rb").read()).hexdigest()
    assert digest == CANON["source_csv_sha256"], (
        "source CSV changed but canonical.json was not regenerated")


@pytest.mark.parametrize("key,col", [
    ("point_estimate", "diff"),
    ("ci_lo", "crossed_ci_lo"),
    ("ci_hi", "crossed_ci_hi"),
    ("exceedance_rate_above_reference", "crossed_p_above_0.014"),
    ("exceedance_rate_2se", "crossed_p_above_0.014_mc2se"),
])
def test_value_matches_source(key, col):
    assert CANON[key] == pytest.approx(float(ROW[col]), abs=1e-12)


@pytest.mark.parametrize("key,col", [
    ("n_training_runs", "n_seeds"), ("n_components", "n_components"),
    ("n_test_videos", "n_items"), ("n_bootstrap_replicates", "crossed_n_used"),
    ("bootstrap_seed", "boot_seed"),
])
def test_count_matches_source(key, col):
    assert CANON[key] == int(ROW[col])


def test_exceedance_count_is_consistent_with_the_rate():
    n = round(CANON["exceedance_rate_above_reference"] * CANON["n_bootstrap_replicates"])
    assert CANON["exceedance_count_above_reference"] == n


def test_mc_band_brackets_the_endpoint_and_clears_the_reference():
    lo, hi = CANON["ci_hi_mc_band"]
    assert lo <= CANON["ci_hi"] <= hi
    assert lo > CANON["reference_effect"], "verdict claims case 1; MC band must clear it"


def test_the_reference_lies_inside_the_interval():
    assert CANON["ci_lo"] < CANON["reference_effect"] < CANON["ci_hi"]
    assert CANON["ci_lo"] < 0 < CANON["ci_hi"]


def test_exceedance_rate_is_consistent_with_inclusion():
    """The endpoint test and the exceedance rate are the same question:
    the reference is inside iff the rate exceeds 0.025."""
    inside = CANON["ci_lo"] < CANON["reference_effect"] < CANON["ci_hi"]
    assert inside == (CANON["exceedance_rate_above_reference"] > 0.025)


def test_superseded_interval_is_recorded_and_different():
    s = CANON["superseded"]
    assert s["n_bootstrap_replicates"] < CANON["n_bootstrap_replicates"]
    assert s["interval"] != [CANON["ci_lo"], CANON["ci_hi"]]


def test_manuscript_sentence_quotes_the_canonical_numbers():
    """The approved wording lives beside the data it quotes. If a number changes,
    the sentence must fail rather than silently disagree with the tables."""
    s = CANON["manuscript_sentence"]
    lo, hi = CANON["ci_hi_mc_band"]
    for value in (f"{CANON['point_estimate']}",
                  f"{CANON['ci_lo']}", f"+{CANON['ci_hi']:.4f}".rstrip("0"),
                  f"{lo}", f"{round(hi, 4)}",
                  f"+{CANON['reference_effect']}",
                  f"{CANON['exceedance_rate_above_reference'] * 100:.2f}%"):
        assert value in s, f"{value!r} missing from the manuscript sentence"


def test_manuscript_sentence_states_the_scope():
    s = CANON["manuscript_sentence"].lower()
    assert "fixed c40 split" in s and "five training runs" in s
    assert "neither demonstrated" in s and "nor excluded" in s
