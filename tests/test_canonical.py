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


# --- the crossed README's own result table -----------------------------------
# The README declares canonical.json "the single source every table and paragraph
# must draw from", but nothing checked the README itself. Its two conditional rows
# had drifted to figures matching no column of the source CSV; the error surfaced
# only because the stored non-additivity ratio would not reconcile from them.

CROSSED_README = pathlib.Path("results/analysis/crossed/README.md").read_text()

_ROW_LABELS = [
    ("test components only; five runs fixed", "component_only"),
    ("training runs only; test content fixed", "seed_only"),
    ("training runs and test components", "crossed"),
]


def _readme_row_numbers(label):
    """The four numeric cells of one table row: ci_lo, ci_hi, halfwidth, se."""
    line = next(l for l in CROSSED_README.splitlines() if label in l)
    plain = line.replace("−", "-").replace("**", "")
    nums = [float(t) for t in __import__("re").findall(r"[-+]?\d*\.\d+", plain)]
    assert len(nums) == 4, f"expected 4 numbers in the {label!r} row, got {nums}"
    return nums


@pytest.mark.parametrize("label,prefix", _ROW_LABELS)
def test_crossed_readme_matches_canonical(label, prefix):
    """Each README row must agree with its CSV columns, allowing 4-dp rounding."""
    ci_lo, ci_hi, halfwidth, se = _readme_row_numbers(label)
    assert ci_lo == pytest.approx(float(ROW[f"{prefix}_ci_lo"]), abs=5e-5)
    assert ci_hi == pytest.approx(float(ROW[f"{prefix}_ci_hi"]), abs=5e-5)
    assert halfwidth == pytest.approx(float(ROW[f"{prefix}_halfwidth"]), abs=5e-5)
    assert se == pytest.approx(float(ROW[f"{prefix}_se"]), abs=5e-7)


def test_nonadditivity_ratio_reconciles_from_the_three_standard_errors():
    """var(crossed) / (var(component_only) + var(seed_only)) — the stored formula.

    This is the check that caught the README drift: the figures printed there gave
    1.640 against a stored 1.6973.
    """
    assert ROW["nonadditivity_formula"] == (
        "var(crossed) / (var(component_only) + var(seed_only))")
    vx = float(ROW["crossed_se"]) ** 2
    vc = float(ROW["component_only_se"]) ** 2
    vs = float(ROW["seed_only_se"]) ** 2
    assert float(ROW["nonadditivity_ratio"]) == pytest.approx(vx / (vc + vs), abs=1e-3)


def test_crossed_readme_half_width_ratio_claim_holds():
    """The README claims the crossed half-width is 1.8-2.0x the conditionals'."""
    assert "1.8–2.0× the conditional half-widths" in CROSSED_README
    crossed = float(ROW["crossed_halfwidth"])
    ratios = [crossed / float(ROW[f"{p}_halfwidth"])
              for p in ("component_only", "seed_only")]
    assert all(1.8 <= r <= 2.05 for r in ratios), ratios
