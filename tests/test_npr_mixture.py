"""Tests for the mixture reporting module (src/npr_mixture.py).

The module exists because crossed_boot.py is protected and its --out-dir writes
only a summary row. These tests pin the exact combinatorics, and check the exact
seed-only distribution against crossed_boot's Monte Carlo interval -- if those two
disagreed, one of them would be computing a different statistic.
"""
from math import comb

import numpy as np
import pytest

from src import npr_mixture as M

A_GLOB = "results/predictions_npr/*_xception_seed*_test.csv"
B_GLOB = "results/predictions_npr/*_xception_npr_seed*_test.csv"


# -- exact combinatorics ---------------------------------------------------

def test_copy_count_probabilities_are_exact_at_five_runs():
    """The numbers the write-up quotes. 0.0512 is EXACTLY three copies -- an
    earlier draft mislabelled it as three-or-more, which is 0.05792."""
    p = M.copy_count_probs(5)
    assert p[0] == pytest.approx(0.32768)
    assert p[1] == pytest.approx(0.40960)
    assert p[2] == pytest.approx(0.20480)
    assert p[3] == pytest.approx(0.05120)
    assert p[4] == pytest.approx(0.00640)
    assert p[5] == pytest.approx(0.00032)
    assert sum(p[k] for k in (3, 4, 5)) == pytest.approx(0.05792)


def test_copy_count_probabilities_sum_to_one():
    for n in (3, 4, 5, 6):
        assert sum(M.copy_count_probs(n).values()) == pytest.approx(1.0)


@pytest.mark.parametrize("n,want", [(4, 35), (5, 126)])
def test_distinct_composition_counts(n, want):
    assert len(M.compositions(n)) == want == comb(2 * n - 1, n - 1)


def test_composition_weights_sum_to_one_and_match_copy_counts():
    for n in (4, 5):
        comps = M.compositions(n)
        assert sum(w for _, w, _ in comps) == pytest.approx(1.0)
        probs = M.copy_count_probs(n)
        for k in range(n + 1):
            got = sum(w for _, w, c0 in comps if c0 == k)
            assert got == pytest.approx(probs[k]), f"n={n} k={k}"


# -- the exact seed-only distribution --------------------------------------

def test_exact_seed_only_enumerates_every_ordered_resample():
    v, w, c = M.exact_seed_only([0.0, 1.0, 2.0])
    assert len(v) == 3 ** 3 == 27
    assert w.sum() == pytest.approx(1.0)
    assert c.max() == 3 and c.min() == 0


def test_exact_seed_only_strata_weights_match_the_closed_form():
    v, w, c = M.exact_seed_only([-0.35, 0.01, 0.00, 0.00, 0.00])
    probs = M.copy_count_probs(5)
    for k in range(6):
        assert w[c == k].sum() == pytest.approx(probs[k])


def test_one_collapsed_run_shifts_each_stratum_by_a_fixed_step():
    """With four near-zero runs and one at -0.35, the strata means form a ladder
    separated by -0.35/5. That is what makes the pooled interval a mixture rather
    than a distribution with a tail."""
    d = [-0.35, 0.0, 0.0, 0.0, 0.0]
    v, w, c = M.exact_seed_only(d)
    means = [np.average(v[c == k], weights=w[c == k]) for k in range(4)]
    steps = np.diff(means)
    assert np.allclose(steps, -0.35 / 5)


def test_single_class_draw_yields_none_like_crossed_boot():
    labels = np.ones(6, dtype=int)
    A = np.zeros((2, 6)); B = np.ones((2, 6))
    assert M.per_seed_diffs(labels, A, B) == [None, None]


# -- summarise -------------------------------------------------------------

def test_summarise_is_weight_aware():
    v = np.array([-1.0, 1.0])
    heavy_low = M.summarise(v, np.array([0.9, 0.1]))
    heavy_high = M.summarise(v, np.array([0.1, 0.9]))
    assert heavy_low["mean"] < 0 < heavy_high["mean"]
    assert heavy_low["p_above_0"] == pytest.approx(0.1)
    assert heavy_high["p_above_0"] == pytest.approx(0.9)


def test_summarise_handles_an_empty_stratum():
    assert M.summarise(np.array([]))["n"] == 0


# -- agreement with the protected module -----------------------------------

@pytest.mark.skipif(not __import__("glob").glob(A_GLOB),
                    reason="campaign 1 dumps not present")
def test_point_estimate_matches_crossed_boot():
    import glob
    from src.crossed_boot import load_matched
    labels, A, B, groups, seeds = load_matched(sorted(glob.glob(A_GLOB)),
                                               sorted(glob.glob(B_GLOB)), True)
    d = M.per_seed_diffs(labels, A, B)
    point = float(np.mean([x for x in d if x is not None]))
    assert round(point, 4) == -0.0667


@pytest.mark.skipif(not __import__("glob").glob(A_GLOB),
                    reason="campaign 1 dumps not present")
def test_exact_seed_only_interval_agrees_with_the_monte_carlo_one():
    """crossed_boot's seed_only draws 4000 samples from a 5**5 = 3125 support.
    Enumerating it exactly must land within Monte Carlo error of that interval --
    if it did not, the two would be computing different statistics."""
    import glob
    from src.crossed_boot import load_matched
    labels, A, B, groups, seeds = load_matched(sorted(glob.glob(A_GLOB)),
                                               sorted(glob.glob(B_GLOB)), True)
    v, w, _ = M.exact_seed_only(M.per_seed_diffs(labels, A, B))
    s = M.summarise(v, w)
    # crossed_boot reported [-0.2099, +0.0088] with 4000 draws, boot_seed 0.
    assert s["p2.5"] == pytest.approx(-0.2099, abs=2e-3)
    assert s["p97.5"] == pytest.approx(+0.0088, abs=2e-3)
