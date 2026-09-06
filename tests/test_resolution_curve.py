"""V7 resolution-curve machinery.

The AUC here replaces sklearn's in a loop that runs hundreds of thousands of
times, so it is pinned against the brute-force pairwise definition of ROC-AUC
(including the tie convention) rather than trusted.
"""

import numpy as np
import pytest

from src.resolution_curve import auc, curve, fit_power_law


def brute_auc(labels, scores):
    """ROC-AUC straight from the definition: P(score_pos > score_neg) + 0.5*ties."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


@pytest.mark.parametrize("seed", range(25))
def test_auc_matches_the_pairwise_definition(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(6, 60))
    labels = rng.integers(0, 2, size=n)
    if labels.min() == labels.max():
        labels[0] = 1 - labels[0]
    scores = rng.normal(size=n).round(1)          # rounding forces ties
    assert auc(labels, scores) == pytest.approx(brute_auc(labels, scores))


def test_auc_handles_perfect_and_inverted_separation():
    labels = np.array([0, 0, 1, 1])
    assert auc(labels, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert auc(labels, np.array([0.9, 0.8, 0.2, 0.1])) == pytest.approx(0.0)
    assert auc(labels, np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.5)


def test_auc_is_nan_when_a_class_is_absent():
    assert np.isnan(auc(np.array([1, 1, 1]), np.array([0.1, 0.2, 0.3])))


def _synthetic(n_units=120, per_unit=5, effect=0.3, seed=0):
    """Clustered paired data: each unit is entirely real or entirely fake, which
    is the structure that makes unstratified small draws degenerate."""
    rng = np.random.default_rng(seed)
    labels, sa, sb, groups = [], [], [], []
    for u in range(n_units):
        y = int(u % 5 != 0)                        # 1 real : 4 fake, as in FF++
        shift = rng.normal(0, 0.5)                 # unit-level correlation
        for _ in range(per_unit):
            labels.append(y)
            base = rng.normal(y + shift, 1.0)
            sa.append(base)
            sb.append(base + effect * y)
            groups.append(f"u{u}")
    return (np.array(labels), np.array(sa), np.array(sb), groups)


def test_curve_width_decreases_with_more_units():
    labels, sa, sb, groups = _synthetic()
    rows = curve(labels, sa, sb, groups, ns=[10, 30, 90], n_draws=25, n_boot=100, seed=1)
    widths = [r["halfwidth_median"] for r in rows]
    assert len(rows) == 3
    assert widths[0] > widths[1] > widths[2], widths


def test_curve_never_returns_a_degenerate_single_class_draw():
    """Units are single-class, so an unstratified draw of 5 could be all-fake and
    silently yield a zero-width interval. Stratification must prevent that."""
    labels, sa, sb, groups = _synthetic()
    rows = curve(labels, sa, sb, groups, ns=[5], n_draws=40, n_boot=60, seed=2)
    assert rows and rows[0]["n_draws_used"] > 0
    assert rows[0]["halfwidth_median"] > 0


def test_fit_recovers_the_sqrt_n_rate_on_ideal_data():
    rows = [{"n_units": n, "halfwidth_median": 1.0 / np.sqrt(n)}
            for n in (10, 20, 40, 80, 160)]
    fit = fit_power_law(rows, targets=(0.05,))
    assert fit["slope"] == pytest.approx(-0.5, abs=1e-6)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-6)
    assert fit["n_for_halfwidth_0.05"] == 400          # (1/0.05)^2
    assert fit["extrapolated_0.05"] is True


def test_fit_flags_when_a_target_is_inside_the_observed_range():
    rows = [{"n_units": n, "halfwidth_median": 1.0 / np.sqrt(n)}
            for n in (10, 20, 40, 80, 160)]
    fit = fit_power_law(rows, targets=(0.2,))         # 1/sqrt(25) -> n=25, observed
    assert fit["extrapolated_0.2"] is False


def test_fit_refuses_too_few_points():
    with pytest.raises(ValueError, match="need >=3"):
        fit_power_law([{"n_units": 10, "halfwidth_median": 0.3}])


def test_degenerate_size_is_flagged_not_reported_as_precise():
    """The trap this guard exists for: at n=5 with a 1:4 unit ratio a draw holds
    one real unit, every bootstrap replicate returns the same AUC, and the
    interval has width 0 — which reads as perfect precision on a resolution
    figure. It must be marked degenerate, not believed."""
    labels, sa, sb, groups = _synthetic(n_units=150, per_unit=1, seed=7)
    rows = curve(labels, sa, sb, groups, ns=[5, 30, 60, 120],
                 n_draws=40, n_boot=120, seed=3)
    by_n = {r["n_units"]: r for r in rows}
    assert by_n[5]["degenerate"] is True
    assert by_n[5]["zero_width_frac"] > 0.5
    assert by_n[120]["degenerate"] is False

    fit = fit_power_law(rows)
    assert fit["sizes_excluded_degenerate"] == "5"
    assert fit["n_points_fitted"] == 3


def test_fit_excludes_zero_widths_instead_of_taking_log_of_zero():
    rows = [{"n_units": 5, "halfwidth_median": 0.0, "degenerate": True},
            {"n_units": 10, "halfwidth_median": 0.32},
            {"n_units": 40, "halfwidth_median": 0.16},
            {"n_units": 160, "halfwidth_median": 0.08}]
    fit = fit_power_law(rows, targets=(0.04,))
    assert np.isfinite(fit["slope"]) and np.isfinite(fit["r2"])
    assert fit["slope"] == pytest.approx(-0.5, abs=1e-6)
    assert fit["n_points_fitted"] == 3


def test_fit_error_names_the_degenerate_sizes():
    rows = [{"n_units": 5, "halfwidth_median": 0.0, "degenerate": True},
            {"n_units": 8, "halfwidth_median": 0.0, "degenerate": True},
            {"n_units": 10, "halfwidth_median": 0.3}]
    with pytest.raises(ValueError, match=r"\[5, 8\]"):
        fit_power_law(rows)


def test_matches_sklearn_where_sklearn_is_available():
    """cluster_boot computes every reported interval with this AUC rather than
    sklearn's, so the two must agree exactly. Skipped where sklearn is absent;
    it runs in CI/Kaggle, which is where the reported numbers are produced."""
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    for _ in range(50):
        n = int(rng.integers(8, 400))
        labels = rng.integers(0, 2, size=n)
        if labels.min() == labels.max():
            labels[0] = 1 - labels[0]
        scores = rng.normal(size=n).round(2)          # ties on purpose
        assert auc(labels, scores) == pytest.approx(
            sklearn_metrics.roc_auc_score(labels, scores), abs=1e-12)
