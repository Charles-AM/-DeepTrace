"""V6 equivalence/threshold-sensitivity flags.

`equivalent` and `excludes` are different claims and the paper reports them
separately; conflating them is exactly how "not significant" gets written up as
"no effect". These pin the distinction.
"""

import pytest

from src.cluster_boot import equivalence_flags


def test_interval_inside_the_margin_is_equivalent_and_excludes():
    f = equivalence_flags(-0.005, 0.008, [0.014])
    assert f["equivalent_0.014"] is True
    assert f["excludes_0.014"] is True


def test_our_c40_component_interval_excludes_but_is_not_equivalent():
    """The real case: [-0.0534, +0.0096] against the verified +0.014 FAD gain.
    The upper bound rules out a benefit that large; the lower bound is nowhere
    near -0.014, so the result is NOT evidence of equivalence."""
    f = equivalence_flags(-0.0534, 0.0096, [0.014])
    assert f["excludes_0.014"] is True
    assert f["equivalent_0.014"] is False


def test_interval_straddling_the_margin_claims_nothing():
    f = equivalence_flags(-0.02, 0.03, [0.014])
    assert f["equivalent_0.014"] is False
    assert f["excludes_0.014"] is False


def test_interval_entirely_above_the_margin_is_a_positive_effect():
    f = equivalence_flags(0.02, 0.05, [0.014])
    assert f["equivalent_0.014"] is False
    assert f["excludes_0.014"] is False


def test_verdict_can_flip_across_the_sensitivity_range():
    """Why the sweep exists: one interval, four defensible margins, two verdicts."""
    f = equivalence_flags(-0.008, 0.011, [0.005, 0.010, 0.014, 0.020])
    assert f["excludes_0.005"] is False
    assert f["excludes_0.01"] is False
    assert f["excludes_0.014"] is True
    assert f["equivalent_0.02"] is True


def test_no_margins_yields_no_flags():
    assert equivalence_flags(-0.01, 0.01, []) == {}


@pytest.mark.parametrize("m", [0.005, 0.01, 0.014, 0.02])
def test_equivalence_implies_exclusion(m):
    """A one-directional check must never be stricter than the two-directional
    one: if the whole interval sits inside +-m, its top is certainly below +m."""
    for lo, hi in [(-0.004, 0.004), (-0.02, 0.001), (0.0, 0.019), (-0.1, 0.1)]:
        f = equivalence_flags(lo, hi, [m])
        if f[f"equivalent_{m:g}"]:
            assert f[f"excludes_{m:g}"]


def test_margin_key_does_not_depend_on_how_the_number_was_typed():
    """0.010 and 0.01 are the same margin; the CSV header must not disagree."""
    assert equivalence_flags(-0.005, 0.005, [0.010]).keys() == \
           equivalence_flags(-0.005, 0.005, [0.01]).keys()
    assert "excludes_0.01" in equivalence_flags(-0.005, 0.005, [0.010])


def test_colliding_margins_are_rejected_not_silently_merged():
    with pytest.raises(ValueError, match="collide"):
        equivalence_flags(-0.005, 0.005, [0.010, 0.01])
