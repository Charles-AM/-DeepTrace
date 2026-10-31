"""The projection arithmetic, which is what the cost claim rests on."""
from src.throughput_probe import project


def test_steps_per_epoch_rounds_up():
    # 24,000 / 64 = 375 exactly; 24,001 must not silently drop the tail batch.
    assert project(100, 24000, 64, 15, 10)["steps_per_epoch"] == 375
    assert project(100, 24001, 64, 15, 10)["steps_per_epoch"] == 376


def test_projection_scales_linearly():
    a = project(100, 24000, 64, 15, 10)
    b = project(200, 24000, 64, 15, 10)
    assert b["epoch_min"] == 2 * a["epoch_min"]
    assert b["campaign_h"] == 2 * a["campaign_h"]


def test_known_values():
    p = project(400.0, 24000, 64, 15, 10)
    assert round(p["epoch_min"], 2) == 2.5          # 375 * 0.4 s = 150 s
    assert round(p["run_min"], 1) == 37.5           # 15 epochs
    assert round(p["campaign_h"], 2) == 6.25        # 10 runs


def test_epochs_and_runs_are_independent_factors():
    doubled_epochs = project(100, 24000, 64, 30, 10)["run_min"]
    base = project(100, 24000, 64, 15, 10)["run_min"]
    assert doubled_epochs == 2 * base
    # Doubling epochs must not also double the per-run figure twice over.
    assert project(100, 24000, 64, 15, 20)["run_min"] == base
