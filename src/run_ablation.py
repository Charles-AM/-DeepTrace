"""Task 5 — ablation / baseline matrix.

Trains a set of named configs (``src/config.py``) across several seeds by shelling
out to ``src.train`` once per run (subprocess = clean GPU memory each time), then
aggregates ``results/summary.csv`` into:

  * ``results/ablation_table.csv`` / ``.md`` — mean +/- std per config
  * ``results/ablation_auc.png``            — ROC-AUC bar chart with error bars
  * paired t-tests (full vs each other config), p-values in the table

    python -m src.run_ablation --data-root /kaggle/input/<ds> --dataset-name rvf \
        --configs baseline_spatial frequency_only no_mask no_fusion_concat full \
        --seeds 0 1 2 --epochs 15
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import METRIC_KEYS

DEFAULT_CONFIGS = ["baseline_spatial", "frequency_only", "no_mask", "no_fusion_concat", "full"]


def _run_one(cfg: str, seed: int, common: list[str]) -> None:
    cmd = [sys.executable, "-m", "src.train", "--config", cfg, "--seed", str(seed), *common]
    print(f"\n=== {cfg} seed={seed} ===\n{' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def train_matrix(configs, seeds, common: list[str]) -> None:
    for cfg in configs:
        for seed in seeds:
            _run_one(cfg, seed, common)


def aggregate(summary_csv: Path, dataset_name: str, configs, seeds, out_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(summary_csv)
    df = df[(df.dataset == dataset_name) & df.config.isin(configs) & df.seed.isin(seeds)]
    if df.empty:
        raise RuntimeError(f"no rows in {summary_csv} for dataset={dataset_name} configs={configs}")

    stats = df.groupby("config")[METRIC_KEYS].agg(["mean", "std"])
    stats = stats.reindex([c for c in configs if c in stats.index])

    # paired t-tests: full vs each other config, on per-seed ROC-AUC
    pvals = {}
    if "full" in configs:
        from scipy.stats import ttest_rel

        full_auc = df[df.config == "full"].sort_values("seed")["roc_auc"].to_numpy()
        for cfg in configs:
            if cfg == "full":
                continue
            other = df[df.config == cfg].sort_values("seed")["roc_auc"].to_numpy()
            if len(other) == len(full_auc) >= 2:
                pvals[cfg] = float(ttest_rel(full_auc, other).pvalue)
            else:
                pvals[cfg] = float("nan")

    out_dir.mkdir(parents=True, exist_ok=True)
    stats.to_csv(out_dir / "ablation_table.csv")
    _write_markdown(stats, pvals, df, seeds, out_dir / "ablation_table.md")
    _bar_chart(df, configs, out_dir / "ablation_auc.png")
    print(f"\nwrote {out_dir/'ablation_table.md'}, {out_dir/'ablation_auc.png'}")
    return stats


def _write_markdown(stats, pvals, df, seeds, path: Path) -> None:
    n_seeds = df.groupby("config")["seed"].nunique().to_dict()
    lines = [f"# Ablation results ({len(seeds)} seeds requested)\n",
             "| config | " + " | ".join(METRIC_KEYS) + " | n | p (vs full) |",
             "|" + "---|" * (len(METRIC_KEYS) + 3)]
    for cfg in stats.index:
        cells = []
        for k in METRIC_KEYS:
            m, s = stats.loc[cfg, (k, "mean")], stats.loc[cfg, (k, "std")]
            cells.append(f"{m:.4f} ± {0.0 if np.isnan(s) else s:.4f}")
        p = pvals.get(cfg)
        p_str = "—" if cfg == "full" or p is None else ("n/a" if np.isnan(p) else f"{p:.4f}")
        lines.append(f"| {cfg} | " + " | ".join(cells) + f" | {n_seeds.get(cfg, 0)} | {p_str} |")
    path.write_text("\n".join(lines) + "\n")


def _bar_chart(df, configs, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = [c for c in configs if c in df.config.unique()]
    means = [df[df.config == c]["roc_auc"].mean() for c in order]
    stds = [df[df.config == c]["roc_auc"].std(ddof=0) for c in order]

    fig, ax = plt.subplots(figsize=(1.6 * len(order) + 1, 4))
    ax.bar(order, means, yerr=stds, capsize=4, color="#4C72B0")
    ax.set_ylabel("Test ROC-AUC")
    ax.set_title("Ablation: ROC-AUC by configuration")
    ax.set_ylim(min(0.5, min(means) - 0.05), 1.0)
    for i, m in enumerate(means):
        ax.text(i, m + 0.005, f"{m:.3f}", ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run the ablation/baseline matrix")
    p.add_argument("--data-root", required=True)
    p.add_argument("--dataset-name", default="rvf")
    p.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out-root", default="results")
    p.add_argument("--aggregate-only", action="store_true", help="skip training, just rebuild tables")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="extra args passed to src.train")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    common = [
        "--data-root", args.data_root, "--dataset-name", args.dataset_name,
        "--epochs", str(args.epochs), "--image-size", str(args.image_size),
        "--batch-size", str(args.batch_size), "--out-root", args.out_root,
        *(["--limit", str(args.limit)] if args.limit else []),
        *args.extra,
    ]
    if not args.aggregate_only:
        train_matrix(args.configs, args.seeds, common)
    aggregate(Path(args.out_root) / "summary.csv", args.dataset_name,
              args.configs, args.seeds, Path(args.out_root))


if __name__ == "__main__":
    main()
