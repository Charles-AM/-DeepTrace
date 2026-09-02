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


def _run_one(spec: str, seed: int, common: list[str]) -> None:
    """`spec` is a config name, optionally with a ``+sas`` suffix to enable
    Spectral Artifact Simulation for that experiment (e.g. ``full_banddrop+sas``)."""
    cfg, *flags = spec.split("+")
    extra = ["--sas"] if "sas" in flags else []
    cmd = [sys.executable, "-m", "src.train", "--config", cfg, "--seed", str(seed), *common, *extra]
    print(f"\n=== {spec} seed={seed} ===\n{' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def _run_name(spec: str, dataset_name: str, seed: int) -> str:
    cfg, *flags = spec.split("+")
    return f"{dataset_name}_{cfg}{'_sas' if 'sas' in flags else ''}_seed{seed}"


def train_matrix(configs, seeds, common: list[str]) -> None:
    for cfg in configs:
        for seed in seeds:
            _run_one(cfg, seed, common)


def aggregate(
    summary_csv: Path, dataset_name: str, configs, seeds, out_dir: Path, reference: str = "full"
) -> pd.DataFrame:
    df = pd.read_csv(summary_csv)

    # map each expected run name -> a readable experiment label ("full+sas" etc.)
    want = {
        _run_name(spec, dataset_name, seed): spec for spec in configs for seed in seeds
    }
    df = df[df.run.isin(want)].copy()
    if df.empty:
        raise RuntimeError(f"no matching rows in {summary_csv} for {configs} x {seeds}")
    df["experiment"] = df.run.map(want)

    order = [c for c in configs if c in set(df.experiment)]
    stats = df.groupby("experiment")[METRIC_KEYS].agg(["mean", "std"]).reindex(order)

    # paired t-tests: reference experiment vs each other, on per-seed ROC-AUC
    pvals = {}
    if reference in order:
        from scipy.stats import ttest_rel

        ref_auc = df[df.experiment == reference].sort_values("seed")["roc_auc"].to_numpy()
        for exp in order:
            if exp == reference:
                continue
            other = df[df.experiment == exp].sort_values("seed")["roc_auc"].to_numpy()
            pvals[exp] = (
                float(ttest_rel(ref_auc, other).pvalue)
                if len(other) == len(ref_auc) >= 2
                else float("nan")
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    stats.to_csv(out_dir / "ablation_table.csv")
    _write_markdown(stats, pvals, df, seeds, reference, out_dir / "ablation_table.md")
    _bar_chart(df, order, out_dir / "ablation_auc.png")
    print(f"\nwrote {out_dir/'ablation_table.md'}, {out_dir/'ablation_auc.png'}")
    return stats


def _write_markdown(stats, pvals, df, seeds, reference, path: Path) -> None:
    n_seeds = df.groupby("experiment")["seed"].nunique().to_dict()
    lines = [f"# Ablation results ({len(seeds)} seeds requested; reference = {reference})\n",
             "| experiment | " + " | ".join(METRIC_KEYS) + f" | n | p (vs {reference}) |",
             "|" + "---|" * (len(METRIC_KEYS) + 3)]
    for exp in stats.index:
        cells = []
        for k in METRIC_KEYS:
            m, s = stats.loc[exp, (k, "mean")], stats.loc[exp, (k, "std")]
            cells.append(f"{m:.4f} ± {0.0 if np.isnan(s) else s:.4f}")
        p = pvals.get(exp)
        p_str = "—" if exp == reference or p is None else ("n/a" if np.isnan(p) else f"{p:.4f}")
        lines.append(f"| {exp} | " + " | ".join(cells) + f" | {n_seeds.get(exp, 0)} | {p_str} |")
    path.write_text("\n".join(lines) + "\n")


def _bar_chart(df, order, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    means = [df[df.experiment == c]["roc_auc"].mean() for c in order]
    stds = [df[df.experiment == c]["roc_auc"].std(ddof=0) for c in order]

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
    p.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS,
                   help="config names; append '+sas' to also enable SAS, e.g. full_banddrop+sas")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--reference", default="full", help="experiment the t-tests compare against")
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
              args.configs, args.seeds, Path(args.out_root), reference=args.reference)


if __name__ == "__main__":
    main()
