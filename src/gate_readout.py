"""Read the learned gated-fusion weight alpha from trained checkpoints.

For a ``fusion="gated"`` hybrid, ``alpha = sigmoid(alpha_logit)`` is the weight on
the SPATIAL branch (``fused = alpha * spatial + (1 - alpha) * frequency``), init 0.5.
alpha -> 1 means the model learned to ignore the frequency branch.

    python -m src.gate_readout --results-root results
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch


def run(results_root: Path, out_dir: Path | None = None) -> pd.DataFrame:
    out_dir = Path(out_dir) if out_dir is not None else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for ckpt_path in sorted(results_root.glob("*/best.pt")):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt["model_state"]
        key = next((k for k in state if k.endswith("alpha_logit")), None)
        if key is None:
            continue
        logit = state[key].item()
        alpha = torch.sigmoid(torch.tensor(logit)).item()
        rows.append(
            {
                "run": ckpt_path.parent.name,
                "config": ckpt["config"],
                "alpha_spatial": round(alpha, 4),
                "one_minus_alpha_freq": round(1 - alpha, 4),
                "alpha_logit": round(logit, 4),
            }
        )
        print(f"{ckpt_path.parent.name:35s} alpha_spatial={alpha:.4f}  freq_weight={1-alpha:.4f}")

    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(out_dir / "fusion_alpha.csv", index=False)
        print(f"\nwrote {out_dir/'fusion_alpha.csv'}")
        g = df.groupby("config")["alpha_spatial"].agg(["mean", "std", "count"])
        print(g.to_string())
    else:
        print("no gated-fusion checkpoints found under", results_root)
    return df


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-root", default="results", help="dir with <run>/best.pt")
    p.add_argument("--out-dir", default=None)
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    run(Path(a.results_root), Path(a.out_dir) if a.out_dir else None)


if __name__ == "__main__":
    main()
