"""C1b — late-fusion complementarity test.

Sidesteps the gated-fusion architecture (and its weight-decay confound, see
docs/validation-plan.md C2) entirely. Question: does frequency_only's score carry
ANY information that helps baseline_spatial, when combined the simplest possible
way — logistic regression fit on held-out val, evaluated on test?

    python -m src.late_fusion --spatial-run ffpp_baseline_spatial_seed0 \
        --freq-run ffpp_frequency_only_seed0 --results-root results \
        --dataset-name ffpp --seed 0 --image-size 128
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .config import build_model
from .data import FaceCropDataset, _transforms, read_manifest
from .engine import predict_scores
from .metrics import compute_metrics
from .utils import get_device


def _scores(ckpt_path: Path, entries, device) -> np.ndarray:
    ckpt = torch.load(ckpt_path, map_location=device)
    sz = ckpt["args"]["image_size"]
    model = build_model(ckpt["config"], image_size=sz, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    ds = FaceCropDataset(entries, transform=_transforms(sz, train=False))
    loader = torch.utils.data.DataLoader(ds, batch_size=128, num_workers=2)
    _, y_score = predict_scores(model, loader, device)
    return y_score


def run(spatial_run: str, freq_run: str, results_root: Path, dataset_name: str,
        seed: int, image_size: int, out_dir: Path | None = None) -> dict:
    import csv
    from sklearn.linear_model import LogisticRegression

    device = get_device()
    manifest = results_root / "manifests" / f"{dataset_name}_seed{seed}_sz{image_size}.csv"
    splits = read_manifest(manifest)
    y_val = np.array([lab for _, lab in splits["val"]])
    y_test = np.array([lab for _, lab in splits["test"]])

    s_val = _scores(results_root / spatial_run / "best.pt", splits["val"], device)
    f_val = _scores(results_root / freq_run / "best.pt", splits["val"], device)
    s_test = _scores(results_root / spatial_run / "best.pt", splits["test"], device)
    f_test = _scores(results_root / freq_run / "best.pt", splits["test"], device)

    spatial_alone = compute_metrics(y_test, s_test)["roc_auc"]
    freq_alone = compute_metrics(y_test, f_test)["roc_auc"]

    # default (L2, C=1.0) fit -- and an effectively unregularised fit, since a
    # weak-but-real coefficient could otherwise be shrunk by the default penalty
    # the same way weight decay shrank alpha_logit (C2) -- close that gap here too.
    Xval = np.stack([s_val, f_val], axis=1)
    Xtest = np.stack([s_test, f_test], axis=1)
    clf = LogisticRegression().fit(Xval, y_val)
    combined = compute_metrics(y_test, clf.predict_proba(Xtest)[:, 1])["roc_auc"]
    clf_unreg = LogisticRegression(C=1e6, max_iter=5000).fit(Xval, y_val)
    combined_unreg = compute_metrics(y_test, clf_unreg.predict_proba(Xtest)[:, 1])["roc_auc"]

    # equal-weight average, no fitting -- a second, dumber combiner as a sanity check
    avg_score = (s_test + f_test) / 2
    avg = compute_metrics(y_test, avg_score)["roc_auc"]

    out = {
        "spatial_run": spatial_run, "freq_run": freq_run, "seed": seed,
        "spatial_alone_auc": round(spatial_alone, 4),
        "freq_alone_auc": round(freq_alone, 4),
        "logreg_fusion_auc": round(combined, 4),
        "logreg_delta_vs_spatial": round(combined - spatial_alone, 4),
        "logreg_unreg_fusion_auc": round(combined_unreg, 4),
        "logreg_unreg_delta_vs_spatial": round(combined_unreg - spatial_alone, 4),
        "avg_fusion_auc": round(avg, 4),
        "avg_delta_vs_spatial": round(avg - spatial_alone, 4),
        "logreg_coef_spatial": round(float(clf.coef_[0][0]), 4),
        "logreg_coef_freq": round(float(clf.coef_[0][1]), 4),
    }
    for k, v in out.items():
        print(f"  {k:28s} {v}")

    out_dir = Path(out_dir) if out_dir is not None else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "late_fusion_results.csv"
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out.keys()))
        if write_header:
            w.writeheader()
        w.writerow(out)
    print(f"  appended to {csv_path}")
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spatial-run", required=True)
    p.add_argument("--freq-run", required=True)
    p.add_argument("--results-root", default="results")
    p.add_argument("--dataset-name", default="ffpp")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--out-dir", default=None)
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    run(a.spatial_run, a.freq_run, Path(a.results_root), a.dataset_name, a.seed, a.image_size,
        out_dir=Path(a.out_dir) if a.out_dir else None)


if __name__ == "__main__":
    main()
