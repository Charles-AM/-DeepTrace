"""Per-prediction dump — the substrate for cluster-aware inference.

Aggregate metrics (a single AUC per run) cannot support inference over clustered
test content: 3,000 crops drawn from 30 videos are not 3,000 independent
observations. Every downstream analysis (cluster bootstrap, video-level
aggregation, equivalence testing, per-manipulation breakdown) needs one row per
prediction with enough metadata to identify the cluster it belongs to.

Writes: run, config, seed, split, compression, manipulation, label, target_seq,
source_seq, video_id, frame_idx, logit_margin, prob_fake, path.

`logit_margin` (logit_fake − logit_real) is saved alongside `prob_fake` because
aggregation and fusion should operate on an unsquashed scale — averaging
probabilities near 0 or 1 discards information that averaging margins retains.

    python -m src.predict --run ffpp_c40_vid_xception_seed0 --results-root results \\
        --dataset-name ffpp_c40_vid --seed 0 --image-size 128 \\
        --out-dir results/predictions
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from .config import build_model
from .data import FaceCropDataset, _transforms, read_manifest
from .utils import get_device

from .cropname import parse_crop_name  # noqa: F401  (re-exported)

FIELDS = ["run", "config", "seed", "split_seed", "split", "compression",
          "manipulation", "label", "target_seq", "source_seq", "video_id",
          "frame_idx", "logit_margin", "prob_fake", "path"]


@torch.no_grad()
def run(run_name: str, results_root: Path, dataset_name: str, seed: int,
        image_size: int, split: str = "test", batch_size: int = 128,
        num_workers: int = 2, split_seed: int | None = None,
        out_dir: Path | None = None) -> Path:
    """`seed` is the TRAINING seed; `split_seed` selects the manifest.

    They differ for the V8 fixed-split runs, where one partition is held constant
    while training randomness varies. The manifest is named by the split seed, so
    passing only `seed` there would either miss the file or -- worse -- succeed
    and stamp the wrong training seed onto every row.
    """
    device = get_device()
    out_dir = Path(out_dir) if out_dir is not None else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)

    split_seed = seed if split_seed is None else split_seed
    manifest = results_root / "manifests" / f"{dataset_name}_seed{split_seed}_sz{image_size}.csv"
    entries = read_manifest(manifest)[split]

    ckpt = torch.load(results_root / run_name / "best.pt", map_location=device)
    sz = ckpt["args"]["image_size"]
    model = build_model(ckpt["config"], image_size=sz, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ds = FaceCropDataset(entries, transform=_transforms(sz, train=False))
    # shuffle=False so batch order matches ds.entries exactly
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False,
                                          num_workers=num_workers)
    margins, probs = [], []
    for x, _ in loader:
        logits = model(x.to(device, non_blocking=True)).float()
        margins.append((logits[:, 1] - logits[:, 0]).cpu())
        probs.append(logits.softmax(dim=1)[:, 1].cpu())
    margins = torch.cat(margins).numpy()
    probs = torch.cat(probs).numpy()
    assert len(margins) == len(entries), "prediction/entry misalignment"

    out_path = out_dir / f"{run_name}_{split}.csv"
    unparsed: list[str] = []
    with out_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for (path, label), margin, prob in zip(entries, margins, probs):
            meta = parse_crop_name(path)
            if not meta["video_id"]:
                unparsed.append(path)
            w.writerow({"run": run_name, "config": ckpt["config"], "seed": seed,
                        "split_seed": split_seed,
                        "split": split, "label": int(label),
                        "logit_margin": round(float(margin), 6),
                        "prob_fake": round(float(prob), 6), "path": path, **meta})

    # An unparsed crop name yields an EMPTY video_id — and an empty key is still a
    # key, so every such row silently merges into one bogus cluster and corrupts
    # any cluster-aware interval computed downstream. Nothing about the output
    # looks wrong when this happens, so refuse to emit it rather than warn.
    if unparsed:
        out_path.unlink(missing_ok=True)
        raise ValueError(
            f"{len(unparsed)}/{len(entries)} crop names did not parse — refusing to "
            f"write {out_path.name}, since empty video_ids would collapse into a "
            f"single cluster. First offenders: {unparsed[:3]}"
        )
    print(f"wrote {out_path}  ({len(entries)} predictions, "
          f"{len({parse_crop_name(p)['video_id'] for p, _ in entries})} video groups)")
    return out_path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--results-root", default="results")
    p.add_argument("--dataset-name", default="ffpp")
    p.add_argument("--seed", type=int, default=0, help="training seed (recorded)")
    p.add_argument("--split-seed", type=int, default=None,
                   help="seed naming the manifest; defaults to --seed")
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--split", default="test")
    p.add_argument("--out-dir", default=None)
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    run(a.run, Path(a.results_root), a.dataset_name, a.seed, a.image_size,
        split=a.split, split_seed=a.split_seed,
        out_dir=Path(a.out_dir) if a.out_dir else None)


if __name__ == "__main__":
    main()
