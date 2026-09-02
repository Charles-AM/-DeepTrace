"""Task 4 — training entrypoint.

    python -m src.train --data-root /kaggle/input/<ds> --config full --epochs 15

Trains one named config (see ``src/config.py``), evaluates on val each epoch,
keeps the best checkpoint by val ROC-AUC, then reports test metrics and appends a
row to ``results/summary.csv`` for the comparison tables (Tasks 5-7).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from .config import build_model
from .data import build_dataloaders
from .engine import evaluate, train_one_epoch
from .losses import FocalLoss
from .metrics import METRIC_KEYS
from .utils import count_parameters, get_device, seed_everything


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a DeepTrace configuration")
    p.add_argument("--data-root", required=True)
    p.add_argument("--config", default="full", help="name from src/config.py MODEL_CONFIGS")
    p.add_argument("--dataset-name", default="ffpp", help="tag used in manifest + results")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--focal-alpha", type=float, default=None, help="default: from train class balance")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--amp", action="store_true", help="mixed precision (CUDA only)")
    p.add_argument("--group-by", default=None, help="regex to group crops (avoid frame leakage)")
    p.add_argument("--limit", type=int, default=None, help="cap items per split (debug)")
    p.add_argument("--sas", action="store_true", help="Spectral Artifact Simulation augmentation (Task 10)")
    p.add_argument("--sas-fake-ratio", type=float, default=0.5, help="fraction of the SAS set that are pseudo-fakes")
    p.add_argument("--band-dropout-p", type=float, default=None, help="override the config's frequency-band dropout")
    p.add_argument("--out-root", default="results")
    p.add_argument("--device", default=None)
    return p.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    seed_everything(args.seed)
    device = get_device(args.device)

    run_name = f"{args.dataset_name}_{args.config}{'_sas' if args.sas else ''}_seed{args.seed}"
    out_dir = Path(args.out_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = Path(args.out_root) / "manifests" / f"{args.dataset_name}_seed{args.seed}_sz{args.image_size}.csv"

    loaders, datasets = build_dataloaders(
        args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        group_by=args.group_by,
        manifest=manifest,
        limit=args.limit,
        sas={"fake_ratio": args.sas_fake_ratio} if args.sas else None,
    )
    n_real, n_fake = datasets["train"].class_counts()
    alpha = args.focal_alpha if args.focal_alpha is not None else n_real / max(n_real + n_fake, 1)

    model_overrides = {}
    if args.band_dropout_p is not None:
        model_overrides["band_dropout_p"] = args.band_dropout_p
    model = build_model(
        args.config, image_size=args.image_size, pretrained=not args.no_pretrained, **model_overrides
    ).to(device)
    criterion = FocalLoss(gamma=args.focal_gamma, alpha=float(min(max(alpha, 1e-3), 1 - 1e-3)))
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler() if (args.amp and device.type == "cuda") else None
    writer = SummaryWriter(out_dir / "tb")

    print(
        f"[{run_name}] device={device} train={len(datasets['train'])} "
        f"val={len(datasets['val'])} test={len(datasets['test'])} "
        f"real:fake={n_real}:{n_fake} focal_alpha={alpha:.3f} "
        f"trainable_params={count_parameters(model):,}"
    )

    best_auc, best_epoch = -1.0, -1
    history = []
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, loaders["train"], optimizer, criterion, device,
            grad_clip=args.grad_clip, scaler=scaler, writer=writer, epoch=epoch,
        )
        scheduler.step()
        val_metrics, _, _ = evaluate(model, loaders["val"], device, criterion)

        for k, v in val_metrics.items():
            writer.add_scalar(f"val/{k}", v, epoch)
        writer.add_scalar("train/loss_epoch", train_loss, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
        if model.get_alpha() is not None:
            writer.add_scalar("fusion/alpha", model.get_alpha(), epoch)

        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})
        print(
            f"  epoch {epoch:2d}  train_loss={train_loss:.4f}  "
            f"val_auc={val_metrics['roc_auc']:.4f}  val_acc={val_metrics['accuracy']:.4f}  "
            f"val_eer={val_metrics['eer']:.4f}  ({time.time()-t0:.0f}s)"
        )

        if val_metrics["roc_auc"] > best_auc:
            best_auc, best_epoch = val_metrics["roc_auc"], epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": args.config,
                    "args": vars(args),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                out_dir / "best.pt",
            )

    # final test with the best checkpoint
    ckpt = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_metrics, y_true, y_score = evaluate(model, loaders["test"], device, criterion)
    test_metrics["best_epoch"] = best_epoch
    test_metrics["alpha"] = model.get_alpha()

    (out_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    _append_summary(Path(args.out_root) / "summary.csv", run_name, args, test_metrics)
    if model.get_frequency_mask() is not None:
        torch.save(model.get_frequency_mask(), out_dir / "frequency_mask.pt")
    writer.close()

    print(
        f"[{run_name}] DONE  test_auc={test_metrics['roc_auc']:.4f}  "
        f"test_acc={test_metrics['accuracy']:.4f}  test_f1={test_metrics['f1']:.4f}  "
        f"test_eer={test_metrics['eer']:.4f}  (best val epoch {best_epoch})"
    )
    return test_metrics


def _append_summary(path: Path, run_name: str, args, metrics: dict) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["run", "config", "sas", "dataset", "seed", "image_size", "epochs", *METRIC_KEYS]
    row = {
        "run": run_name,
        "config": args.config,
        "sas": int(bool(getattr(args, "sas", False))),
        "dataset": args.dataset_name,
        "seed": args.seed,
        "image_size": args.image_size,
        "epochs": args.epochs,
        **{k: round(metrics.get(k, float("nan")), 5) for k in METRIC_KEYS},
    }
    # rewrite the whole file so a schema change (e.g. adding the `sas` column) can't
    # misalign appended rows against an older header
    prior = []
    if path.exists():
        with path.open(newline="") as fh:
            prior = list(csv.DictReader(fh))
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in prior:
            w.writerow({k: r.get(k, "") for k in fields})
        w.writerow(row)


if __name__ == "__main__":
    main()
