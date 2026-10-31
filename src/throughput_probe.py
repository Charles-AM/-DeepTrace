"""Measure training throughput per step, and project the campaign cost from it.

The prespecified timing gate trains one full epoch, which on the real c40 set
means waiting several minutes with no output before learning anything. This
measures the same quantity faster and more directly: time N optimiser steps on
the real data, then project.

It also reports the figure the full-epoch timer cannot isolate -- the NPR
overhead -- by probing both configs back to back under identical conditions.

⚠️ Throughput only. It trains nothing, saves nothing, and its loss values are
meaningless. It must never be cited as a result.

    python -m src.throughput_probe --data-root /kaggle/input/.../ffpp_c40_crops
"""
from __future__ import annotations

import argparse
import json
import math
import time

import torch

from .config import build_model
from .data import build_dataloaders
from .losses import FocalLoss
from .utils import get_device


def project(ms_per_step: float, n_train: int, batch_size: int, epochs: int,
            runs: int) -> dict:
    """Pure, so the arithmetic is testable without a GPU."""
    steps_per_epoch = math.ceil(n_train / batch_size)
    epoch_s = steps_per_epoch * ms_per_step / 1000.0
    run_s = epoch_s * epochs
    return {"steps_per_epoch": steps_per_epoch,
            "epoch_min": epoch_s / 60.0,
            "run_min": run_s / 60.0,
            "campaign_h": run_s * runs / 3600.0}


def probe(config: str, loader, device, steps: int, warmup: int,
          image_size: int) -> float:
    """Median ms per optimiser step, after discarding warmup steps."""
    model = build_model(config, image_size=image_size, pretrained=False).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    crit = FocalLoss(gamma=2.0, alpha=0.2)
    times: list[float] = []
    it = iter(loader)
    for i in range(steps + warmup):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)
        x, y = x.to(device), y.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        crit(model(x), y).backward()
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000.0
        if i >= warmup:
            times.append(dt)
    times.sort()
    return times[len(times) // 2]


def run(data_root: str, configs: list[str], steps: int = 20, warmup: int = 5,
        batch_size: int = 64, image_size: int = 128, num_workers: int = 2,
        epochs: int = 15, runs: int = 10, split_seed: int = 0,
        group_by: str = "videos-([0-9]+)") -> dict:
    device = get_device()
    loaders, dsets = build_dataloaders(data_root=data_root, image_size=image_size,
                                       batch_size=batch_size, num_workers=num_workers,
                                       seed=split_seed, group_by=group_by)
    n_train = len(dsets["train"])
    print(f"device={device}  train={n_train}  batch={batch_size}  "
          f"steps={steps} (+{warmup} warmup)")
    out: dict = {"device": str(device), "n_train": n_train,
                 "batch_size": batch_size, "steps": steps, "configs": {}}
    for cfg in configs:
        ms = probe(cfg, loaders["train"], device, steps, warmup, image_size)
        p = project(ms, n_train, batch_size, epochs, runs)
        out["configs"][cfg] = {"ms_per_step": round(ms, 1), **p}
        print(f"\n{cfg}")
        print(f"  {ms:7.1f} ms/step   {p['steps_per_epoch']} steps/epoch")
        print(f"  epoch ~{p['epoch_min']:.1f} min   {epochs}-epoch run "
              f"~{p['run_min'] / 60:.2f} h   {runs} runs ~{p['campaign_h']:.1f} h")
    if len(configs) == 2:
        a, b = configs
        ma = out["configs"][a]["ms_per_step"]
        mb = out["configs"][b]["ms_per_step"]
        out["overhead_pct"] = round((mb / ma - 1.0) * 100.0, 1)
        # Both arms are trained, so the campaign pays the baseline cost five
        # times and the NPR cost five times -- not the NPR cost ten times.
        half = runs // 2
        out["campaign_h_both_arms"] = round(
            (out["configs"][a]["run_min"] * half
             + out["configs"][b]["run_min"] * half) / 60.0, 2)
        print(f"\n{b} vs {a}: {out['overhead_pct']:+.1f}% per step")
        print(f"campaign ({half} runs each arm): "
              f"~{out['campaign_h_both_arms']:.1f} GPU-h")
    print("\n⚠️ Throughput only. Not a result. Loss values here are meaningless.")
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True)
    p.add_argument("--configs", default="xception,xception_npr")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--out", default=None, help="write the measurements as JSON")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    res = run(a.data_root, a.configs.split(","), a.steps, a.warmup, a.batch_size,
              a.image_size, a.num_workers, a.epochs, a.runs)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=2)
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
