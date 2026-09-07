"""Build a training command from the recorded reference configuration.

One implementation, used by every runbook script. The `--amp` incident cost a
155-minute run because a flag lived in one script and nowhere else; two scripts
each building their own command line would reintroduce exactly that failure mode
with an extra step.

`results/reference/train_args_c40_vid.json` records the EFFECTIVE configuration
of the varying-split c40 runs — what `run_ablation.py` passes plus the
`train.py` defaults it inherits. Anything compared against those runs must be
trained with the same command.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCALARS = {"epochs": "--epochs", "batch_size": "--batch-size", "lr": "--lr",
           "weight_decay": "--weight-decay", "image_size": "--image-size",
           "num_workers": "--num-workers", "focal_gamma": "--focal-gamma",
           "focal_alpha": "--focal-alpha", "grad_clip": "--grad-clip",
           "group_by": "--group-by", "limit": "--limit",
           "band_dropout_p": "--band-dropout-p"}
FLAGS = {"amp": "--amp", "no_pretrained": "--no-pretrained", "sas": "--sas"}


def load_reference(path="results/reference/train_args_c40_vid.json") -> dict:
    return json.loads(Path(path).read_text())


def build_train_cmd(args: dict, config: str, seed: int, split_seed: int,
                    data_root: str, out_root: str,
                    dataset_name: str | None = None) -> list[str]:
    """Store-true flags are emitted only when the reference says true, so a false
    entry cannot silently become a passed flag. A reference key that this builder
    does not handle raises — silently dropping a hyperparameter is the same class
    of bug as adding one, in the other direction."""
    cmd = [sys.executable, "-m", "src.train",
           "--data-root", data_root, "--config", config,
           "--dataset-name", dataset_name or args["dataset_name"],
           "--seed", str(seed), "--split-seed", str(split_seed),
           "--out-root", str(out_root)]
    for key, opt in SCALARS.items():
        if args[key] is not None:
            cmd += [opt, str(args[key])]
    for key, opt in FLAGS.items():
        if args[key]:
            cmd.append(opt)
    unhandled = {k for k in args if not k.startswith("_")
                 and k not in SCALARS and k not in FLAGS and k != "dataset_name"}
    if unhandled:
        raise ValueError(f"reference file has parameters this builder does not "
                         f"pass: {sorted(unhandled)}")
    return cmd
