#!/usr/bin/env python3
"""Night-1 batch: B1 smoke, B2 (V2 seen/unseen), C1 lr sweep, C2 repeat audit.

Written for Kaggle *Save Version -> Save & Run All*. Runs unattended.

Design, inherited from docs/v8_run.py because it was learned the hard way:
  * every training command is built from results/reference/train_args_c40_vid.json,
    so --amp cannot reappear and no hyperparameter can silently drift
  * a reference parameter this script does not pass is a FATAL error, not a
    silent omission
  * stages run cheapest-and-highest-value first, so a session death loses least
  * each stage is independent; one failing does not stop the rest
  * --dry-run prints every command without executing, for validation first

Governed by docs/c1-lr-prespecification.md and docs/c2-repeat-prespecification.md,
both committed before this ran.
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

REPO = Path(os.environ.get("DT_REPO", "/kaggle/working/-DeepTrace"))
OUT = Path(os.environ.get("DT_OUT", "/kaggle/working/night1"))
REF = REPO / "results/reference/train_args_c40_vid.json"
SPLIT_REF = REPO / "results/reference/ffpp_c40_vid_seed0_test_split.json"
DS = "ffpp_c40_vid"
SPLIT_SEED = 0
LRS = [1e-4, 3e-4, 1e-3]          # prespecified, symmetric about the reference 3e-4
CONFIGS = ["xception", "xception_fad"]
C1_EPOCHS = 5                      # prespecified; c40 best-epoch mean ~5
C2_REPEATS = 2

def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

def load_ref():
    if not REF.exists():
        sys.exit(f"FATAL: reference file missing: {REF}")
    return json.loads(REF.read_text())

def train_cmd(ref, cfg, seed, data_root, lr=None, epochs=None, out=None, tag=None):
    """Build a training command entirely from the reference file.

    Overrides are explicit arguments, so any deviation from the reference is
    visible in this signature rather than buried in a dict.
    """
    cmd = [sys.executable, "-m", "src.train",
           "--data-root", str(data_root), "--config", cfg,
           "--dataset-name", tag or DS, "--seed", str(seed),
           "--split-seed", str(SPLIT_SEED), "--out-root", str(out or OUT)]
    scalars = {"epochs": "--epochs", "batch_size": "--batch-size", "lr": "--lr",
               "weight_decay": "--weight-decay", "image_size": "--image-size",
               "num_workers": "--num-workers", "focal_gamma": "--focal-gamma",
               "focal_alpha": "--focal-alpha", "grad_clip": "--grad-clip",
               "group_by": "--group-by", "limit": "--limit",
               "band_dropout_p": "--band-dropout-p"}
    flags = {"amp": "--amp", "no_pretrained": "--no-pretrained", "sas": "--sas"}
    override = {"lr": lr, "epochs": epochs}
    for key, opt in scalars.items():
        val = override.get(key) if override.get(key) is not None else ref[key]
        if val is not None:
            cmd += [opt, str(val)]
    for key, opt in flags.items():
        if ref[key]:
            cmd.append(opt)
    unhandled = {k for k in ref if not k.startswith("_")
                 and k not in scalars and k not in flags and k != "dataset_name"}
    if unhandled:
        sys.exit(f"FATAL: reference has parameters this script does not pass: "
                 f"{sorted(unhandled)}. Add them or remove them.")
    if "--amp" in cmd:
        sys.exit("FATAL: --amp present. Every comparison run in this project is fp32.")
    return cmd

def sh(cmd, dry):
    log("RUN:", " ".join(str(c) for c in cmd))
    if dry:
        return 0
    return subprocess.run(cmd, cwd=REPO).returncode

def find_crops(dry=False):
    for p in ("/kaggle/input/c40-run/ffpp_c40_crops",
              "/kaggle/input/notebooks/charlesappiahmanu/c40-run/ffpp_c40_crops"):
        if Path(p).exists():
            return p
    hits = list(Path("/kaggle/input").glob("**/ffpp_c40_crops"))
    if hits:
        return str(hits[0])
    if dry:
        return "<DRY-RUN-CROPS>"
    sys.exit("FATAL: ffpp_c40_crops not found. Attach the c40-run notebook output.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stages", default="B1,B2,C1,C2")
    a = ap.parse_args()
    dry = a.dry_run
    stages = [s.strip() for s in a.stages.split(",")]
    OUT.mkdir(parents=True, exist_ok=True)
    ref = load_ref()
    log("reference loaded:", {k: ref[k] for k in ("epochs", "lr", "amp", "group_by")})
    if ref["amp"]:
        sys.exit("FATAL: reference says amp=true. Every headline run is fp32.")
    data_root = find_crops(dry)
    log("crops:", data_root)
    results = {}

    # ---- B1: band_ablation smoke test (~5 min) -------------------------
    if "B1" in stages:
        log("=== B1 band_ablation smoke ===")
        runs = sorted((REPO / "results/predictions_v8").glob("*xception_seed0*.csv"))
        cmd = [sys.executable, "-m", "src.band_ablation",
               "--runs", str(runs[0].stem) if runs else "ffpp_c40_vid_xception_seed0",
               "--results-root", str(OUT), "--limit", "200",
               "--out-dir", str(OUT / "band_ablation")]
        results["B1"] = sh(cmd, dry)

    # ---- B2: V2 seen/unseen (~80 min) ---------------------------------
    if "B2" in stages:
        log("=== B2 V2 seen vs unseen ===")
        man = REPO / f"results/manifests/{DS}_seed0_sz128.csv"
        cmd = [sys.executable, "-m", "src.seen_unseen",
               "--manifest", str(man), "--out-root", str(OUT), "--seed", "0"]
        rc = sh(cmd, dry)
        if rc == 0:
            # train ONE model on the SEEN manifest, then score against both
            results["B2_train"] = sh(
                train_cmd(ref, "xception", 0, data_root, out=OUT / "v2",
                          tag="ffpp_c40_v2seen"), dry)
        results["B2"] = rc

    # ---- C1: lr sweep, BOTH arms (~1.2 h) ------------------------------
    if "C1" in stages:
        log("=== C1 learning-rate sweep — both arms, 5 epochs ===")
        for lr in LRS:
            for cfg in CONFIGS:
                tag = f"c1_lr{lr:g}"
                rc = sh(train_cmd(ref, cfg, 0, data_root, lr=lr,
                                  epochs=C1_EPOCHS, out=OUT / "c1", tag=tag), dry)
                results[f"C1_{cfg}_{lr:g}"] = rc

    # ---- C2: repeat audit (~2.4 h) -------------------------------------
    if "C2" in stages:
        log("=== C2 repeat audit — 2 in-session repeats, both arms, 15 epochs ===")
        for rep in range(1, C2_REPEATS + 1):
            for cfg in CONFIGS:
                tag = f"c2_rep{rep}"
                rc = sh(train_cmd(ref, cfg, 0, data_root,
                                  out=OUT / "c2", tag=tag), dry)
                results[f"C2_{cfg}_rep{rep}"] = rc

    log("=== SUMMARY ===")
    for k, v in results.items():
        log(f"  {k:28s} {'OK' if v == 0 else 'FAILED rc=' + str(v)}")
    (OUT / "night1_summary.json").write_text(json.dumps(results, indent=2))
    log("wrote", OUT / "night1_summary.json")

if __name__ == "__main__":
    main()
