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
import argparse, csv, glob, json, os, re, shutil, subprocess, sys, time
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


def resolve_manifest(data_root, dry):
    """Locate or regenerate the seed-0 manifest, then VERIFY it.

    Manifests are not tracked in git (zero under results/manifests/), so a fresh
    clone has none. make_splits is deterministic given (items, seed, group_by), so
    it can be regenerated from the same crops. Either way the result is checked
    against the recorded seed-0 test split -- a partition that quietly differed
    would make every comparison here incomparable with the runs it is set beside,
    which is the one failure that invalidates the experiment while raising no
    error at all.
    """
    dest = OUT / "manifests" / f"{DS}_seed{SPLIT_SEED}_sz128.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dry:
        log("manifest: <DRY-RUN - would resolve and verify>")
        return dest
    hits = glob.glob(f"/kaggle/input/**/manifests/{DS}_seed{SPLIT_SEED}_sz128.csv",
                     recursive=True)
    local = REPO / f"results/manifests/{DS}_seed{SPLIT_SEED}_sz128.csv"
    if hits:
        shutil.copy(hits[0], dest); log("manifest copied from", hits[0])
    elif local.exists():
        shutil.copy(local, dest); log("manifest copied from repo", local)
    else:
        log("no manifest found - regenerating deterministically from crops")
        sys.path.insert(0, str(REPO))
        from src.data import make_splits, scan_images, write_manifest
        write_manifest(make_splits(scan_images(data_root), seed=SPLIT_SEED,
                                   group_by="videos-([0-9]+)"), dest)
        log("manifest regenerated ->", dest)

    rows = list(csv.DictReader(open(dest)))
    if not rows:
        sys.exit("FATAL: manifest is empty.")
    if not os.path.exists(rows[0]["path"]):
        sys.exit(f"FATAL: manifest points at crops that are not mounted:\n  {rows[0]['path']}")
    if not SPLIT_REF.exists():
        sys.exit(f"FATAL: split reference missing: {SPLIT_REF}")
    ref = json.loads(SPLIT_REF.read_text())
    got = sorted({m.group(1) for r in rows if r["split"] == "test"
                  for m in [re.search(r"videos-(\d+)", r["path"])] if m})
    if got != ref["test_targets"]:
        sys.exit("FATAL: test split does not match the recorded seed-0 split.\n"
                 f"  expected {len(ref['test_targets'])} targets, got {len(got)}\n"
                 f"  missing: {sorted(set(ref['test_targets']) - set(got))[:8]}\n"
                 f"  extra:   {sorted(set(got) - set(ref['test_targets']))[:8]}")
    log(f"split VERIFIED against reference: {len(got)} test targets match")
    counts = {sp: sum(1 for r in rows if r["split"] == sp) for sp in ("train","val","test")}
    log("splits:", counts)
    return dest


def verify_v2_manifests(dry):
    """Abort before training if train.py would not consume the V2 manifests.

    On 2026-09-18 this exact check was absent: seen_unseen wrote its manifests to
    OUT/manifests/ while training ran with --out-root OUT/v2, so train.py found
    nothing, silently regenerated a default split, and trained 80 minutes on the
    wrong data while reporting entirely plausible numbers.

    The guard reconstructs the path train.py WILL resolve and checks the file
    there is the one seen_unseen produced, by its split shape. Under Save & Run All
    nobody is watching the train= line, so the script has to look instead.
    """
    if dry:
        log("v2 manifest guard: <DRY-RUN>")
        return
    expect_test = 750          # seen_unseen: 150 videos x 5 crops
    for tag in ("ffpp_c40_v2seen", "ffpp_c40_v2unseen"):
        path = OUT / "manifests" / f"{tag}_seed{SPLIT_SEED}_sz128.csv"
        if not path.exists():
            sys.exit(f"FATAL: {path} missing. train.py resolves its manifest as "
                     f"out_root/manifests/<dataset-name>_seed<n>_sz<n>.csv, so it "
                     f"would regenerate a DEFAULT split and train on the wrong data.")
        rows = list(csv.DictReader(open(path)))
        counts = {sp: sum(1 for r in rows if r["split"] == sp)
                  for sp in ("train", "val", "test")}
        if counts["test"] != expect_test:
            sys.exit(f"FATAL: {tag} test split is {counts['test']}, expected "
                     f"{expect_test}. This is not the V2 manifest.")
        if tag == "ffpp_c40_v2seen" and counts["train"] >= 24000:
            sys.exit(f"FATAL: {tag} has {counts['train']} training crops. The seen "
                     f"manifest holds crops back, so it must be < 24000. A full "
                     f"24000 means a default split was regenerated -- the exact "
                     f"failure of 2026-09-18.")
        log(f"v2 manifest VERIFIED  {tag}: {counts}")


def dump_predictions(run_name, tag, dry, out=None):
    """Write the per-prediction CSV for a finished run.

    train.py saves a per-run JSON and best.pt but NOT per-frame scores; predict.py
    is a separate step. Every run before 2026-09-18 that skipped it is stuck at
    aggregate level and cannot be re-analysed -- the exact hole that leaves the L1
    column unable to carry a cluster bootstrap. A dump is ~200 KB against an 83 MB
    checkpoint, so there is no reason ever to skip it.
    """
    return sh([sys.executable, "-m", "src.predict",
               "--run", run_name, "--results-root", str(out or OUT),
               "--dataset-name", tag, "--seed", "0", "--split-seed", str(SPLIT_SEED),
               "--split", "test",
               "--out-dir", str(OUT / "predictions" / run_name)], dry)


def verify_distinct_dumps(dry):
    """Confirm the two B2 scorings produced DISTINCT files.

    predict.py names its output from --run, not --dataset-name, so writing both
    scorings to one directory silently overwrites the first. On 2026-09-18 the
    driver reported OK for both calls while only one file survived: a zero return
    code records that a process finished, not that it produced a distinct artifact.
    """
    if dry:
        log("distinct-dump check: <DRY-RUN>")
        return
    import hashlib
    seen = OUT / "v2_predictions" / "ffpp_c40_v2seen" / "ffpp_c40_v2seen_xception_seed0_test.csv"
    unseen = OUT / "v2_predictions" / "ffpp_c40_v2unseen" / "ffpp_c40_v2seen_xception_seed0_test.csv"
    for p in (seen, unseen):
        if not p.exists():
            sys.exit(f"FATAL: expected dump missing: {p}")
    h = [hashlib.sha256(p.read_bytes()).hexdigest() for p in (seen, unseen)]
    if h[0] == h[1]:
        sys.exit("FATAL: the seen and unseen dumps are byte-identical. One "
                 "overwrote the other, or both scored the same manifest.")
    log(f"distinct dumps VERIFIED  seen {h[0][:12]}  unseen {h[1][:12]}")

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
    manifest = resolve_manifest(data_root, dry)
    results = {}

    # ---- B2: V2 seen/unseen (~80 min) ---------------------------------
    if "B2" in stages:
        log("=== B2 V2 seen vs unseen ===")
        cmd = [sys.executable, "-m", "src.seen_unseen",
               "--manifest", str(manifest), "--out-root", str(OUT), "--seed", "0"]
        rc = sh(cmd, dry)
        results["B2_manifests"] = rc
        if rc == 0:
            verify_v2_manifests(dry)
            # Train ONE model on the SEEN manifest.
            # out-root MUST be OUT, not OUT/v2: train.py resolves its manifest as
            # out_root/manifests/<dataset-name>_seed<n>_sz<n>.csv, and seen_unseen
            # wrote both manifests under OUT/manifests/. Pointing elsewhere makes
            # train.py regenerate a DEFAULT split and silently train on the wrong
            # data -- which is exactly what happened on 2026-09-18 (the seen
            # manifest has 23250 train crops; the run reported 24000).
            run_name = "ffpp_c40_v2seen_xception_seed0"
            results["B2_train"] = sh(
                train_cmd(ref, "xception", 0, data_root, out=OUT,
                          tag="ffpp_c40_v2seen"), dry)
            # Score that ONE fixed model against BOTH manifests. This is the
            # experiment; training alone produces no leakage estimate.
            # predict.py names its output <run>_<split>.csv, which is IDENTICAL for
            # both calls -- same run, same split, only the manifest differs. Writing
            # both to one directory silently overwrites the first (2026-09-18). Give
            # each its own directory so the two dumps survive.
            for tag in ("ffpp_c40_v2seen", "ffpp_c40_v2unseen"):
                results[f"B2_score_{tag}"] = sh(
                    [sys.executable, "-m", "src.predict",
                     "--run", run_name, "--results-root", str(OUT),
                     "--dataset-name", tag, "--seed", "0", "--split-seed", "0",
                     "--split", "test",
                     "--out-dir", str(OUT / "v2_predictions" / tag)],
                    dry)
            verify_distinct_dumps(dry)

    # ---- B1: band_ablation smoke test (~5 min) -------------------------
    # Runs AFTER B2 by necessity: it re-scores a trained model with frequency
    # bands masked, so it needs a checkpoint. No .pt files are committed, so in a
    # fresh container the only checkpoint available is the one B2 just produced.
    if "B1" in stages:
        log("=== B1 band_ablation smoke (uses B2's checkpoint) ===")
        cmd = [sys.executable, "-m", "src.band_ablation",
               "--runs", "ffpp_c40_v2seen_xception_seed0",
               "--results-root", str(OUT), "--dataset-name", "ffpp_c40_v2seen",
               "--seed", str(SPLIT_SEED), "--limit", "200",
               "--out-dir", str(OUT / "band_ablation")]
        results["B1"] = sh(cmd, dry)

    # ---- C1: lr sweep, BOTH arms (~1.2 h) ------------------------------
    if "C1" in stages:
        log("=== C1 learning-rate sweep — both arms, 5 epochs ===")
        for lr in LRS:
            for cfg in CONFIGS:
                tag = f"c1_lr{lr:g}"
                rc = sh(train_cmd(ref, cfg, 0, data_root, lr=lr,
                                  epochs=C1_EPOCHS, out=OUT / "c1", tag=tag), dry)
                results[f"C1_{cfg}_{lr:g}"] = rc
                if rc == 0:
                    results[f"C1_dump_{cfg}_{lr:g}"] = dump_predictions(
                        f"{tag}_{cfg}_seed0", tag, dry, out=OUT / "c1")

    # ---- C2: repeat audit (~2.4 h) -------------------------------------
    if "C2" in stages:
        log("=== C2 repeat audit — 2 in-session repeats, both arms, 15 epochs ===")
        for rep in range(1, C2_REPEATS + 1):
            for cfg in CONFIGS:
                tag = f"c2_rep{rep}"
                rc = sh(train_cmd(ref, cfg, 0, data_root,
                                  out=OUT / "c2", tag=tag), dry)
                results[f"C2_{cfg}_rep{rep}"] = rc
                if rc == 0:
                    results[f"C2_dump_{cfg}_rep{rep}"] = dump_predictions(
                        f"{tag}_{cfg}_seed0", tag, dry, out=OUT / "c2")

    log("=== SUMMARY ===")
    for k, v in results.items():
        if dry:
            status = "DRY-RUN (not executed)"
        else:
            status = "OK" if v == 0 else "FAILED rc=" + str(v)
        log(f"  {k:28s} {status}")
    (OUT / "night1_summary.json").write_text(json.dumps(results, indent=2))
    log("wrote", OUT / "night1_summary.json")

if __name__ == "__main__":
    main()
