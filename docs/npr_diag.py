"""Loss-geometry diagnostic — three same-session runs at seed 0, NPR arm only.

Prespecification: docs/npr-loss-geometry-prespecification.md

🚨 DIAGNOSTIC RUNS. They may never enter an arm comparison or an NPR effect
estimate. They write to results_diag/ and preds_diag/ so no analysis glob can
reach them.

The control runs FIRST and is command-identical to the campaign's NPR runs apart
from --out-root and --dataset-name. If the control escapes, the premise that seed
0 collapses deterministically has failed on this instance and neither intervention
can be read causally.

    python docs/npr_diag.py --dry-run     # print the plan, run nothing
    python docs/npr_diag.py               # the three runs
"""
import argparse
import csv
import glob
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

REPO = os.environ.get("DEEPTRACE_REPO", "/kaggle/working/-DeepTrace")
OUT = Path(os.environ.get("DEEPTRACE_DIAG_OUT", "/kaggle/working/results_diag"))
PRED = Path(os.environ.get("DEEPTRACE_DIAG_PRED", "/kaggle/working/preds_diag"))
DS = "ffpp_c40_vid"           # the reference split this must reproduce
CFG = "xception_npr"
SEED = SPLIT_SEED = 0

# Prespecified collapse rule. Accuracy is deliberately absent: the 0.8/0.2 flip is
# an artifact of the optimum sitting exactly on the decision threshold, so an
# accuracy rule would miss a collapse under any condition that moves it.
AUC_MAX = 0.55
SD_MIN = 0.01

# (tag, focal_alpha override, focal_gamma override, best constant prediction)
# The optima are computed in the prespecification with src/losses.FocalLoss.
CONDITIONS = [
    ("ctrl",  None, 2.0, 0.500),
    ("alpha", 0.5,  2.0, 0.627),
    ("gamma", None, 0.0, 0.500),
]
T0 = time.time()


def log(*a):
    print(f"[{(time.time() - T0) / 60:7.1f}m]", *a, flush=True)


def verdict(auc, sd):
    """The prespecified rule, as a pure function so it can be tested.

    Both conditions must hold. An AUC at chance with a *spread* of scores is not
    a constant-prediction collapse — it is a model that ranks no better than
    chance, which is a different failure and must not be reported as this one.
    """
    if auc is None or sd is None:
        return "unknown"
    return "collapsed" if (auc <= AUC_MAX and sd < SD_MIN) else "escaped"


def dump_stats(path):
    """Frame-pooled AUC, sd and mean of prob_fake from a prediction dump."""
    from sklearn.metrics import roc_auc_score
    import statistics as st
    y, p = [], []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            y.append(int(r["label"]))
            p.append(float(r["prob_fake"]))
    if len(set(y)) < 2:
        return None, st.pstdev(p), st.fmean(p), len(p)
    return roc_auc_score(y, p), st.pstdev(p), st.fmean(p), len(p)


def sh(cmd, stream=False):
    if stream:
        rc = subprocess.run(cmd).returncode
        if rc:
            log("FAILED:", " ".join(map(str, cmd)))
        return rc == 0
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        log("FAILED:", " ".join(map(str, cmd)))
        print(r.stdout[-3000:])
        print(r.stderr[-3000:], flush=True)
    return r.returncode == 0


def device_name():
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:                                        # noqa: BLE001
        return "unknown"


def versions(dev):
    v = {"python": platform.python_version(), "platform": platform.platform(),
         "device": dev}
    try:
        import torch
        v["torch"], v["cuda"] = torch.__version__, torch.version.cuda
    except Exception as e:                                   # noqa: BLE001
        v["torch"] = f"unavailable: {e}"
    return v


def discover_crops():
    hits = glob.glob("/kaggle/input/**/ffpp_c40_crops", recursive=True)
    if not hits:
        sys.exit("FATAL: no ffpp_c40_crops under /kaggle/input — attach the "
                 "'c40-run' notebook output.")
    return hits[0]


def build_manifest_once(img, data_root):
    """Generate and verify ONE manifest, then copy it to each condition's
    filename. train.py reads out_root/manifests/{dataset_name}_seed{s}_sz{i}.csv,
    and a copy is provably the same split; regenerating per condition would only
    add ways to differ."""
    import re
    ref = json.load(open(f"{REPO}/results/reference/{DS}_seed{SPLIT_SEED}_test_split.json"))
    dest = OUT / "manifests" / f"{DS}_seed{SPLIT_SEED}_sz{img}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    hits = glob.glob(f"/kaggle/input/**/manifests/{DS}_seed{SPLIT_SEED}_sz{img}.csv",
                     recursive=True)
    if hits:
        shutil.copy(hits[0], dest)
        log("manifest copied from", hits[0])
    else:
        log("no manifest attached — regenerating deterministically")
        from src.data import make_splits, scan_images, write_manifest
        write_manifest(make_splits(scan_images(data_root), seed=SPLIT_SEED,
                                   group_by="videos-([0-9]+)"), dest)
    rows = list(csv.DictReader(open(dest)))
    if not os.path.exists(rows[0]["path"]):
        sys.exit(f"FATAL: manifest points at unmounted crops:\n  {rows[0]['path']}")
    targets = sorted({m.group(1) for r in rows if r["split"] == "test"
                      for m in [re.search(r"videos-(\d+)", r["path"])] if m})
    if targets != ref["test_targets"]:
        sys.exit(f"FATAL: test split does not match the recorded seed-{SPLIT_SEED} "
                 f"split. expected {len(ref['test_targets'])}, got {len(targets)}")
    log(f"split VERIFIED: {len(targets)} test targets match the reference")
    for tag, *_ in CONDITIONS:
        shutil.copy(dest, OUT / "manifests" / f"{DS}_{tag}_seed{SPLIT_SEED}_sz{img}.csv")
    return {s: sum(1 for r in rows if r["split"] == s) for s in ("train", "val", "test")}


def report_campaign_sds():
    """Print sd(prob_fake) for the ten campaign dumps when they are attached, so
    the 0.01 threshold's margin is auditable rather than asserted."""
    hits = sorted(glob.glob("/kaggle/input/**/preds_npr/*_test.csv", recursive=True))
    if not hits:
        log("campaign dumps not attached — skipping the threshold audit")
        return
    log(f"threshold audit over {len(hits)} attached campaign dumps:")
    for h in hits:
        auc, sd, mean, n = dump_stats(h)
        print(f"    {Path(h).name:54s} auc={auc:.4f} sd={sd:.5f} "
              f"mean={mean:.4f}  -> {verdict(auc, sd)}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--session-id", default=None)
    a = ap.parse_args()

    sys.path.insert(0, REPO)
    from src.reference_cmd import build_train_cmd, load_reference

    args = load_reference(f"{REPO}/results/reference/train_args_c40_vid.json")
    img = args["image_size"]
    dev = device_name()
    session_id = a.session_id or f"S-{uuid.uuid4().hex[:8]}"
    log(f"session {session_id}  device {dev}")
    log("DIAGNOSTIC RUNS — never for an arm comparison or an NPR effect estimate")

    plan = []
    for tag, alpha, gamma, p_star in CONDITIONS:
        ov = dict(args, focal_alpha=alpha, focal_gamma=gamma,
                  dataset_name=f"{DS}_{tag}")
        cmd = build_train_cmd(ov, config=CFG, seed=SEED, split_seed=SPLIT_SEED,
                              data_root="<data-root>", out_root=str(OUT))
        plan.append((tag, alpha, gamma, p_star, ov, cmd))

    if a.dry_run:
        for tag, alpha, gamma, p_star, _, cmd in plan:
            print(f"\n{tag}: alpha={alpha} gamma={gamma} best-constant p={p_star}")
            print("   " + " ".join(map(str, cmd[1:])))
        print(f"\ncollapse rule: AUC <= {AUC_MAX} AND sd(prob_fake) < {SD_MIN}")
        return 0

    data_root = discover_crops()
    log("crops:", data_root)
    OUT.mkdir(parents=True, exist_ok=True)
    PRED.mkdir(parents=True, exist_ok=True)
    report_campaign_sds()
    log("splits=%s" % build_manifest_once(img, data_root))

    records, results = {}, []
    for tag, alpha, gamma, p_star, ov, _ in plan:
        name = f"{DS}_{tag}_{CFG}_seed{SEED}"
        log(f"TRAIN {name}  alpha={alpha if alpha is not None else '0.2 (derived)'} "
            f"gamma={gamma}  best-constant p={p_star}")
        t = time.time()
        ok = sh(build_train_cmd(ov, config=CFG, seed=SEED, split_seed=SPLIT_SEED,
                                data_root=data_root, out_root=str(OUT)), stream=True)
        dt = time.time() - t
        log(f"  {name} finished in {dt / 60:.1f} min  ok={ok}")
        if ok:
            sh([sys.executable, "-m", "src.predict", "--run", name,
                "--results-root", str(OUT), "--dataset-name", f"{DS}_{tag}",
                "--seed", str(SEED), "--split-seed", str(SPLIT_SEED),
                "--image-size", str(img), "--out-dir", str(PRED)])
        dump = PRED / f"{name}_test.csv"
        auc = sd = mean = None
        if dump.exists():
            auc, sd, mean, _ = dump_stats(dump)
        v = verdict(auc, sd)
        results.append((tag, alpha, gamma, p_star, auc, sd, mean, v))
        records[name] = {
            "run": name, "tag": tag, "config": CFG, "seed": SEED,
            "focal_alpha": alpha, "focal_gamma": gamma,
            "best_constant_prediction": p_star,
            "session_id": session_id, "duration_s": round(dt, 1), "ok": ok,
            "auc": auc, "sd_prob_fake": sd, "mean_prob_fake": mean,
            "verdict": v, "diagnostic_only": True, **versions(dev)}
        (PRED / "diag_runs.json").write_text(json.dumps(
            sorted(records.values(), key=lambda r: r["run"]), indent=2))

    print("\n== loss-geometry diagnostic ==", flush=True)
    print(f"rule: collapsed iff AUC <= {AUC_MAX} AND sd(prob_fake) < {SD_MIN}\n")
    print(f"{'run':8s} {'alpha':>7s} {'gamma':>6s} {'p*':>6s} {'AUC':>8s} "
          f"{'sd':>9s} {'mean':>8s}  verdict")
    for tag, alpha, gamma, p_star, auc, sd, mean, v in results:
        al = "0.2*" if alpha is None else f"{alpha}"
        f = lambda x, w, d: (f"{x:{w}.{d}f}" if x is not None else " " * (w - 1) + "-")
        print(f"{tag:8s} {al:>7s} {gamma:>6} {p_star:>6.3f} {f(auc,8,4)} "
              f"{f(sd,9,5)} {f(mean,8,4)}  {v}")
    print("\nmean(prob_fake) vs p* is the corroborating check on the arithmetic; "
          "it is NOT part of the collapse call.")
    if results and results[0][7] == "escaped":
        print("\n🚨 THE CONTROL ESCAPED. Seed 0 is not deterministic on this "
              "instance. Neither intervention can be read causally — report the "
              "non-replication (prespecification, outcome 5).", flush=True)
    log(f"manifest -> {PRED / 'diag_runs.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
