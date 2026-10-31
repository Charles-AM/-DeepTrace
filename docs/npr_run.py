"""NPR supplementary campaign — ten runs, counterbalanced, one session.

Prespecification: docs/npr-supplementary-prespecification.md
Campaign constants: results/reference/npr_campaign.json
Hyperparameters: results/reference/train_args_c40_vid.json (the file V8 used)

Differs from docs/v8_run.py in three ways that matter:

1. **Both arms are retrained.** The committed V8 xception runs are NOT the
   baseline. A nominally identical seed-0 configuration moved 0.0336 across
   sessions and flipped sign, with the two arms moving in opposite directions,
   so a cross-session pairing does not cancel.
2. **Order is counterbalanced** from the committed table. V8 trained xception
   first for every seed, which is the order confound this campaign must avoid.
3. **A run manifest is written** recording session, order, device and versions,
   because the prespecified session rule is otherwise unverifiable.

It deliberately does NOT run the bootstrap. `src.campaign_gate` must pass first.

    python docs/npr_run.py --time-one-epoch   # the prespecified timing gate
    python docs/npr_run.py --dry-run          # print the plan, run nothing
    python docs/npr_run.py                    # the campaign
"""
import argparse
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
OUT = Path(os.environ.get("DEEPTRACE_OUT", "/kaggle/working/results"))
PRED = Path(os.environ.get("DEEPTRACE_PRED", "/kaggle/working/preds_npr"))
BUDGET_S = 10.5 * 3600
# Until an NPR epoch is timed this is a guess, and it is used only to decide
# whether a seed PAIR still fits in the session. Overestimating is the safe
# direction: it stops early rather than orphaning half a pair.
INITIAL_RUN_S = 75 * 60
T0 = time.time()


def log(*a):
    print(f"[{(time.time() - T0) / 60:7.1f}m]", *a, flush=True)


def versions(device_name):
    v = {"python": platform.python_version(), "platform": platform.platform(),
         "device": device_name}
    try:
        import torch
        v["torch"] = torch.__version__
        v["cuda"] = torch.version.cuda
    except Exception as e:                                   # noqa: BLE001
        v["torch"] = f"unavailable: {e}"
    try:
        import timm
        v["timm"] = timm.__version__
    except Exception as e:                                   # noqa: BLE001
        v["timm"] = f"unavailable: {e}"
    try:
        v["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=REPO).stdout.strip()
    except Exception as e:                                   # noqa: BLE001
        v["git_commit"] = f"unavailable: {e}"
    return v


def device_name():
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:                                        # noqa: BLE001
        return "unknown"


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


def discover_crops():
    hits = glob.glob("/kaggle/input/**/ffpp_c40_crops", recursive=True)
    if not hits:
        sys.exit("FATAL: no ffpp_c40_crops under /kaggle/input — attach the "
                 "'c40-run' notebook output.")
    return hits[0]


def verify_split(ds, split_seed, img, data_root):
    """Same verification V8 used. A regenerated manifest that quietly differs
    would make this campaign incomparable with everything else at c40."""
    import csv
    import re
    ref = json.load(open(f"{REPO}/results/reference/{ds}_seed{split_seed}_test_split.json"))
    dest = OUT / "manifests" / f"{ds}_seed{split_seed}_sz{img}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    hits = glob.glob(f"/kaggle/input/**/manifests/{ds}_seed{split_seed}_sz{img}.csv",
                     recursive=True)
    if hits:
        shutil.copy(hits[0], dest)
        log("manifest copied from", hits[0])
    else:
        log("no manifest attached — regenerating deterministically")
        from src.data import make_splits, scan_images, write_manifest
        write_manifest(make_splits(scan_images(data_root), seed=split_seed,
                                   group_by="videos-([0-9]+)"), dest)
    rows = list(csv.DictReader(open(dest)))
    if not os.path.exists(rows[0]["path"]):
        sys.exit(f"FATAL: manifest points at unmounted crops:\n  {rows[0]['path']}")
    targets = sorted({m.group(1) for r in rows if r["split"] == "test"
                      for m in [re.search(r"videos-(\d+)", r["path"])] if m})
    if targets != ref["test_targets"]:
        sys.exit(f"FATAL: test split does not match the recorded seed-{split_seed} "
                 f"split. expected {len(ref['test_targets'])}, got {len(targets)}\n"
                 f"  missing: {sorted(set(ref['test_targets']) - set(targets))[:8]}\n"
                 f"  extra:   {sorted(set(targets) - set(ref['test_targets']))[:8]}")
    log(f"split VERIFIED: {len(targets)} test targets match the reference")
    return {s: sum(1 for r in rows if r["split"] == s) for s in ("train", "val", "test")}


def run_name_for(ds, cfg, seed, split_seed):
    # Matches the V8 naming convention exactly, including the quirk that the run
    # whose seed equals split_seed carries no suffix.
    return f"{ds}_{cfg}_seed{seed}" + (f"_split{split_seed}" if seed != split_seed else "")


def load_manifest(path):
    if path.exists():
        try:
            return {r["run"]: r for r in json.loads(path.read_text())}
        except Exception as e:                               # noqa: BLE001
            log(f"WARNING: could not parse {path} ({e}); starting a new manifest")
    return {}


def write_manifest(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(records.values(),
                                      key=lambda r: (r["seed"], r["run"])), indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    ap.add_argument("--time-one-epoch", action="store_true",
                    help="the prespecified timing gate: one NPR epoch, then stop")
    ap.add_argument("--session-id", default=None, help="override the generated id")
    ap.add_argument("--budget-hours", type=float, default=BUDGET_S / 3600)
    a = ap.parse_args()

    os.chdir(REPO)
    sys.path.insert(0, REPO)
    from src.reference_cmd import build_train_cmd, load_reference

    args = load_reference()
    camp = json.load(open("results/reference/npr_campaign.json"))
    ds, img = args["dataset_name"], args["image_size"]
    split_seed, seeds = camp["split_seed"], camp["seeds"]
    cfg_a, cfg_b = camp["configs"]["a"], camp["configs"]["b"]
    order = camp["order"]
    budget = a.budget_hours * 3600

    # One session id per invocation. If the session dies and the script reruns,
    # a new id appears, so a pair finished across two sessions is detectable by
    # src.campaign_gate rather than silently accepted.
    session_id = a.session_id or f"S-{uuid.uuid4().hex[:8]}"
    dev = device_name()
    log(f"session {session_id}  device {dev}")
    log(f"order table (committed): " +
        "  ".join(f"seed{s}={order[str(s)]}" for s in seeds))

    if a.dry_run:
        log("DRY RUN — commands below are not executed")
        for seed in seeds:
            first, second = ((cfg_a, cfg_b) if order[str(seed)] == "a_first"
                             else (cfg_b, cfg_a))
            for cfg in (first, second):
                cmd = build_train_cmd(args, config=cfg, seed=seed,
                                      split_seed=split_seed, data_root="<crops>",
                                      out_root=str(OUT))
                log(f"  seed{seed} {cfg:13s} {' '.join(map(str, cmd[2:]))}")
        return 0

    data_root = discover_crops()
    log("crops:", data_root)
    counts = verify_split(ds, split_seed, img, data_root)
    log(f"splits={counts}")

    if a.time_one_epoch:
        # The prespecified gate. A distinct dataset-name and out-root keep this
        # out of the campaign's results entirely.
        timing_args = dict(args)
        timing_args["epochs"] = 1
        tmp = OUT.parent / "npr_timing"
        cmd = build_train_cmd(timing_args, config=cfg_b, seed=0,
                              split_seed=split_seed, data_root=data_root,
                              out_root=str(tmp), dataset_name=f"{ds}_npr_timing")
        log("TIMING one NPR epoch (excluded from the campaign):")
        log("   " + " ".join(map(str, cmd[2:])))
        t = time.time()
        ok = sh(cmd, stream=True)
        dt = time.time() - t
        log(f"one epoch (plus setup) took {dt / 60:.1f} min -> "
            f"{args['epochs']} epochs ~= {dt * args['epochs'] / 3600:.1f} h per run, "
            f"10 runs ~= {dt * args['epochs'] * 10 / 3600:.1f} GPU-h")
        log("Compare against the 4-6 GPU-h estimate in the prespecification and "
            "record the measured figure before launching the campaign.")
        return 0 if ok else 1

    log("effective training command (from results/reference/train_args_c40_vid.json):")
    log("   " + " ".join(map(str, build_train_cmd(
        args, config=cfg_a, seed=0, split_seed=split_seed,
        data_root=data_root, out_root=str(OUT))[2:])))

    PRED.mkdir(parents=True, exist_ok=True)
    man_path = PRED / "runs.json"
    records = load_manifest(man_path)
    durations = []
    done, failed, skipped_pairs = [], [], []

    for seed in seeds:
        first, second = ((cfg_a, cfg_b) if order[str(seed)] == "a_first"
                         else (cfg_b, cfg_a))
        pair = [run_name_for(ds, c, seed, split_seed) for c in (first, second)]
        pending = [n for n in pair if not (OUT / n / "best.pt").exists()]

        # Session rule, enforced BEFORE starting: never begin a pair that the
        # remaining budget cannot finish, or the campaign orphans half of it.
        per_run = (sum(durations) / len(durations)) if durations else INITIAL_RUN_S
        need = len(pending) * per_run * 1.15
        left = budget - (time.time() - T0)
        if pending and need > left:
            log(f"SESSION BUDGET — stopping before seed {seed}: needs "
                f"{need / 3600:.1f} h, {left / 3600:.1f} h left. Not starting a "
                f"pair this session cannot finish.")
            skipped_pairs.append(seed)
            break

        for cfg, name in zip((first, second), pair):
            if (OUT / name / "best.pt").exists():
                log("skip (exists):", name)
                done.append(name)
                if name not in records:
                    # Pre-existing from an earlier session with no record. Give it
                    # a unique id so the gate's single-session check FAILS rather
                    # than treating unknowns as one session.
                    records[name] = {"run": name, "seed": seed, "config": cfg,
                                     "session_id": f"UNKNOWN-{name}",
                                     "model_order": order[str(seed)],
                                     "note": "pre-existing checkpoint, session "
                                             "unknown; excluded by the gate"}
                continue
            log(f"TRAIN {name}  ({order[str(seed)]}, "
                f"{'first' if cfg == first else 'second'} in pair)")
            t = time.time()
            ok = sh(build_train_cmd(args, config=cfg, seed=seed,
                                    split_seed=split_seed, data_root=data_root,
                                    out_root=str(OUT)), stream=True)
            dt = time.time() - t
            durations.append(dt)
            (done if ok else failed).append(name)
            records[name] = {
                "run": name, "seed": seed, "config": cfg,
                "session_id": session_id, "model_order": order[str(seed)],
                "position_in_pair": "first" if cfg == first else "second",
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                             time.gmtime(t)),
                "duration_s": round(dt, 1),
                "checkpoint": str(OUT / name / "best.pt"),
                "ok": ok, **versions(dev)}
            write_manifest(man_path, records)
            log(f"  {name} finished in {dt / 60:.1f} min")

        for cfg, name in zip((first, second), pair):
            if not (OUT / name / "best.pt").exists():
                continue
            ok = sh([sys.executable, "-m", "src.predict", "--run", name,
                     "--results-root", str(OUT), "--dataset-name", ds,
                     "--seed", str(seed), "--split-seed", str(split_seed),
                     "--image-size", str(img), "--out-dir", str(PRED)])
            log(("  pred OK " if ok else "  pred FAILED "), name)

    write_manifest(man_path, records)
    log(f"trained/kept {len(done)}, failed {len(failed)}: {failed}")
    if skipped_pairs:
        log(f"seeds not attempted this session: {skipped_pairs} — rerun BOTH arms "
            f"for each in a new session, per the session rule.")
    log(f"manifest -> {man_path}")

    # The gate, not the bootstrap. Nothing may be estimated until this passes.
    log("running the pre-analysis gate")
    sh([sys.executable, "-m", "src.campaign_gate",
        "--a-glob", str(PRED / f"*_{cfg_a}_seed*_test.csv"),
        "--b-glob", str(PRED / f"*_{cfg_b}_seed*_test.csv"),
        "--dir", str(PRED), "--manifest", str(man_path),
        "--a-config", cfg_a, "--b-config", cfg_b], stream=True)
    log("DONE — package results and preds_npr. Run the bootstrap only if the "
        "gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
