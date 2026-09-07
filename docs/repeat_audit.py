"""Repeatability audit — how much does a nominally identical run move?

V8 found that seed 0, rerun in a different session with the same verified split,
the same code and the same hyperparameters, moved the FAD-Xception difference
from -0.0196 to +0.0140. That is n=1: one repeated configuration, so we can say
repeatability was not guaranteed but nothing about the magnitude.

This runs the seed-0 pair `--repeats` times **inside one session**, which buys
more than another across-session repeat would:

  * repeat vs repeat  -> movement with the session held constant. Pure run-level
                         nondeterminism.
  * repeat vs V8      -> movement across sessions.

If those two are similar, there is no session effect to speak of and the V8
observation is ordinary run-level nondeterminism. If the across-session movement
is systematically larger, a session-level component exists. Either answer is
reportable; without the within-session arm we cannot tell them apart.

Each repeat trains under its own `--dataset-name` (…_rep1, _rep2) so run
directories never collide, with the verified manifest copied under each name so
the split is byte-identical every time.

    python docs/repeat_audit.py --repeats 2
"""
import argparse, csv, glob, json, os, re, shutil, subprocess, sys, time
from pathlib import Path

REPO = "/kaggle/working/-DeepTrace"
OUT = Path("/kaggle/working/results")
PRED = Path("/kaggle/working/preds_repeat")
T0 = time.time()


def log(*a):
    print(f"[{(time.time()-T0)/60:7.1f}m]", *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--repo", default=REPO)
    a = ap.parse_args()

    os.chdir(a.repo); sys.path.insert(0, a.repo)
    from src.reference_cmd import build_train_cmd, load_reference

    ARGS = load_reference()
    DS, SPLIT_SEED, IMG = ARGS["dataset_name"], 0, ARGS["image_size"]
    CONFIGS = ARGS["_v8_only"]["configs"]
    REF = json.load(open(f"{a.repo}/results/reference/{DS}_seed{SPLIT_SEED}_test_split.json"))
    OUT.mkdir(parents=True, exist_ok=True); PRED.mkdir(parents=True, exist_ok=True)
    (OUT / "manifests").mkdir(exist_ok=True)

    crops = glob.glob("/kaggle/input/**/ffpp_c40_crops", recursive=True)
    if not crops:
        sys.exit("FATAL: attach the 'c40-run' notebook output (ffpp_c40_crops).")
    data_root = crops[0]
    log("crops:", data_root)

    base = OUT / "manifests" / f"{DS}_seed{SPLIT_SEED}_sz{IMG}.csv"
    if not base.exists():
        log("regenerating the split deterministically from crops")
        from src.data import make_splits, scan_images, write_manifest
        write_manifest(make_splits(scan_images(data_root), seed=SPLIT_SEED,
                                   group_by=ARGS["group_by"]), base)
    rows = list(csv.DictReader(open(base)))
    tt = sorted({m.group(1) for r in rows if r["split"] == "test"
                 for m in [re.search(r"videos-(\d+)", r["path"])] if m})
    if tt != REF["test_targets"]:
        sys.exit("FATAL: split does not match the recorded seed-0 split.")
    log(f"split VERIFIED: {len(tt)} test targets match")

    # Freeze the conditions in an artifact, not in anyone's memory. A later diff
    # of two of these answers "genuine nondeterminism or an unnoticed difference?"
    # -- the question the --amp incident could not be answered from artifacts.
    from src.runenv import capture
    env = capture("repeat_audit", manifest=base, probe_config=CONFIGS[0],
                  image_size=IMG, seed=SPLIT_SEED, repo=a.repo,
                  extra={"repeats": a.repeats, "configs": CONFIGS,
                         "split_seed": SPLIT_SEED, "train_seed": SPLIT_SEED,
                         "reference_args": ARGS})
    (Path(a.repo) / "results" / "reference" / "runenv_repeat_audit.json").write_text(
        json.dumps(env, indent=2))
    t = env["torch"]
    log(f"env: {t.get('gpus')} torch={t.get('torch')} cuda={t.get('cuda')} "
        f"cudnn={t.get('cudnn')}")
    log(f"     deterministic={t.get('cudnn_deterministic')} "
        f"benchmark={t.get('cudnn_benchmark')} "
        f"tf32(cudnn/matmul)={t.get('cudnn_allow_tf32')}/{t.get('matmul_allow_tf32')}")
    log(f"     manifest sha256 {env['manifest']['sha256'][:16]}")
    log(f"     init probe sha256 {str(env['init_probe']['param_sha256'])[:16]}")

    for rep in range(1, a.repeats + 1):
        ds = f"{DS}_rep{rep}"
        shutil.copy(base, OUT / "manifests" / f"{ds}_seed{SPLIT_SEED}_sz{IMG}.csv")
        for cfg in CONFIGS:
            cmd = build_train_cmd(ARGS, cfg, SPLIT_SEED, SPLIT_SEED,
                                  data_root, str(OUT), dataset_name=ds)
            log(f"TRAIN rep{rep} {cfg}")
            if rep == 1 and cfg == CONFIGS[0]:
                log("   " + " ".join(str(x) for x in cmd[2:]))
            if subprocess.run(cmd).returncode:
                log(f"FAILED rep{rep} {cfg}")
        for cfg in CONFIGS:
            run = f"{ds}_{cfg}_seed{SPLIT_SEED}"
            if (OUT / run / "best.pt").exists():
                subprocess.run([sys.executable, "-m", "src.predict", "--run", run,
                                "--results-root", str(OUT), "--dataset-name", ds,
                                "--seed", str(SPLIT_SEED), "--split-seed", str(SPLIT_SEED),
                                "--image-size", str(IMG), "--out-dir", str(PRED)],
                               capture_output=True)
                log(f"   pred OK {run}")

    # Hash the prediction dumps: identical hashes would mean identical outputs,
    # which is the strongest possible evidence and has not been observed yet.
    from src.runenv import hash_file
    hashes = {Path(f).name: hash_file(f) for f in sorted(glob.glob(str(PRED / "*.csv")))}
    (Path(a.repo) / "results" / "reference" / "runenv_repeat_pred_hashes.json").write_text(
        json.dumps(hashes, indent=2))
    log("prediction dump hashes:")
    for k, v in hashes.items():
        log(f"   {v[:16]}  {k}")

    if (OUT / "summary.csv").exists():
        print("\n=== summary.csv ===", flush=True)
        print(open(OUT / "summary.csv").read(), flush=True)
    log("DONE — package /kaggle/working/results and /kaggle/working/preds_repeat")


if __name__ == "__main__":
    main()
