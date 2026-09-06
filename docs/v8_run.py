"""V8 — fixed split, varying training seed. Written for Kaggle Save & Run All.

Runs unattended: discovers its inputs, aborts immediately if they are wrong,
interleaves configs so a partial session still yields complete pairs, dumps
predictions after each pair, and stops launching work in time to package results.
"""
import csv, glob, os, shutil, subprocess, sys, time
from pathlib import Path

REPO = "/kaggle/working/-DeepTrace"
OUT = Path("/kaggle/working/results")
PRED = Path("/kaggle/working/preds")
DS, SPLIT_SEED, SEEDS = "ffpp_c40_vid", 0, [0, 1, 2, 3, 4]
CONFIGS = ["xception", "xception_fad"]
EPOCHS, IMG = 15, 128
BUDGET_S = 10.5 * 3600          # stop launching new runs after this
T0 = time.time()

def log(*a):
    print(f"[{(time.time()-T0)/60:7.1f}m]", *a, flush=True)

os.chdir(REPO)
sys.path.insert(0, REPO)
OUT.mkdir(parents=True, exist_ok=True); PRED.mkdir(parents=True, exist_ok=True)
(OUT / "manifests").mkdir(exist_ok=True)

# ---------------------------------------------------------------- discovery
hits = glob.glob(f"/kaggle/input/**/manifests/{DS}_seed{SPLIT_SEED}_sz{IMG}.csv",
                 recursive=True)
if not hits:
    sys.exit(f"FATAL: no manifest {DS}_seed{SPLIT_SEED}_sz{IMG}.csv under /kaggle/input "
             "— attach the c40 video-level notebook output.")
src_manifest = hits[0]
log("manifest:", src_manifest)

rows = list(csv.DictReader(open(src_manifest)))
first = rows[0]["path"]
if not os.path.exists(first):
    sys.exit(f"FATAL: manifest points at crops that are not mounted:\n  {first}\n"
             "Attach the c40 crops dataset so paths resolve as they did originally.")
# Only needed as a required CLI arg: build_dataloaders reuses the manifest when
# it exists, so data_root is never scanned. Derived defensively all the same.
parts = Path(first).parts
try:
    data_root = str(Path(*parts[:parts.index("ffpp_c40_crops") + 1]))
except ValueError:
    data_root = str(Path(first).parent.parent)
counts = {s: sum(1 for r in rows if r["split"] == s) for s in ("train", "val", "test")}
log(f"crops OK  splits={counts}  data_root={data_root}")

# Copy rather than regenerate: byte-identical to the split the comparison runs used.
shutil.copy(src_manifest, OUT / "manifests" / Path(src_manifest).name)
log("manifest copied — split is frozen and identical to the c40 seed-0 runs")

# ---------------------------------------------------------------- run loop
def sh(cmd, stream=False):
    """stream=True for training: a 6-hour batch log that prints nothing until each
    run ends is unreadable, and a hung run would be indistinguishable from a slow
    one. Prediction calls are short, so those stay captured."""
    if stream:
        r = subprocess.run(cmd)
        if r.returncode:
            log("FAILED:", " ".join(cmd))
        return r.returncode == 0, ""
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        log("FAILED:", " ".join(cmd)); print(r.stdout[-3000:]); print(r.stderr[-3000:], flush=True)
    return r.returncode == 0, r.stdout

done, failed = [], []
for seed in SEEDS:                      # interleaved: both configs per seed
    if time.time() - T0 > BUDGET_S:
        log(f"BUDGET REACHED — stopping before seed {seed}"); break
    for cfg in CONFIGS:
        run_name = f"{DS}_{cfg}_seed{seed}"
        if seed != SPLIT_SEED:
            run_name += f"_split{SPLIT_SEED}"
        if (OUT / run_name / "best.pt").exists():
            log("skip (exists):", run_name); done.append(run_name); continue
        log("TRAIN", run_name)
        ok, _ = sh([sys.executable, "-m", "src.train",
                    "--data-root", data_root, "--config", cfg,
                    "--dataset-name", DS, "--epochs", str(EPOCHS),
                    "--image-size", str(IMG), "--seed", str(seed),
                    "--split-seed", str(SPLIT_SEED),
                    "--group-by", "videos-([0-9]+)", "--amp",
                    "--num-workers", "2", "--out-root", str(OUT)], stream=True)
        (done if ok else failed).append(run_name)

    # dump predictions for this seed's pair while the data is fresh
    for cfg in CONFIGS:
        run_name = f"{DS}_{cfg}_seed{seed}"
        if seed != SPLIT_SEED:
            run_name += f"_split{SPLIT_SEED}"
        if not (OUT / run_name / "best.pt").exists():
            continue
        ok, _ = sh([sys.executable, "-m", "src.predict", "--run", run_name,
                    "--results-root", str(OUT), "--dataset-name", DS,
                    "--seed", str(seed), "--split-seed", str(SPLIT_SEED),
                    "--image-size", str(IMG), "--out-dir", str(PRED)])
        log(("  pred OK " if ok else "  pred FAILED "), run_name)

log(f"trained {len(done)}, failed {len(failed)}: {failed}")

# ---------------------------------------------------------------- analysis
for seed in SEEDS:
    sfx = f"_split{SPLIT_SEED}" if seed != SPLIT_SEED else ""
    a = PRED / f"{DS}_xception_seed{seed}{sfx}_test.csv"
    b = PRED / f"{DS}_xception_fad_seed{seed}{sfx}_test.csv"
    if a.exists() and b.exists():
        sh([sys.executable, "-m", "src.cluster_boot", "--a", str(a), "--b", str(b),
            "--margins", "0.005", "0.010", "0.014", "0.020",
            "--out-dir", str(OUT / "analysis" / "v8")])

if (OUT / "summary.csv").exists():
    print("\n=== summary.csv ===", flush=True)
    print(open(OUT / "summary.csv").read()[-4000:], flush=True)
log("DONE — package /kaggle/working/results and /kaggle/working/preds")
