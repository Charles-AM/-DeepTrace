"""V8 — fixed split, varying training seed. Written for Kaggle Save & Run All.

Runs unattended: discovers its inputs, aborts immediately if they are wrong,
interleaves configs so a partial session still yields complete pairs, dumps
predictions after each pair, and stops launching work in time to package results.
"""
import csv, glob, json, os, re, shutil, subprocess, sys, time
from pathlib import Path

REPO = "/kaggle/working/-DeepTrace"
OUT = Path("/kaggle/working/results")
PRED = Path("/kaggle/working/preds")
# Every training hyperparameter comes from the reference file, which records the
# EFFECTIVE config of the varying-split runs this experiment is compared against.
# Nothing here is typed twice: --amp cost a 155-minute run precisely because a
# flag lived in this script and nowhere else.
ARGS = json.load(open("results/reference/train_args_c40_vid.json"))
DS = ARGS["dataset_name"]
SPLIT_SEED = ARGS["_v8_only"]["split_seed"]
SEEDS = ARGS["_v8_only"]["seeds"]
CONFIGS = ARGS["_v8_only"]["configs"]
IMG = ARGS["image_size"]
BUDGET_S = 10.5 * 3600          # stop launching new runs after this
# Seed 0 doubles as a reproducibility control: split_seed=0/seed=0 is nominally
# the same configuration as the varying-split seed-0 run, so its AUCs should
# reproduce (xception 0.81347, +FAD 0.79391). If they do, cross-session
# environment drift is negligible and the sd comparison is sound; if they do
# not, the decomposition cannot be done across sessions and we report that.
T0 = time.time()

def log(*a):
    print(f"[{(time.time()-T0)/60:7.1f}m]", *a, flush=True)

os.chdir(REPO)
sys.path.insert(0, REPO)
OUT.mkdir(parents=True, exist_ok=True); PRED.mkdir(parents=True, exist_ok=True)
(OUT / "manifests").mkdir(exist_ok=True)

# ---------------------------------------------------------------- discovery
REF = json.load(open(f"{REPO}/results/reference/{DS}_seed{SPLIT_SEED}_test_split.json"))
DEST = OUT / "manifests" / f"{DS}_seed{SPLIT_SEED}_sz{IMG}.csv"

crops = glob.glob("/kaggle/input/**/ffpp_c40_crops", recursive=True)
if not crops:
    sys.exit("FATAL: no ffpp_c40_crops directory under /kaggle/input — attach the "
             "'c40-run' notebook output, which contains the c40 crops.")
data_root = crops[0]
log("crops:", data_root)

# The manifest is preferred but not required: make_splits is deterministic given
# (items, seed, group_by), so it can be regenerated from the same crops. Either
# way the result is VERIFIED against the recorded seed-0 test split below, so a
# silent partition change cannot slip through.
hits = glob.glob(f"/kaggle/input/**/manifests/{DS}_seed{SPLIT_SEED}_sz{IMG}.csv",
                 recursive=True)
if hits:
    shutil.copy(hits[0], DEST)
    log("manifest copied from", hits[0])
else:
    log("no manifest attached — regenerating deterministically from crops")
    sys.path.insert(0, REPO)
    from src.data import make_splits, scan_images, write_manifest
    write_manifest(make_splits(scan_images(data_root), seed=SPLIT_SEED,
                               group_by="videos-([0-9]+)"), DEST)
    log("manifest regenerated ->", DEST)

rows = list(csv.DictReader(open(DEST)))
first = rows[0]["path"]
if not os.path.exists(first):
    sys.exit(f"FATAL: manifest points at crops that are not mounted:\n  {first}")

# Verify the split is the one every prior c40 result used. A regenerated or
# copied manifest that quietly differs would make V8 incomparable with the
# varying-split runs it is subtracted from -- the single failure that would
# invalidate the whole experiment without any error being raised.
test_targets = sorted({m.group(1) for r in rows if r["split"] == "test"
                       for m in [re.search(r"videos-(\d+)", r["path"])] if m})
if test_targets != REF["test_targets"]:
    sys.exit(f"FATAL: test split does not match the recorded seed-0 split.\n"
             f"  expected {len(REF['test_targets'])} targets, got {len(test_targets)}\n"
             f"  missing: {sorted(set(REF['test_targets']) - set(test_targets))[:8]}\n"
             f"  extra:   {sorted(set(test_targets) - set(REF['test_targets']))[:8]}")
log(f"split VERIFIED against reference: {len(test_targets)} test targets match")
# data_root is a required CLI arg but is never scanned: build_dataloaders reuses
# the manifest whenever it exists, which it now always does by this point.
counts = {s: sum(1 for r in rows if r["split"] == s) for s in ("train", "val", "test")}
log(f"crops OK  splits={counts}  data_root={data_root}")

# ---------------------------------------------------------------- command build
# NOTE: the canonical builder now lives in src/reference_cmd.py and is shared with
# docs/repeat_audit.py. This copy is retained only because this script has already
# been executed and its provenance should not change after the fact.
def train_cmd(cfg, seed, data_root):
    """Build the training command entirely from ARGS.

    Store-true flags are emitted only when the reference says true, so a false
    entry cannot silently become a passed flag. Anything in ARGS that is not
    handled here raises rather than being ignored -- a hyperparameter added to
    the reference file and quietly dropped from the command line is the same
    class of bug as --amp, just in the other direction.
    """
    cmd = [sys.executable, "-m", "src.train",
           "--data-root", data_root, "--config", cfg,
           "--dataset-name", DS, "--seed", str(seed),
           "--split-seed", str(SPLIT_SEED), "--out-root", str(OUT)]
    scalars = {"epochs": "--epochs", "batch_size": "--batch-size", "lr": "--lr",
               "weight_decay": "--weight-decay", "image_size": "--image-size",
               "num_workers": "--num-workers", "focal_gamma": "--focal-gamma",
               "focal_alpha": "--focal-alpha", "grad_clip": "--grad-clip",
               "group_by": "--group-by", "limit": "--limit",
               "band_dropout_p": "--band-dropout-p"}
    flags = {"amp": "--amp", "no_pretrained": "--no-pretrained", "sas": "--sas"}
    for key, opt in scalars.items():
        if ARGS[key] is not None:
            cmd += [opt, str(ARGS[key])]
    for key, opt in flags.items():
        if ARGS[key]:
            cmd.append(opt)
    unhandled = {k for k in ARGS
                 if not k.startswith("_") and k not in scalars and k not in flags
                 and k != "dataset_name"}
    if unhandled:
        sys.exit(f"FATAL: reference file has parameters this script does not pass: "
                 f"{sorted(unhandled)}. Add them to train_cmd or remove them.")
    return cmd


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

# Print the exact command once, BEFORE hours of compute depend on it, and name
# the reproducibility control. Both were previously implicit.
log("effective training command (from results/reference/train_args_c40_vid.json):")
log("   " + " ".join(str(x) for x in train_cmd(CONFIGS[0], 0, data_root)[2:]))
ctrl = ARGS["_reproducibility_control"]
log(f"seed {ctrl['seed']} is a reproducibility control: expect xception "
    f"{ctrl['expected_xception_roc_auc']}, +FAD {ctrl['expected_fad_roc_auc']}")

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
        ok, _ = sh(train_cmd(cfg, seed, data_root), stream=True)
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
