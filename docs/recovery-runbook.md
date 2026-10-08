# Recovery runbook — close the two reproducibility gaps in one Kaggle session

**No retraining.** Everything here is a file copy or inference. Budget ~40 minutes.

Closes:
1. **V2 manifests not committed** → four membership checks currently SKIP locally
2. **C1 and C2 have no prediction dumps** → both are aggregate-only

🚨 **Time-sensitive.** `docs/ARTIFACT-POLICY.md`: *"Kaggle version outputs are not
permanent."* The C2 checkpoints hold the **sd 0.0226** repeat-audit result. If that
version expires, C2 can never be re-analysed at the prediction level.

---

## Setup — attach BOTH inputs

New notebook. Add Input →

1. **Notebook Output → the night-1 version** (`contribution-3`, the one containing
   `night1/`) — holds checkpoints and manifests
2. **The `c40-run` crops dataset**

⚠️ The 2026-09-18 re-score failed at first with only the output attached. Manifests
store absolute crop paths, so inference needs the crops mounted at the recorded
location. **Both inputs, every time.**

CPU is sufficient. GPU is faster.

## Cell 1 — clone

```
!git clone https://github.com/Charles-AM/-DeepTrace.git /kaggle/working/-DeepTrace
```

## Cell 2 — locate everything, do not guess

```
!find /kaggle/input -name "best.pt" | sort
!find /kaggle/input -name "*v2*seed0_sz128.csv" | sort
!find /kaggle/input -maxdepth 4 -type d -name "night1"
```

Set `NIGHT1` to the directory **containing** `night1/`. Record the exact run
directory names printed — do not assume the `c1_lr*` / `c2_rep*` spellings in the
registry are exact.

```
!ls $NIGHT1/night1/
!cat $NIGHT1/night1/<one_c2_run>/*.json
```

The per-run JSON gives that run's `dataset_name`, `seed` and `split_seed`. **Read
them; do not infer them.**

---

## STEP 1 — V2 manifests (do this first, it is a file copy)

The membership check is pure string set intersection — `verify_v2.py` never opens
the crop files. Absolute Kaggle paths are fine exactly as stored. **No rewriting.**

```
!mkdir -p /kaggle/working/out/manifests
!cp $NIGHT1/manifests/ffpp_c40_v2seen_seed0_sz128.csv  /kaggle/working/out/manifests/
!cp $NIGHT1/manifests/ffpp_c40_v2unseen_seed0_sz128.csv /kaggle/working/out/manifests/
!head -2 /kaggle/working/out/manifests/*.csv && wc -l /kaggle/working/out/manifests/*.csv
```

Expect header `path,label,split` and roughly 27,000 rows in the seen manifest
(23,250 train / 3,000 val / 750 test).

⚠️ If the filenames differ, use what Cell 2 printed.

---

## STEP 2 — C2 dumps (4 runs — the ones that matter most)

`c2_rep{1,2}` × `{xception, xception_fad}`, seed 0.

🚨 **A distinct `--out-dir` per run is mandatory.** `predict.py` names its output
`<run>_<split>.csv`. The night-1 bug was two calls writing the same path, and the
second silently overwrote the first.

```
import subprocess, pathlib
NIGHT1 = "<paste from Cell 2>"
RUNS = ["c2_rep1_xception_seed0", "c2_rep1_xception_fad_seed0",
        "c2_rep2_xception_seed0", "c2_rep2_xception_fad_seed0"]   # fix to actual names
for r in RUNS:
    out = f"/kaggle/working/out/predictions_c2/{r}"
    pathlib.Path(out).mkdir(parents=True, exist_ok=True)
    print("==", r)
    subprocess.run(["python","-m","src.predict","--run",r,
        "--results-root",NIGHT1,"--dataset-name","ffpp_c40_vid",   # confirm from JSON
        "--seed","0","--split-seed","0","--split","test",
        "--out-dir",out], cwd="/kaggle/working/-DeepTrace", check=True)
```

**Expect** `wrote ... (3000 predictions, ...)` per run — 3,000, not 750. V2 dumps
are 750 because its test slice is smaller; these are standard c40 test splits.

```
!find /kaggle/working/out/predictions_c2 -name "*.csv" | wc -l   # expect 4
!md5sum $(find /kaggle/working/out/predictions_c2 -name "*.csv") # expect 4 DISTINCT
```

⚠️ Identical hashes mean the collision bug recurred. Stop and fix `--out-dir`.

## STEP 3 — C1 dumps (6 runs)

Three learning rates × both arms, seed 0. Same loop, same guards.

```
RUNS = [...]   # the six c1_lr* names from Cell 2
```
Write to `/kaggle/working/out/predictions_c1/<run>/`, then repeat the count and
hash checks — expect **6** files, **6 distinct** hashes.

⚠️ C1 ran at a **5-epoch** budget, not 15 (`docs/c1-lr-prespecification.md`).
That is prespecified and correct. Do not "fix" it.

---

## STEP 4 — download and commit

```
!cd /kaggle/working/out && zip -rq /kaggle/working/recovery.zip . && du -h /kaggle/working/recovery.zip
```

Expect ~10 MB (2 manifests ≈ 5 MB + 10 dumps ≈ 6.5 MB). Download from the
notebook's Output panel.

Locally, unzip into the repo so files land at:

```
results/manifests/ffpp_c40_v2seen_seed0_sz128.csv
results/manifests/ffpp_c40_v2unseen_seed0_sz128.csv
results/predictions_c2/<run>/<run>_test.csv     × 4
results/predictions_c1/<run>/<run>_test.csv     × 6
```

Commit the manifests and each campaign separately — `ARTIFACT-POLICY.md`:
*commit per milestone, not in one batch at the end.*

---

## STEP 5 — verify locally, which is the point

```bash
python -m src.verify_v2 \
  --seen results/predictions_v2/v2seen_xception_seed0_test.csv \
  --unseen results/predictions_v2/v2unseen_xception_seed0_test.csv \
  --manifest-seen results/manifests/ffpp_c40_v2seen_seed0_sz128.csv \
  --manifest-unseen results/manifests/ffpp_c40_v2unseen_seed0_sz128.csv \
  --expect-seen-auc 0.9884
```

**Expect 13/13 with zero SKIPs.** The four that currently skip are:
zero seen-eval crops in training · zero unseen-eval crops in training ·
seen groups ARE represented in training · unseen groups are NOT.

Then `python -m src.reproduce` — still **16/16**, and its V2 section should no
longer report the manifest caveat.

## If a checkpoint is already gone

Do not retrain to recreate it. A retrained run is a different run, and presenting
it as the original would be a worse error than the missing dump. Record the loss in
`ARTIFACT-POLICY.md` and mark that campaign permanently aggregate-only.
