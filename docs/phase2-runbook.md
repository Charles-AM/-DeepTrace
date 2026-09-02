# Phase 2 runbook — FaceForensics++ experiments

Every stage has an explicit **✅ validation check** — do not move on until it passes.
Times are rough T4 estimates. Run everything in **one Kaggle notebook** (GPU T4 x2,
Internet On, Persistence "Files only"). Save a Version after each stage that produces
outputs.

---

## Stage 0 — Notebook setup  (2 min)

```python
import os
if not os.path.isdir("-DeepTrace"):
    !git clone https://github.com/Charles-AM/-DeepTrace.git
%cd ./-DeepTrace
!git pull -q
!nvidia-smi --query-gpu=name --format=csv,noheader
!pytest -q
```

**✅ check:** GPU says `Tesla T4`; pytest ends `~140 passed, 1 skipped`.

---

## Stage 1 — Download a scoped FaceForensics++ subset  (20–40 min, ~15–25 GB)

Put the emailed `faceforensics_download_v4.py` into the notebook (paste it into a
`%%writefile download.py` cell). Then:

```python
!python download.py ./ffpp --server EU2 -c c23 -n 200 -d original       -t videos
!python download.py ./ffpp --server EU2 -c c23 -n 200 -d Deepfakes      -t videos
!python download.py ./ffpp --server EU2 -c c23 -n 200 -d Face2Face      -t videos
!python download.py ./ffpp --server EU2 -c c23 -n 200 -d FaceSwap       -t videos
!python download.py ./ffpp --server EU2 -c c23 -n 200 -d NeuralTextures -t videos
```

If EU2 rejects `c23`, use `-c c40`. `-n 200` = 200 videos/category.

```python
!find ./ffpp -name "*.mp4" | wc -l
!find ./ffpp -maxdepth 4 -type d
```

**✅ check:** ~1000 `.mp4` files; folders `original_sequences/...` and
`manipulated_sequences/{Deepfakes,Face2Face,FaceSwap,NeuralTextures}/...`.

---

## Stage 2 — Extract face crops + save as a private dataset  (30–60 min)

```python
!pip -q install --no-deps facenet-pytorch
!python -m src.extract_faces --videos ./ffpp --out /kaggle/working/ffpp_crops \
  --size 160 --every 12 --max-per-video 20 --device cuda
```

```python
!find /kaggle/working/ffpp_crops -name "*.jpg" | wc -l
!ls /kaggle/working/ffpp_crops
import random, glob
from PIL import Image
for p in random.sample(glob.glob("/kaggle/working/ffpp_crops/**/*.jpg", recursive=True), 4):
    display(Image.open(p)); print(p)
```

**✅ check:** ~15–25k crops; `real/` and `fake/` folders; the 4 samples are actual
cropped faces, not full frames or garbage.

Then: notebook **Output tab → New Dataset → Private**, name it `ffpp-crops-160`.
From here on, **add that dataset as Input** and use its mount path — you never
re-download or re-extract.

---

## Stage 3 — Pipeline smoke test on real FF++  (3 min)

```python
FFPP = "/kaggle/input/ffpp-crops-160"      # your private dataset mount
!python -m src.train --data-root "{FFPP}" --config full --dataset-name ffpp \
  --image-size 128 --epochs 2 --batch-size 64 --limit 1500
```

**✅ check:** prints `real:fake` roughly balanced, `focal_alpha≈0.5`, two epochs of
`val_auc`, a final `test_auc=…`. No crash. (AUC value irrelevant here.)

---

## Stage 4 — Full training matrix  (~6–10 h GPU, split across sessions)

```python
!python -m src.run_ablation --data-root "{FFPP}" --dataset-name ffpp \
  --configs baseline_spatial xception efficientnet_b0 f3net \
            frequency_only no_mask no_fusion_concat \
            full full_banddrop full_banddrop+sas \
  --reference full_banddrop+sas --seeds 0 1 2 \
  --image-size 128 --epochs 15 --batch-size 64
```

10 configs × 3 seeds = 30 runs. If a session times out, re-run — completed runs are
in `results/summary.csv` and skipped is fine to re-do (rows append; the aggregation
de-dups by run+seed). To only rebuild tables after a partial run, add
`--aggregate-only`.

```python
import pandas as pd
print(pd.read_csv("results/summary.csv").groupby("run").size())
print(open("results/ablation_table.md").read())
```

**✅ check:** `summary.csv` has 30 rows (10 experiments × 3 seeds); `ablation_table.md`
shows mean ± std and p-values; `full_banddrop+sas` is at or near the top on ROC-AUC
and EER. **Save a Version.**

---

## Stage 5 — Robustness sweep  (~1–2 h)

```python
RUNS = ["ffpp_baseline_spatial_seed0", "ffpp_xception_seed0", "ffpp_f3net_seed0",
        "ffpp_full_seed0", "ffpp_full_banddrop_sas_seed0"]
!python -m src.robustness --runs {" ".join(RUNS)} --dataset-name ffpp --seed 0 --limit 3000
```

```python
print(open("results/robustness_scores.csv").read())
```

**✅ check:** `robustness.csv` + `robustness_{jpeg,blur,noise,resize,contrast}.png`
exist; scores table printed. Hypothesis to inspect: does `full_banddrop_sas` degrade
*less* than `baseline_spatial` as severity rises? (Report either way.)

---

## Stage 5.5 — Acquire the extra cross-test sets  (~1 h each)

FF++ + Celeb-DF alone = workshop tier. For ICASSP/WACV we need **three** zero-shot
targets, one of them diffusion-based. None of these needs an approval form.

### Celeb-DF v2  (already approved — see the earlier download notes)
9.3 GB Drive zip → unzip → extract **test-set** crops only:
```python
!python -m src.extract_faces --videos ./celebdf_v2 --out /kaggle/working/celebdf_crops \
  --video-list ./celebdf_v2/List_of_testing_videos.txt \
  --size 160 --every 15 --max-per-video 15 --device cuda
```

### DFDC — use the **Preview** set (public, ~5 GB, no form)
Add Input → Datasets → search `deepfake detection challenge` → pick the one by
**`gauravsharma`** / **`deepfake-detection-challenge`** *preview* (or `dfdc_train_part`
sample). Confirm it has `.mp4` + a `metadata.json` (label per video).
```python
!ls /kaggle/input/deepfake-detection-challenge* | head
# labels come from metadata.json: {"real"/"fake"} -> our scanner keys off folder
# names, so first sort videos into real/ and fake/ using the metadata:
!python - <<'PY'
import json, shutil, glob, os
root = glob.glob("/kaggle/input/*dfdc*") or glob.glob("/kaggle/input/deepfake-detection*")
root = root[0]
meta = json.load(open(glob.glob(f"{root}/**/metadata.json", recursive=True)[0]))
out = "/kaggle/working/dfdc_sorted"
for v, info in meta.items():
    src = glob.glob(f"{root}/**/{v}", recursive=True)
    if not src: continue
    d = os.path.join(out, "fake" if info["label"].lower()=="fake" else "real")
    os.makedirs(d, exist_ok=True)
    try: os.symlink(src[0], os.path.join(d, v))
    except FileExistsError: pass
print("sorted", len(meta), "videos ->", out)
PY
!python -m src.extract_faces --videos /kaggle/working/dfdc_sorted --out /kaggle/working/dfdc_crops \
  --size 160 --every 20 --max-per-video 12 --limit-videos 800 --device cuda
```

### DF40 — a **diffusion subset** (the acid test for SAS)
DF40 (Yan et al., NeurIPS 2024) spans 40 methods. Search Kaggle/HF for `DF40`
(project: `YZY-stack/DF40`). Download only a handful of method folders — pick
**2 diffusion** (e.g. `sit`, `ddim`/`collaborative_diffusion`, `e4s`), **1 GAN-era**
(`stylegan2` / `stargan`), **1 face-edit** — plus the DF40 real set.
```python
# DF40 ships already as face frames in method-named folders; the scanner needs
# real/ + fake/ — symlink the real set to real/, each chosen method to fake/:
!mkdir -p /kaggle/working/df40_eval/real /kaggle/working/df40_eval/fake
!ln -s /kaggle/input/df40/<real_folder>/*        /kaggle/working/df40_eval/real/ 2>/dev/null; true
!for m in sit collaborative_diffusion e4s stargan; do \
    ln -s /kaggle/input/df40/$m/* /kaggle/working/df40_eval/fake/ 2>/dev/null; done; true
!find /kaggle/working/df40_eval -type l | wc -l
```
If DF40 gives frames not videos, skip `extract_faces` and point `--targets` straight
at `/kaggle/working/df40_eval` (already `real/`+`fake/`).

**✅ check:** three crop dirs, each with `real/` + `fake/`, each ≥ ~3–4k images per
class. Sample 4 from each and eyeball them. Save each as a **private** dataset
(`celebdf-crops`, `dfdc-crops`, `df40-crops`).

**Licence note:** Celeb-DF & FF++ crops stay private (EULA). DFDC and DF40 are
research-licensed — keep private too and cite them.

---

## Stage 6 — Cross-dataset generalization  (eval: ~15 min)

```python
!python -m src.cross_dataset \
  --runs ffpp_full_banddrop_sas_seed0 ffpp_full_seed0 ffpp_xception_seed0 ffpp_f3net_seed0 \
  --targets celebdf=/kaggle/input/celebdf-crops \
            dfdc=/kaggle/input/dfdc-crops \
            df40=/kaggle/input/df40-crops \
  --limit 4000
```

```python
print(open("results/cross_dataset_pivot.csv").read())
```

**✅ check:** `cross_dataset.csv` + `_pivot.csv` exist; every model has a `mean_delta`.
The claim to support: `full_banddrop+sas` has the **smallest mean ΔAUC**.

---

## Stage 7 — Explainability figures  (~10 min)

```python
!python -m src.visualize --run ffpp_full_banddrop_sas_seed0 \
  --data-root "{FFPP}" --dataset-name ffpp --seed 0 --projection tsne
```

```python
from IPython.display import Image as IPy
for f in ["mask_heatmap", "radial_frequency_importance", "gradcam", "features_tsne"]:
    display(IPy(f"results/ffpp_full_banddrop_sas_seed0/figures/{f}.png"))
```

**✅ check:** four figures render. The radial profile should show a clear
low/mid/high bias (this is the sentence that goes under the figure in the paper).

---

## Stage 8 — Collect results off Kaggle

Zip `results/` (tables, CSVs, figures — **not** checkpoints) and download it via the
Output tab, or create a `deeptrace-results` private dataset. Those CSVs + PNGs are
what the paper is written from.

```python
!cd results && zip -qr /kaggle/working/deeptrace_results.zip . -x "*.pt" && ls -la /kaggle/working/deeptrace_results.zip
```

---

## If a stage fails

Paste the error. Common ones: dataset mount path differs (adjust `--data-root`),
`extract_faces` finds 0 videos (check `--videos` points above `original_sequences/`),
timm download blocked (Internet must be On), session OOM (drop `--batch-size` to 48).
