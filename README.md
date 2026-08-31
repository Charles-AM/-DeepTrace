# DeepTrace

**Detecting Deepfakes Through Frequency Artifacts Using Custom DCT Layers and Hybrid Deep Learning**

A hybrid spatial–frequency deepfake detector built around a custom **differentiable 2D DCT layer**
and a **learnable frequency mask**, fused with a spatial CNN branch via a gated mechanism.

This repo is a graduate Advanced Image Processing project. The goal is to *rigorously test*
whether learnable frequency masking helps — not to assume it does.

---

## Status

| Task | Component | State |
|------|-----------|-------|
| 1 | `DifferentiableDCT2D` — custom differentiable DCT layer | ✅ done (`src/models/dct.py`) |
| 2 | `LearnableFrequencyMask` | ✅ done (`src/models/frequency_mask.py`) |
| 3 | `SpatialFrequencyDetector` (hybrid model + focal loss) | ✅ done (`src/models/detector.py`, `src/losses.py`) |
| 4 | Training / evaluation pipeline | ✅ done (`src/{data,engine,metrics,train}.py`) |
| 5 | Ablations & baselines | ✅ code (`src/run_ablation.py`) |
| 6 | JPEG compression robustness | ✅ code (`src/jpeg_robustness.py`) |
| 7 | Cross-dataset generalization (FF++ → Celeb-DF v2) | ✅ code (`src/cross_dataset.py`) |
| 8 | Explainability & visualizations | ✅ code (`src/visualize.py`) |
| + | Established baselines: XceptionNet, EfficientNet-B0/B4/B7 (`timm`) | ✅ code (`src/models/baselines.py`) |
| + | Face-crop extraction from FF++/Celeb-DF videos | ✅ code (`src/extract_faces.py`) |

Tasks 5–8 are code-complete and unit-tested; they produce final numbers once real
training runs exist (needs a working GPU + the target datasets).

## Compute plan

Development is disk-constrained locally, so:

- **Local (Apple Silicon Mac):** code editing + unit tests only. No datasets.
- **Kaggle Notebooks (free P100/T4):** all dataset handling, training, and evaluation.
  See [`docs/kaggle-setup.md`](docs/kaggle-setup.md).

## Scoped experiment settings

To fit free-tier GPU budgets, the headline configuration is:

- input `128 × 128`, `~20k / 4k / 4k` train/val/test face crops
- backbones: ResNet-18 (spatial), EfficientNet-B0 (baseline)
- 2 seeds per config (3 for the final Full-vs-best-ablation comparison)
- Xception baseline via `timm`; SpecXNet and similar dual-domain nets are **cited**, not reproduced

## Datasets

- **FaceForensics++** — request access: <https://github.com/ondyari/FaceForensics> (academic email; approval can take days)
- **Celeb-DF v2** — request access via the form in <https://github.com/yuezunli/celeb-deepfakeforensics>
- **Stand-in for early development:** a real-vs-fake face dataset already hosted on Kaggle
  (e.g. "140k Real and Fake Faces"), so the pipeline can be smoke-tested before FF++ approval lands.

Do **not** commit dataset files or redistribute FF++ — the EULA forbids it.

## Layout

```
src/
  models/
    dct.py              # Task 1 — DifferentiableDCT2D
    frequency_mask.py   # Task 2 — LearnableFrequencyMask
    frequency_branch.py # Task 3 — DCT -> mask -> CNN -> embedding
    detector.py         # Task 3 — SpatialFrequencyDetector (hybrid + gated fusion)
  losses.py             # Task 3 — FocalLoss
  config.py             # named configs for the Task 5 ablation matrix
tests/
  test_dct.py           # scipy equivalence, inverse roundtrip, gradient flow, Parseval
  test_frequency_mask.py
  test_detector.py      # all 7 configs build + train; focal loss; fusion; freezing
docs/
  kaggle-setup.md
```

## Running the tests

```bash
pip install -r requirements.txt
pytest -q
```

## Running a training job (Task 4)

```bash
python -m src.train --data-root /kaggle/input/<dataset> --config full \
  --dataset-name ffpp --image-size 128 --epochs 15 --batch-size 64 --seed 0
```

- `--config` is any name from `src/config.py` (`full`, `no_mask`, `no_dct`,
  `no_spatial`, `no_fusion_concat`, `baseline_spatial`, `frequency_only`).
- The train/val/test split is written once to `results/manifests/…csv` and reused
  by every later run with the same `--dataset-name`/`--seed`/`--image-size`, so all
  configs are compared on identical images.
- Outputs per run: `results/<run>/best.pt`, `test_metrics.json`, `history.json`,
  TensorBoard logs, `frequency_mask.pt`; plus a row appended to `results/summary.csv`.
- Add `--limit 200` for a fast smoke test, `--amp` on CUDA, `--group-by '(id\d+)'`
  to split by video/identity.
- Baselines run through the same command: `--config xception` / `efficientnet_b0`.

## Downstream experiments (Tasks 5–8)

```bash
# Task 5 — ablation matrix + comparison table + bar chart + paired t-tests
python -m src.run_ablation --data-root <ds> --dataset-name rvf \
  --configs baseline_spatial frequency_only no_mask no_fusion_concat full \
  --seeds 0 1 2 --epochs 15

# Task 6 — JPEG robustness (ROC-AUC vs quality) over trained checkpoints
python -m src.jpeg_robustness --runs rvf_full_seed0 rvf_baseline_spatial_seed0 --seed 0

# Task 7 — cross-dataset (train FF++, test Celeb-DF, no retraining)
python -m src.cross_dataset --runs ffpp_full_seed0 --cross-root <celebdf_crops> --cross-name celebdf

# Task 8 — mask heatmap, radial importance, Grad-CAM, t-SNE for one run
python -m src.visualize --run rvf_full_seed0 --data-root <ds> --dataset-name rvf

# Prep — extract face crops from FF++/Celeb-DF videos
python -m src.extract_faces --videos <video_dir> --out <crops_dir> --size 160
```
