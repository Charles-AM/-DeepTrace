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
| 3 | `SpatialFrequencyDetector` (hybrid model + focal loss) | ⬜ |
| 4 | Training / evaluation pipeline | ⬜ |
| 5 | Ablations & baselines | ⬜ |
| 6 | JPEG compression robustness | ⬜ |
| 7 | Cross-dataset generalization (FF++ → Celeb-DF v2) | ⬜ |
| 8 | Explainability & visualizations | ⬜ |

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
    dct.py             # Task 1 — DifferentiableDCT2D
    frequency_mask.py  # Task 2 — LearnableFrequencyMask
tests/
  test_dct.py          # scipy equivalence, inverse roundtrip, gradient flow, Parseval
  test_frequency_mask.py
docs/
  kaggle-setup.md
```

## Running the tests

```bash
pip install -r requirements.txt
pytest -q
```
