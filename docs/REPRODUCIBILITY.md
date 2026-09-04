# Reproducibility — methods, environment, and experiment registry

Single canonical reference for the paper's Methods/Reproducibility section and
supplementary material. Everything here is either verified against a genuine
generated artifact or explicitly marked as a gap to close.

---

## Code & data availability

- **Code:** github.com/Charles-AM/-DeepTrace (public; `git log` is a full audit
  trail — every experiment, code change, and finding is a commit with reasoning
  in the message, not just a diff).
- **Trained checkpoints:** NOT in the public repo (`.gitignore` blocks `*.pt`).
  Live in the Kaggle notebook output `charlesappiahmanu/deeptrace3` (the original
  30) and local backups (`~/Downloads/*.tar.gz`, see below for the manifest).
  Provenance JSONs (`test_metrics.json`, containing every metric + the exact
  learned α where applicable) for every run ARE in git, under
  `results/in_domain/per_run/` and `results/analysis/`.
- **Datasets:** not redistributed (FaceForensics++ / Celeb-DF EULAs forbid it —
  see `CITATIONS.md`). Exact acquisition parameters below let anyone with their
  own EULA approval reconstruct the identical crop set.

## Environment

**Hardware:** Kaggle Notebooks, GPU accelerator **T4 ×2** (P100 is explicitly
incompatible with the installed torch build — `CUDA error: no kernel image
available` — never use it for this project).

**Software:** ⚠️ **GAP — not yet captured.** `requirements.txt` only pins lower
bounds (`torch>=2.2`, `torchvision>=0.17`, `numpy>=1.24`, `scipy>=1.10`,
`timm>=0.9`, `scikit-learn>=1.3`, `matplotlib>=3.7`, `pandas>=2.0`), not what
Kaggle actually had installed at run time. **Action for next session:** run
`!pip freeze > /kaggle/working/pip_freeze.txt` early in the c40 runbook, download
it, and commit as `docs/pip_freeze_2026-09.txt`. Until then, exact package
versions used for any run before that capture are unverified.

**Install notes (do not deviate):**
- Never `pip install -r requirements.txt` on Kaggle — it breaks the pre-installed
  torch build.
- Only ever install individual small packages as needed: `facenet-pytorch
  --no-deps` (face extraction), `fvcore` + `iopath` (efficiency profiling, needs
  both — `--no-deps` on fvcore alone breaks it).

## Datasets — exact acquisition parameters

| dataset | source | parameters used | split |
|---|---|---|---|
| FaceForensics++ c23 | official request form + `src/ffpp_fastdl.py` | `--num-pairs 150 --originals --methods Deepfakes Face2Face FaceSwap NeuralTextures --compression c23` → ~300 real + ~1200 fake videos | `src/extract_faces_v2.py --size 160 --every 12 --max-per-video 20` → 30,000 crops (6,000 real / 24,000 fake) |
| FaceForensics++ c40 | same, `--compression c40` | not yet run (planned, `docs/c40-runbook.md`) | same video IDs (same `--num-pairs 150`) so directly paired with c23 |
| Celeb-DF v2 | official EULA + Google Drive zip (id `1iLx76wsbi9itnkxSqz9BVBl4ZvnbIazj`) | prior extraction used a **random balanced sample, not the official `List_of_testing_videos.txt`** — ⚠️ flagged for correction on re-extraction (planned) | 8,951 crops (4,468/4,483) — superseded, will be re-extracted with the official list |
| DFDC | Kaggle `train_sample_videos`, `metadata.json` labels | 400 videos (914 real / 3,756 fake crops) | superseded, will be re-extracted alongside Celeb-DF |

**Train/val/test split:** seeded (`--seed {0,1,2}`), 80/10/10, deterministic given
the same crop directory + seed (`src/data.py::make_splits`) — manifests are
regenerable, not just stored. Manifests themselves are *not* committed to git
(contain full crop paths; kept local/Kaggle-side only) but ARE in the local backup
tarballs.

## Training hyperparameters

Defaults (`src/train.py`), used for every in-domain/ablation run unless noted:

| param | value |
|---|---|
| optimizer | AdamW, `lr=3e-4`, `weight_decay=0.05` |
| weight-decay grouping | **as of commit `7bb4e11`**: `utils.no_decay_param_groups` — biases/norms/scalar gates excluded from decay. Runs before this commit (all `ffpp_*`, i.e. everything except the two `ffppfix_*` diagnostic runs) used a single undifferentiated param group — see the C2 confound-check note below. |
| LR schedule | CosineAnnealingLR, `T_max=epochs` |
| epochs | 15 (12 in a few early smoke runs, noted per-run in `history.json`) |
| batch size | 64 |
| image size | 128 |
| loss | Focal loss, γ=2, `α_pos = n_real/(n_real+n_fake)` ≈ 0.20 |
| grad clip | 1.0 (global norm) |
| AMP | enabled on CUDA |
| seeds | {0, 1, 2} for all headline comparisons |

Model architecture parameters (embed_dim, mask channels, band-dropout p, fusion
type) are named-config-driven — see `src/config.py::MODEL_CONFIGS` and
`docs/related-work.md` for which config corresponds to which paper's method.

## Statistical protocol

- Paired t-test + (where noted) bootstrap CI on the metric *difference*, not
  independent per-arm CIs.
- Effect sizes reported as Cohen's d where relevant (spectral gap, CKA).
- 3 seeds minimum for any headline claim; flagged explicitly wherever a result is
  only 1 seed (e.g. the two `ffppfix_*` gate-confound diagnostic runs) as
  preliminary, not final.

## Experiment registry — every run, mapped to its command and commit

| experiment | script | exact command (see file for full flags) | result location | commit |
|---|---|---|---|---|
| In-domain matrix (10 configs × 3 seeds, c23) | `src.run_ablation` | `--data-root <ffpp_crops> --dataset-name ffpp --configs baseline_spatial xception f3net full ... --seeds 0 1 2 --epochs 15` | `results/in_domain/` | `81e5bdd` |
| Robustness sweep | `src.robustness` | see `docs/c40-runbook.md` pattern; seed-0 all perturbations + 3-seed JPEG | `results/robustness/` | `eb86486`, `ca268bc` |
| Fusion-gate readout | `src.gate_readout` | `--results-root results` | `results/analysis/fusion_alpha.csv` | `a5bd9fa`, `9deb97f` |
| Per-manipulation breakdown | `src.permanip` | 3 seeds × 6 configs | `results/analysis/permanip/` | `a5bd9fa` |
| Spectral analysis (rewritten) | `src.spectra` | `--limit 6000 --jpeg 30 --crop 160 --splits all` | `results/analysis/spectra/` | `fda0d73`, `a5bd9fa` |
| Late-fusion complementarity (C1b) | `src.late_fusion` | 3 seeds, `baseline_spatial` × `frequency_only` | `results/analysis/late_fusion/` | `d5446eb` (genuine artifact) |
| Efficiency table (C8) | `eff_table.py` (fvcore) | see `docs/c40-runbook.md` | `results/analysis/efficiency/` | `18279e9` |
| CKA (C4) | `src.cka` | 3 seeds, `full` config, spatial vs frequency branch | `results/analysis/cka/` | `d5446eb` (genuine artifact) |
| Gate weight-decay confound check | `src.train` (fixed optimizer) + `src.gate_readout` | `--dataset-name ffppfix --config {full,no_mask} --seed 0` | `results/in_domain/per_run/ffppfix_*.json` | `9deb97f`, `c9aefb6` |
| c40 training run | `src.run_ablation` | **not yet run** | — | — |
| Cross-dataset seeds 1-2 | `src.cross_dataset` | **not yet run** (seed 0 only, superseded crops) | — | — |

## Known deviations / limitations to disclose

- Celeb-DF/DFDC seed-0 cross-dataset numbers currently in the repo used a
  **non-official** Celeb-DF test sample (random balanced, not
  `List_of_testing_videos.txt`) — will be corrected before those numbers are
  used as final results; flag clearly if cited before the re-extraction lands.
- FF++ subset (150 pairs, not the full ~1000) — scoped for compute budget, stated
  explicitly in the paper's limitations.
- Package versions unpinned until the `pip freeze` capture above happens.
- Gate weight-decay confound check is 1 seed each (`full`, `no_mask`) — real
  evidence, not exhaustive; more seeds would strengthen it further if time
  allows.
