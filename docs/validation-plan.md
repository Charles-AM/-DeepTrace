# Validation plan — proving every contribution

One row per claim: the experiment that validates it, the data it needs, what result
would **confirm** vs **falsify** it, current status, and cost. A claim is "proven"
only when it holds across **multiple seeds** and, where it's a generalisation claim,
**multiple datasets**.

Legend — status: ✅ done · 🟡 partial · ⬜ not started.

---

## C1 — On a matched backbone, adding frequency modelling does not improve accuracy

| | |
|---|---|
| Experiment | Paired comparison, same backbone, same data/augmentation/budget: (a) ResNet-18 ± frequency branch (`baseline_spatial` vs `full`/`no_mask`); (b) Xception ± FAD (`xception` vs `f3net`). Paired t-test + bootstrap CI on the AUC difference + Cohen's d, per seed. |
| Data | FF++ c23 (have); repeat on FF++ c40 and raw. |
| Seeds | 3 done → extend headline pairs to **5**. |
| Confirms if | ΔAUC CI includes 0 / \|d\| < 0.2 at every compression level. |
| Falsified if | frequency variant beats its matched baseline by a significant margin at any level. |
| Status | 🟡 c23 done (2 matched anchors, 3 seeds, frequency ≈ 0 gain). Need c40, raw, +2 seeds. |
| Cost | c40 run ≈ 6 h; raw run ≈ 6 h; +2 seeds on 6 configs ≈ 6 h. |

## C2 — The learnable fusion gate never engages

| | |
|---|---|
| Experiment | Read `sigmoid(alpha_logit)` from every gated checkpoint (`src/gate_readout.py`). Also: freeze the gate at 0.5 and retrain one config — if AUC is unchanged, the gate is inert by construction. |
| Data | existing checkpoints; one FF++ c23 retrain for the frozen-gate control. |
| Seeds | 3 done → 5 with the extended runs. |
| Confirms if | α stays within ~0.01 of 0.5 across all configs/seeds AND frozen-gate AUC ≈ learned-gate AUC. |
| Falsified if | α drifts systematically (e.g. → >0.6 or <0.4) or frozen-gate hurts AUC. |
| Status | ✅ readout done (α = 0.496 ± 0.001). ⬜ frozen-gate control. |
| Cost | frozen-gate control ≈ 30 min × 3 seeds. |

## C3 — Per-manipulation inversion: frequency helps on crude forgeries, not subtle ones

| | |
|---|---|
| Experiment | Per-method AUC breakdown (`src/permanip.py`) for every model. Correlate frequency-only's per-method AUC with a "crudeness" proxy (e.g. mean SSIM to the real frame, or blending-boundary strength). |
| Data | FF++ (4 methods) — have. Extend to FaceShifter (5th FF++ method) and, if downloaded, DF40's per-method splits. |
| Seeds | 3 done. |
| Confirms if | frequency-only ranks methods crude→subtle in the same order every seed; spatial CNN ≥ frequency on the subtlest method. |
| Falsified if | frequency-only beats the spatial CNN on NeuralTextures / FaceShifter, or the ordering is seed-dependent. |
| Status | ✅ FF++ 4-method, 3 seeds. 🟡 add FaceShifter + a crudeness quantification. |
| Cost | ≈ 30 min (inference only) + FaceShifter extraction ≈ 1 h. |

## C4 — "Redundant, not fragile": the spectral signal survives compression but is weak and already captured by the spatial CNN

| | |
|---|---|
| Experiment | (a) `src/spectra.py` real-vs-fake DCT gap + per-coefficient t-map at c23, c40, raw, and JPEG-{90..30}. (b) **CKA** between the spatial-branch and frequency-branch features on the test set — high alignment ⇒ redundant. (c) Test-time frequency-band ablation of a trained spatial model — if zeroing the "forensic" bands drops its AUC, it was already using them. |
| Data | FF++ c23 (have), c40, raw. |
| Seeds | spectra is dataset-level (seed only affects the split) — run on all 3 split seeds. |
| Confirms if | gap has small effect size (\|d\| < 0.3) AND changes little c23→c40 AND CKA(spatial, freq) is high AND the spatial model already relies on the informative bands. |
| Falsified if | the gap is large, or collapses under c40 (that would mean "fragile", not "redundant"), or CKA is low (branches encode different things). |
| Status | 🟡 c23 spectra done (d ≈ 0.15, JPEG-robust). ⬜ c40/raw spectra, CKA, band-ablation. |
| Cost | spectra ≈ 5 min/condition; CKA script ≈ 1 h to write + 20 min run; band-ablation ≈ 1 h. |

## C5 — No frequency advantage under real-world perturbation

| | |
|---|---|
| Experiment | `src/robustness.py` sweep — JPEG, blur, noise, resize, contrast × 4 severities, all key checkpoints. Add the **DeeperForensics-1.0** standard perturbation set and the **DFDC** perturbation protocol for external validity. |
| Data | FF++ test split (have); DeeperForensics-1.0, DFDC perturbed. |
| Seeds | 3-seed JPEG done → all perturbations × 3 seeds. |
| Confirms if | frequency models' retention ≈ spatial models' retention (CI overlap) on every perturbation, every seed, every dataset. |
| Falsified if | a frequency model degrades significantly slower on any standard perturbation across seeds. |
| Status | 🟡 FF++ seed-0 all perturbations + 3-seed JPEG. ⬜ other seeds/perturbations, external datasets. |
| Cost | full FF++ 3-seed sweep ≈ 3 h; DeeperForensics extraction ≈ 2 h. |

## C6 — Frequency hybrids transfer no better (worse) across datasets

| | |
|---|---|
| Experiment | Train on FF++, evaluate zero-shot on every external dataset (`src/cross_dataset.py`). Report ΔAUC per target and mean ΔAUC, per seed, with CIs. |
| Data | **targets: Celeb-DF v2, DFDC, DF40 (diffusion + GAN + edit subsets), WildDeepfake, DeeperForensics-1.0.** The more targets that show the same pattern, the stronger. |
| Seeds | seed 0 only → **3 seeds** minimum. |
| Confirms if | across ≥4 targets and 3 seeds, spatial baselines ≥ frequency hybrids (mean ΔAUC not better for frequency). |
| Falsified if | a frequency hybrid has a smaller mean ΔAUC (better transfer) on the majority of targets. |
| Status | ⬜ seed-0 partial (Celeb-DF, DFDC). Need all targets × 3 seeds; crops must be saved as **private** Kaggle datasets. |
| Cost | extraction ≈ 1–2 h per dataset (5 datasets); cross-eval inference ≈ 3–4 h total. |

## C7 — F3-Net's headline low-quality (c40) gains do not replicate under matched training

| | |
|---|---|
| Experiment | Re-download FF++ at c40, same video split as c23 (`extract_faces_v2 --video-list`), train `baseline_spatial` + `xception` + `f3net` × 3 seeds. Compare `f3net` vs `xception` at c40 with paired test + CI. Cross-reference F3-Net's published c40 numbers. |
| Data | FF++ c40. |
| Seeds | 3. |
| Confirms the paper's spine if | `f3net` − `xception` at c40 is not significantly positive (CI includes 0). |
| "Reverse" outcome | if F3-Net *does* win at c40, the paper reframes to "we reproduce the c40 gain but show it's the only regime it exists and it doesn't transfer" — still publishable. |
| Status | ⬜ not started — **the pivotal run.** |
| Cost | ≈ 6 h (download + extract + 9 training runs). |

## C8 — The frequency branch is a pure computational cost (constructive turn)

| | |
|---|---|
| Experiment | Measure params, FLOPs (`fvcore`/`thop`), and GPU inference latency + peak memory for `baseline_spatial`, `xception`, `f3net`, `full` at matched input size. Pair with the accuracy numbers already in `results/`. |
| Data | none (model-only). |
| Seeds | n/a (latency: report mean ± std over 100 batches). |
| Confirms if | frequency variants cost meaningfully more (params/FLOPs/latency) at equal-or-worse AUC. |
| Falsified if | the frequency branch is ~free. |
| Status | ⬜ not started. |
| Cost | ≈ 20 min. |

---

## Datasets to acquire (in priority order)

| dataset | role | access | est. prep |
|---|---|---|---|
| FF++ c40 | C1, C4, C5, **C7** | official (have c23 pipeline) | 1.5 h |
| Celeb-DF v2 | C6 | have EULA access; re-extract, save private | 1.5 h |
| DFDC (preview/sample) | C5, C6 | Kaggle | 1 h |
| DF40 | C6 (diffusion + modern) | Kaggle / HF | 2 h |
| DeeperForensics-1.0 | C5, C6 (perturbation protocol) | public | 2 h |
| WildDeepfake | C6 (in-the-wild) | public (request) | 2 h |
| FaceShifter (FF++ 5th method) | C3 | bundled with FF++ | 0.5 h |
| FF++ raw | C1, C4 (optional) | official | 1.5 h |

EULA: FF++ and Celeb-DF crops stay **private** on Kaggle — never "Make Public".

## Backbones (rule out "backbone was too weak to need frequency")

Have: ResNet-18 (hybrid family), Xception (baseline + F3-Net). Add: **EfficientNet-B4**
(bigger), and if time **a ViT/Swin-Tiny**. Re-run C1's ± frequency comparison on each.

## Statistical rigor checklist (apply to every headline number)

- ≥ 3 seeds (5 for the two headline comparisons)
- paired t-test **and** bootstrap 95 % CI on the *difference*
- Cohen's d (effect size) reported alongside every p-value
- Holm–Bonferroni correction across the ablation family
- video-level split (already in place via `group_by`) — state it explicitly

## Rough 4-week schedule

- **Week 1** — save Celeb-DF/DFDC crops private; C7 c40 run; C8 efficiency table.
- **Week 2** — acquire DF40 + DeeperForensics + FaceShifter; C6 cross-dataset seed 0 on all targets.
- **Week 3** — C6 seeds 1–2; C5 full robustness (3 seeds + DeeperForensics); C1 +2 seeds.
- **Week 4** — C2 frozen-gate control; C4 CKA + band-ablation + c40 spectra; C1 EfficientNet-B4; start the draft.
