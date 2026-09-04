# Validation plan — proving every contribution

One row per claim: the experiment that validates it, the data it needs, what result
would **confirm** vs **falsify** it, current status, and cost. A claim is "proven"
only when it holds across **multiple seeds** and, where it's a generalisation claim,
**multiple datasets**.

Legend — status: ✅ done · 🟡 partial · ⬜ not started.

**Base/target (2026-09-04):** base/foundation = F3-Net (ECCV 2020, implemented +
c40-tested = C7). Recent target = FreqDebias (CVPR 2025, cited/positioned against,
not reproduced) — its "spectral bias" diagnosis is what C4's band-ablation directly
tests against our own frequency branch. Details: `docs/related-work.md`.

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

## C1b — Late-fusion complementarity test (added 2026-09-04, external review)

The end-to-end gated branch could fail to help for reasons unrelated to whether
frequency carries *any* independent signal — e.g. the optimisation confound in C2.
This test is architecture-agnostic: it asks the question in the cleanest possible
way, sidestepping the gate entirely.

| | |
|---|---|
| Experiment | Using the **already-trained** `baseline_spatial` and `frequency_only` checkpoints (no retraining): get each model's predicted score on val+test. Fit a simple logistic regression (or a single learned scalar) combining the two score vectors on val, evaluate the combined score on test. Compare combined AUC to `baseline_spatial` alone. |
| Data | none new — existing checkpoints + existing test split. |
| Seeds | all 3 existing seeds. |
| Confirms low incremental value if | combined AUC ≈ `baseline_spatial` alone (frequency-only's ~0.70 AUC adds nothing even in the most favourable, unconstrained combination). |
| Falsified if | logistic-regression fusion beats `baseline_spatial` by a real margin — would mean frequency *does* carry complementary signal and our end-to-end architecture (not the concept) is the problem. |
| Status | ⬜ not started — **top priority, do first**: zero new training, ~30 min of scripting, runs on CPU. |
| Why it matters | Much stronger evidence for "low incremental predictive value" than the gated branch alone, because it removes every confound in C2 (weight decay, gate optimisation, architecture) — just two independently-trained models' scores, combined the simplest possible way. |

## C1c — Parameter-matched capacity control (added 2026-09-04, external review)

Addresses a real confound: if `full` (ResNet-18 + frequency branch) doesn't beat
`baseline_spatial` (ResNet-18 alone), is that because frequency specifically doesn't
help, or because *any* second branch/fusion mechanism doesn't help once the task is
near ceiling on FF++ c23 (~0.995+ AUC everywhere)?

| | |
|---|---|
| Experiment | New config: take the frequency branch's exact architecture (mask analog → conv blocks → GAP → proj) and feed it a **second RGB view** instead of DCT coefficients (same parameter count, same fusion mechanism, only the input domain changes). Compare `baseline_spatial` vs this `full_dualspatial` vs `full`. |
| Data | FF++ c23 (have). |
| Seeds | 3. |
| Confirms if | `full_dualspatial` ALSO doesn't beat `baseline_spatial` — rules out "any second branch helps regardless of domain" and isolates the null result to frequency specifically (or to the near-ceiling task, which then generalises the finding). |
| Falsified if | `full_dualspatial` beats `baseline_spatial` while `full` (frequency) does not — would mean capacity/second-view helps but frequency-as-a-domain specifically doesn't; still a fine result, just a different one. |
| Status | ⬜ not started — needs a small new config in `src/config.py` + `src/models/`, then 3 training runs. |
| Cost | ~30 min engineering + 3 × ~15 min training ≈ 1.5 h. |

## C2 — The learnable fusion gate never engages

| | |
|---|---|
| Experiment | Read `sigmoid(alpha_logit)` from every gated checkpoint (`src/gate_readout.py`). Also: freeze the gate at 0.5 and retrain one config — if AUC is unchanged, the gate is inert by construction. |
| Data | existing checkpoints; one FF++ c23 retrain for the frozen-gate control. |
| Seeds | 3 done → 5 with the extended runs. |
| Confirms if | α stays within ~0.01 of 0.5 across all configs/seeds AND frozen-gate AUC ≈ learned-gate AUC. |
| Falsified if | α drifts systematically (e.g. → >0.6 or <0.4) or frozen-gate hurts AUC. |
| Status | ✅ readout done (α = 0.496 ± 0.001). ✅ **confound checked and RULED OUT 2026-09-04** — see below. ⬜ frozen-gate control still optional/nice-to-have, not required. |
| Cost | frozen-gate control ≈ 30 min × 3 seeds (now optional). |

**Confound identified and checked (external review + code check + re-run,
2026-09-04):** `alpha_logit` (`src/models/detector.py:118`,
`nn.Parameter(torch.zeros(()))`) was swept into the single AdamW param group in
`src/train.py` (`weight_decay=0.05`, applied uniformly, no exclusion for
scalars/biases/gates — standard recipes exclude these). Decoupled weight decay
pulls a scalar toward 0 every step regardless of the task gradient, so α sitting at
*exactly* 0.496–0.497 was plausibly a training artifact, not evidence, and the claim
was flagged as untrustworthy pending a check.

**Fix applied** (`utils.no_decay_param_groups`, commit `7bb4e11`, tests in
`tests/test_utils.py`) and **verified on Kaggle**: retrained `full` and `no_mask`
(seed 0 each, dataset-name `ffppfix`) with the corrected optimizer.
- `ffppfix_full_seed0`: α = 0.4947 (vs 0.4944–0.4964 range across the 9 original
  biased runs)
- `ffppfix_no_mask_seed0`: α = 0.4941 (same range)
- Test AUC essentially unchanged from the original biased runs (0.9974 vs 0.9969
  for `full`; 0.9972 vs 0.9973 for `no_mask`) — the fix didn't change training
  outcomes, only removed the decay-on-the-gate artifact.

**Conclusion: weight decay was NOT the primary driver.** Both fixed runs land
squarely inside the original range rather than drifting toward 1 (which removing an
artificial pull-to-zero should have allowed, if a real gradient signal were being
suppressed). This is 1 seed each with the fix — not fully exhaustive, more seeds
would make it airtight — but it is real evidence the "gate never engages" finding
survives its own strongest objection. Safe to use in the paper with a short methods
note describing the check (reviewers respond well to "we suspected X, checked, and
ruled it out" — it's the strongest form of this kind of claim).

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

## C4 — Non-discriminative, not redundant, not fragile: the frequency branch learns a genuinely distinct representation that just isn't useful for this task

**RESOLVED 2026-09-04.** CKA ran and **falsified** the redundancy hypothesis this
entry was originally set up to test (pre-registered falsification criterion: "CKA
is low ⇒ branches encode different things" — that's what happened). Updated
wording: do **not** say "the spatial CNN already captures/encodes the same
information" or call the frequency branch "redundant" — CKA(spatial, freq) ≈
CKA(spatial, random) on all 3 seeds (0.0051–0.0077 vs 0.0055–0.0058), i.e.
statistically no representational alignment. The correct, now evidence-backed
mechanism: **the two branches learn distinct representations; the frequency
branch's distinct information is simply non-discriminative for this task** (not
duplicated by the spatial branch — just not useful). `frequency_only` = 0.70 AUC is
consistent with this: some real signal, but not enough, and not overlapping with
what makes the spatial branch work.

| | |
|---|---|
| Experiment | (a) `src/spectra.py` real-vs-fake DCT gap + per-coefficient t-map at c23, c40, raw, and JPEG-{90..30}. (b) **CKA** between the spatial-branch and frequency-branch features (`src/cka.py`) — ✅ done, see above. (c) Test-time frequency-band ablation of a trained spatial model — if zeroing the "forensic" bands drops its AUC, it was already using them. |
| Data | FF++ c23 (have), c40, raw. |
| Seeds | spectra is dataset-level (seed only affects the split) — run on all 3 split seeds. |
| Confirms if | gap has small effect size (\|d\| < 0.3) AND changes little c23→c40 AND (CKA resolved: low, not high — see above) AND the spatial model already relies on the informative bands. |
| Falsified/updated | ~~"CKA is low ⇒ branches encode different things"~~ — this happened; the "redundant" framing is now retired in favour of "non-discriminative." |
| Status | 🟡 c23 spectra done (d ≈ 0.15, JPEG-robust). ✅ CKA done (3 seeds, falsifies redundancy). ⬜ c40/raw spectra, band-ablation. |
| Cost | spectra ≈ 5 min/condition; band-ablation ≈ 1 h. |

## C4b — "Spectral bias" diagnostic (added 2026-09-04, engages FreqDebias directly)

| | |
|---|---|
| Experiment | On `frequency_only` and `full`'s frequency branch (existing checkpoints, no retraining): zero each radial DCT band **one at a time** at test time, measure the AUC drop per band → a per-band importance profile. Once C6's Celeb-DF/DFDC crops exist: check whether the band the model relies on most in-domain also predicts the cross-dataset AUC drop. |
| Data | FF++ c23 (have) for the in-domain profile; Celeb-DF/DFDC (from C6) for the correlation check — no separate acquisition. |
| Confirms FreqDebias's "spectral bias" in our model if | the AUC-drop-per-band curve is sharply peaked on a narrow range (over-reliance on one band) AND that band's importance correlates with worse cross-dataset transfer. |
| Falsified if | importance is spread evenly across bands, or the most-relied-on band is *not* the one associated with the cross-dataset drop. |
| Status | ⬜ not started. |
| Cost | ~1–2 h, inference-only, reuses `src/spectra.py` band machinery. |
| Why it matters | Without this, citing FreqDebias is decorative. This makes the connection substantive — same vocabulary, our data, no need to reproduce their training pipeline. |

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
| Outcome tree (external review, 2026-09-04) | **(a) no c40 gain** → "the classic compression-robustness motivation does not survive matched modern controls" (strongest version of our current claim). **(b) c40 gain, and it survives to C6 cross-dataset** → "frequency has a narrow, compression-specific utility" (conditional-value framing, still a fine paper). **(c) c40 gain but C6 cross-dataset gets WORSE for the frequency model** → "frequency trades corruption robustness for domain robustness" — a genuinely interesting result, arguably the most interesting of the three, and one the current plan already collects the data to detect (C7 + C6 together). |
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
