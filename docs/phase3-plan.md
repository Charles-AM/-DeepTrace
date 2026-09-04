# Phase 3 — analysis paper: next steps + supporting evidence

Supersedes the Phase-2 execution plan in `docs/phase2-runbook.md` (that phase is
done; result was negative — see `docs/progress-log.md` 2026-09-03).

**Deliverable:** a controlled trade-off study.
**Central RQ** (tightened 2026-09-04, external review): *Under matched training
data, backbone capacity, and inference constraints, does an explicit DCT-based
frequency pathway provide incremental robustness beyond a spatial detector,
especially under compression?* — "incremental" is deliberate: the claim is about
predictive value added on top of a spatial baseline, not about whether frequency
information is discriminative in isolation (it is — see C1 in `validation-plan.md`).
Survives a positive OR negative c40 result.

**Central claim (adopt near-verbatim, external review):** *Explicit frequency
modelling is not intrinsically beneficial for face-forgery detection. In controlled
FF++ experiments, spectral cues are measurable and survive compression, yet
DCT-based frequency pathways provide little or no incremental accuracy over matched
spatial baselines and may reduce cross-dataset generalisation. Their value, if any,
is therefore conditional on degradation regime and must be weighed against
additional compute and architectural complexity.*

**Base/foundation (implemented + directly engaged):** F3-Net (ECCV 2020) — we build
its FAD architecture and test its literal c40 claim.
**Recent target — a productive CONTRAST, not an ally (corrected 2026-09-04):**
FreqDebias (CVPR 2025) argues frequency modelling *can* help when its "spectral
bias" failure mode is explicitly corrected for (Fo-Mixup + dual consistency
regularisation). We show a *simpler, matched-backbone, no-special-machinery*
frequency branch fails outright — i.e. whatever value frequency carries apparently
needs substantial extra engineering to realise, which itself undercuts the
"obvious win" assumption behind F3-Net-style architectures (ours included). Do not
describe FreqDebias as "supporting our scepticism" — it explicitly does not share
our conclusion, it just diagnoses a related problem.
**Antecedents:** Frank et al. (ICML 2020), Zhang et al. (WIFS 2019), Durall et al.
(CVPR 2020).
**Support (co-primary):** SBI (CVPR 2022), **CADDM/Implicit Identity Leakage (Dong
et al., CVPR 2023)** — on-task, positive result, independent mechanism (identity
leakage, not frequency). Further: Ojha et al. (CVPR 2023), Gragnaniello et al.
(ICME 2021), Corvi et al. (ICASSP 2023), "Fake or JPEG?" (2024). Full list:
`docs/related-work.md`.

---

## What we already have (paper-ready)

- In-domain FF++ c23, 3 seeds, 10 configs → `results/ablation_table.md`.
  Xception 0.9977, F3-Net 0.9971, proposed 0.9949; F3-Net > proposed p=0.049;
  no frequency component significant; `frequency_only` = 0.70 (near chance).
- Cross-dataset **seed 0 only**: Celeb-DF — Xception 0.827, F3-Net 0.820,
  freq-hybrid 0.744; SAS hurts Celeb-DF transfer. DFDC — all ~0.73–0.75.

## The gap: our evidence is c23-only and single-seed cross-dataset.
F3-Net's headline claim is about **c40**. We have not tested it.

---

## DONE 2026-09-03 (results in `results/`)

- ✅ **#1 Robustness sweep** — seed-0 all perturbations + 3-seed JPEG. F3-Net's JPEG
  edge was a seed-0 artifact; over 3 seeds all models tie under JPEG (~0.85 AUC at
  sev4). Proposed `full` reliably ~2 AUC below the spatial baselines.
- ✅ **#5 Spectral analysis** (`src/spectra.py`, rewritten). A real but small
  (Cohen's d ≈ 0.15) real-vs-fake DCT gap; JPEG-q30 attenuates it only ~15–25 %.
  Reframes the thesis: signal is *marginal and redundant*, not destroyed by
  compression.
- ✅ **#6 Fusion gate α** — 0.496 ± 0.001 every config/seed; never left init.
- ✅ **Per-manipulation breakdown** (3 seeds) — spatial Xception best on all 4 FF++
  methods incl. NeuralTextures; `frequency_only` best on Deepfakes / worst on NT.

Still open: #2 (c40 training — now the pivotal one), #3 (cross-dataset seeds 1&2),
#4 (matched-backbone grid), #7 CKA, #8 band-ablation, #9 leave-one-out, #10 mask
figures, Tier 3. A *proper* spectral figure now exists; a c40 spectra comparison
(needs c40 crops) would still add value.

## Experiments — ranked

### Tier 1 — required for the claim to hold

| # | Experiment | Command / notes | Cost |
|---|---|---|---|
| 1 | **Compression-robustness sweep** on the 30 existing checkpoints — JPEG {90,70,50,30} + blur/noise/resize/contrast. Question: does F3-Net's FAD degrade *less* than Xception as quality drops? | `python -m src.robustness --runs <key runs> --dataset-name rvf --seed 0 --limit 3000 --results-root /kaggle/input/... --out-dir /kaggle/working` | ~1–2 h |
| 2 | **One c40 training run** — Xception vs F3-Net only, same subset re-encoded at c40 (or download c40 crops). Directly engages F3-Net's main result. | re-run `ffpp_fastdl` with `--compression c40`, re-extract, train 2 configs × 3 seeds | ~3–4 h |
| 3 | **Cross-dataset seeds 1 & 2** — error bars on the Celeb-DF / DFDC table. | `src.cross_dataset` on seed-1 and seed-2 checkpoints | ~35 min |
| 4 | **Matched-backbone grid** — add `Xception + learnable mask` and `ResNet-18 + FAD` so the table is {ResNet-18, Xception} × {spatial, +FAD, +mask}. Frequency should fail to help in every cell. | new configs in `src/config.py`, 4 runs × 3 seeds | ~3 h |
| 5 | **Data spectral analysis figure** — mean radial FFT/DCT power of real vs fake crops at raw / c23 / c40. Show the real-vs-fake gap that exists at raw is gone at c40. Explains *why* frequency fails on deployed (compressed) media. | new script `src/spectra.py` | ~1 h |

### Tier 2 — makes it an analysis paper, not just a table

| # | Experiment | Why |
|---|---|---|
| 6 | **Report the learned fusion gate α** (already logged in checkpoints). If α → ~1, the hybrid itself down-weights the frequency branch to nothing. | free — read from `best.pt` |
| 7 | **CKA(spatial features, frequency features)** on the test set. High alignment ⇒ the frequency branch is redundant, not complementary. | ~1 h, `src/cka.py` |
| 8 | **Test-time frequency-band ablation of a trained spatial model** — zero radial DCT bands of the input, measure AUC drop. Shows the spatial CNN already uses the "forensic" bands. | ~1 h |
| 9 | **Cross-manipulation leave-one-out** within FF++ (train on 3 methods, test on the 4th). Does frequency help the hardest generalisation split? | ~2 h, re-split existing crops |
| 10 | **Mask-interpretability figures** across seeds/datasets — show the learned mask is inconsistent and doesn't transfer. | ~10 min, `src.visualize` |

### Tier 3 — strengthen / preempt reviewers

| # | Experiment | Why |
|---|---|---|
| 11 | 5 seeds (not 3) for the two headline comparisons; bootstrap CIs on the AUC *difference*; report Cohen's d. | tighter significance |
| 12 | Full F3-Net (FAD + LFS + MixBlock), not just FAD. If even that ties Xception, very strong. | closes the "you crippled F3-Net" objection |
| 13 | DF40 diffusion subset as a 3rd cross-dataset target. | modern generators; ties to Corvi et al. |
| 14 | "Published numbers vs our controlled numbers" table — F3-Net/SPSL as reported vs in our harness. | shows the gap is protocol, not re-implementation error |
| 15 | Add EfficientNet-B4 as a higher-capacity backbone to rule out "the backbone was too weak to need frequency". | capacity control |

---

## Fixed order to execute

1. Save `celebdf_crops` / `dfdc_crops` as **private** Kaggle datasets (currently
   only in `/kaggle/working`, lost on session end). *(EULA: never make public.)*
2. Tier-1 #1 robustness → #3 cross-dataset seeds → #5 spectral figure (no training).
3. Tier-1 #2 c40 run, #4 matched-backbone grid (training).
4. Tier-2 #6–#10.
5. Tier-3 as time allows.
6. Write-up: fill `results/` tables into the draft, verify every citation in
   `docs/related-work.md`, draft LaTeX (intro → related work → controlled protocol →
   results → analysis (#5–#8) → limitations → conclusion).

## Framing — trade-off study (tightened 2026-09-03)

**Central research question:** *Under constrained inference budgets and increasing
levels of video/image compression, does adding an explicit frequency-domain branch
to a spatial CNN provide enough robustness improvement to justify its computational
overhead?*

Not "we built a better detector", not "frequency is useless". A **trade-off study**
of a real architectural decision — the intersection of C5 (compression robustness) +
C7 (c40) + C8 (compute cost). Accommodates any c40 outcome ("worth it above
compression X, not below" is still a clean result). The negative findings (C1–C4)
are supporting evidence, not standalone claims.

**Premise (conservative):** practical detectors balance performance vs compute /
latency while staying robust to compression and post-processing; the design choice
is whether a spatial CNN gains enough from a frequency branch to justify its cost.
**Do not** assert knowledge of proprietary platform architectures/workloads — the
literature establishes these as important *conditions*, nothing more.

**Design pattern is current** (cite to show it's not a 2020 relic — verify first):
TSFF-Net, WGN, DSTF-Net, FreqMamba, hybrid spatial-frequency EfficientNet.

**Base/target updated 2026-09-04:** base/foundation is now **F3-Net (ECCV 2020)**
(implemented + c40-tested directly); recent target is **FreqDebias (CVPR 2025)**,
which diagnoses "spectral bias" — frequency over-reliance hurting generalisation —
independent support for our scepticism, cited not reproduced (its Fo-Mixup + dual
consistency regulariser is out of scope for our compute/timeline). We engage it via
its own diagnostic lens: the DCT band-ablation in C4 (`docs/validation-plan.md`)
tests whether *our* frequency branch shows the spectral bias FreqDebias describes.
Full rationale in `docs/related-work.md`.

## Paper positioning & the constructive turn (decided 2026-09-03)

**Why "this is already known" does not sink it:** prior critiques of frequency
detection (Frank et al. ICML'20, Gragnaniello et al. ICME'21, "Fake or JPEG?" '24)
are about **GAN-generated whole images**. Our target is **face-swap deepfake video**
(FF++) — a different problem, and the frequency-for-face-forgery line (F3-Net, SPSL,
+ 2024 follow-ups) has not had a controlled mechanistic re-evaluation.

**The three findings that are ours, not folklore:** (1) the learnable fusion gate
never engages (α = 0.496 ± 0.001, all seeds); (2) per-manipulation inversion —
frequency-only best on the *crudest* forgery, worst on the *subtlest*; (3)
"redundant, not fragile" — the spectral signal survives JPEG (d ≈ 0.15, barely
attenuated), so it fails from redundancy, not brittleness (a distinct claim from the
prior work).

**Constructive turn — chosen:** *reproducibility framing as the spine* ("F3-Net's
c40 gains do not replicate under matched training") *+ efficiency angle as a
secondary section* ("the frequency pathway is a pure computational cost — F3-Net
≈ 2× Xception FLOPs for zero gain"). Scoping angle ("find the regime where
frequency wins") rejected as primary — open-ended search, and current data shows no
regime where frequency wins in absolute terms.

**Data-readiness:** in-domain (matched, 3-seed), fusion-gate, per-manipulation, and
JPEG-robustness claims are all supported by committed `results/`. Still needed:
c40 run (pivotal), cross-dataset seeds 1 & 2, the params/FLOPs/latency table.

## Venues

Target a workshop or short-paper track that welcomes negative / analysis results:
CVPR/ICCV/ECCV workshops on media forensics (e.g. WMF, DFAD), IEEE WIFS, IH&MMSec,
or a "Reproducibility / negative results" track. DeepfakeBench (NeurIPS D&B 2023) is
the methodological precedent to cite and match.

## Known repo state

- HEAD adds `--out-dir` to `src/cross_dataset.py` and `src/robustness.py`
  (they crashed writing into a read-only mounted `--results-root`).
- `src/extract_faces_v2.py` is the correct extractor (path-slug names).
  `src/extract_faces.py` is buggy — deprecate.
- `frequency_branch.py` v1 is broken (GAP over a raw DCT map discards frequency
  position). A v2 (DCT → mask → IDCT → spatial CNN, FAD-style) is designed but not
  built; only needed if we want a *working* frequency branch to also lose fairly.
