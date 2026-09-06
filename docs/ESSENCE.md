# The essence — what this project is, in one place

The north-star document. If any decision, experiment, or paragraph doesn't serve
what's below, it's drift. Read this first; everything else is detail.

Last updated 2026-09-05.

---

## 1. The one sentence

> Conclusions about whether an explicit frequency pathway helps deepfake detection
> depend more on **how you evaluate** than on **what you build** — and we quantify
> both effects under a controlled protocol, using F3-Net's FAD component as the
> case study.

Deliberately **result-independent**: it holds whether FAD turns out to help, not
help, or remain inconclusive. That is the property that has protected this project
through three separate findings that overturned what we expected.

## 2. Title options

| | title | centre of gravity | risk |
|---|---|---|---|
| **1 (recommended)** | *Protocol Before Architecture: Evaluation Design and the Assessment of Frequency Features in Deepfake Detection* | protocol finding leads, frequency is the case study | low — works under any outcome |
| 2 | *Evaluation Units Matter: Cluster-Aware Assessment of Frequency-Domain Deepfake Detection* | statistical inference leads | needs the uncertainty-decomposition figure to earn it |
| 3 | *What Does Frequency Add? A Protocol-Controlled Assessment of DCT Pathways* | original question leads | weaker — the frequency null is semi-expected |
| 4 | *Detectable but Not Complementary: Frequency Features Under Leakage-Free Evaluation* | finding leads | **premature** — commits to a result our intervals don't yet support |

## 3. Motivation

Deepfake detection carries real stakes (fraud — the 2024 Arup HK deepfake-call
case; non-consensual imagery; journalistic verification; video-KYC). Practitioners
building detectors face a concrete recurring choice: **add an explicit
frequency-domain branch, or spend that budget elsewhere?** The literature has said
yes since 2020 and the design pattern is still active (Frequency Masking ICASSP'24,
FreqMamba, FreqDebias CVPR'25).

Answering that question requires an evaluation protocol capable of *resolving* it.
We found that the protocol commonly used in reimplementation pipelines is not —
frame-level splits inflate AUC by ~18 points and compress all architectural
differences into a ceiling regime, where a null result and a masked real effect are
observationally identical.

## 4. Application

| audience | what changes for them |
|---|---|
| Practitioners building detectors | Evidence on whether a frequency branch earns its cost — ours measures +31% FLOPs / +44% latency for the separate-branch design, ~+3% for FAD |
| Anyone benchmarking on FF++ | Frame-level splits inflate by ~18 points; grouped splits and cluster-aware intervals are necessary, not optional |
| Researchers running ablations | Uncertainty must be estimated over *test content*, not just seeds — otherwise intervals understate what they don't know |

**Do not claim** knowledge of proprietary deployment architectures, and **do not
claim** published FF++ results are inflated — reputable work uses the official
video-level splits. The finding is about what happens when they aren't used.

## 5. Prior work — what they say vs. what we say

### Base / foundation — the method we implement and test
**Qian et al., "Thinking in Frequency" (F3-Net), ECCV 2020.**
*They say:* frequency-aware decomposition (FAD) + local frequency statistics (LFS)
in a two-stream design improves face-forgery detection, with the largest gains on
low-quality (c40) media.
*We say:* under matched backbone, data, and budget, we cannot detect an advantage
from **FAD specifically** at c23 or c40.
⚠️ **We implement FAD only** — not LFS, not MixBlock. Report configurations as
`Xception` and `Xception + FAD`, never as "F3-Net". We test whether *its FAD
component* reproduces the reported low-quality advantage; we do not reproduce the
complete system.

### Recent target — the current state of the art
**Kashiani, Alipour Talemi, Afghah, "FreqDebias," CVPR 2025.**
*They say:* frequency reliance produces *spectral bias* — over-reliance on narrow
bands that harms generalisation — and propose augmentation + consistency
regularisation to correct it.
*We say:* consistent in spirit, and it sharpens our framing — if the field's newest
work needs dedicated machinery to make frequency reliance safe, the *unregularised*
frequency pathway (F3-Net's, and ours) is exactly what should be questioned. A
productive contrast, **not an ally**.

### Support — closest intellectual kin
**Dong et al., "Implicit Identity Leakage" (CADDM), CVPR 2023.** *They say:*
detectors latch onto identity as a shortcut, harming generalisation.
*We say:* identity leakage corrupts **evaluation** too, not just generalisation —
and we quantify by how much. Now the most load-bearing support paper, because our
identity-disjoint splitting is a direct application of their thesis.

**Shiohara & Yamasaki, "SBI," CVPR 2022.** *They say:* spatial self-blending
generalises best, with no real manipulated training data.
*We say:* consistent — the field's strongest generalisation comes from spatial
artifact modelling.

Antecedents (cite, don't reproduce): Frank et al. ICML 2020; Zhang et al. WIFS
2019; Durall et al. CVPR 2020. Full annotated list: `docs/related-work.md`.

## 6. What we're trying to achieve

Three contributions, in priority order:

1. **A controlled measurement of protocol effect.** Frame-level vs video-level vs
   identity-disjoint splitting, across matched architectures — how much does the
   evaluation unit move absolute performance, and does it move *conclusions*?
2. **A cost-accounted, cluster-aware assessment of FAD.** Paired comparison with
   intervals estimated over independent test clusters, against a prespecified
   practical-effect margin.
3. **Evidence on complementarity.** Whether a frequency representation can be
   distinct and weakly discriminative without adding incremental predictive value
   (late fusion, gate α-sweep).

## 7. What is already achieved

| finding | status | evidence |
|---|---|---|
| Frame-level splits inflate FF++ AUC by 17.6–19.2 pts, consistent across 3 architectures | **solid** (c40; c23 running) | `results/in_domain_c40_vid/` |
| Protocol changed absolute performance ~18 pts but not the FAD-vs-Xception conclusion | **moderate** — shown for one contrast, 3 configs | same |
| No *detectable* FAD benefit at c23 or c40 | **bounded, not established** — video-level CI [−0.048, +0.038] still admits +3.8 pts | `results/in_domain_c40_vid/` |
| Late fusion of independently-trained spatial + frequency scores adds nothing (4 dp, 3 seeds, regularised and not) | **strong** (frame-level; re-run queued) | `results/analysis/late_fusion/` |
| Learned fusion gate never leaves 0.5; weight-decay confound tested and ruled out | **moderate** — needs α-sweep | `results/analysis/fusion_alpha.csv` |
| Frequency branch is representationally *distinct* from spatial, not redundant | **suggestive only** — needs untrained-control matrix; likely supplementary | `results/analysis/cka/` |
| Frequency-only best on the crudest manipulation, worst on the subtlest | **descriptive** — 4 methods is too few to generalise | `results/analysis/permanip/` |
| Separate frequency branch costs +31% FLOPs / +44% latency; FAD only ~+3% | **solid** | `results/analysis/efficiency/` |
| Real-vs-fake DCT gap exists but is small (d≈0.15) and survives JPEG-q30 | **solid**, though c40 (H.264) validation still owed | `results/analysis/spectra/` |

**Scale:** ~50 training runs, 10 configurations, ~32 GPU-hours
(`docs/EXPERIMENTS.md`). **Three self-caught errors** documented and corrected
(weight-decay confound; CKA falsifying our own redundancy hypothesis; split
leakage) — `docs/progress-log.md`.

## 8. Validation still required

**Essential before submission**

| # | test | why |
|---|---|---|
| V1 | Per-prediction dumps + three-unit cluster bootstrap (frame/video/identity) | Current intervals use the wrong unit. Infrastructure built (`src/cluster_boot.py`), not yet run. |
| V2 | Seen-video vs unseen-video matched test sets, one trained model | Isolates the leakage effect without confounding it with two differently-trained models. ~1 h. |
| V3 | Identity-disjoint (L3) splits | Video grouping keys on target only; source-side overlap remains (CADDM's leak). |
| V4 | Video-level AUC *and* frame AUC on the same held-out videos | Separates aggregation effect from split effect; needed for comparability with published work. |
| V5 | c23 and c40 under the same grouped splits and seeds | Tests architecture × compression as an interaction, not two descriptive numbers. |
| V6 | Prespecified practical-effect margin + equivalence test | "Not significant" ≠ "no effect". |

**Strongly recommended**

- V7 — learning curve over independent video count (60/120/180/240): disentangles
  leakage from sample-size collapse, and tells us whether scaling the data is worth
  the compute *before* spending it.
- V8 — epoch audit: best val epoch was 3 at video level while we run 15. Fixing
  this roughly halves the cost of everything downstream.
- V9 — gate α-sweep from 0 to 1 (post-hoc): far stronger than "α didn't move".
- V10 — spectral survival under real H.264 c40 rather than simulated JPEG-q30.

**Explicitly out of scope** (this is not a benchmark paper): full FF++ (~1000
pairs), multiple external datasets, many architectures, reproducing full F3-Net
with LFS + MixBlock, 10+ seeds.

## 9. Claim-strength discipline

Wording that survives review, and what it replaces:

| don't say | say |
|---|---|
| "FAD provides no benefit" | "We observed no detectable FAD benefit; our interval excludes effects larger than X." |
| "The comparison is unanswerable" | "Ceiling compression severely limits resolution." |
| "Relative comparisons are robust" | "The FAD-vs-Xception conclusion was unchanged in this experiment." |
| "Frame-level splitting is a competing protocol" | "FF++ prescribes video-level splits; we quantify the cost of the crop-level shortcut." |
| "F3-Net" (for our config) | "Xception + FAD" |
| "matched inference cost" | "cost-accounted" |
| "inverse of the premise" | "contrary to the expectation that frequency cues aid subtle manipulations" |

## 10. Target

4–5 page IEEE workshop / short-paper venue (media forensics workshop, WIFS,
IH&MMSec, ICASSP/ICIP-style). Structure: protocol result as opening motivation,
paired cluster-aware frequency result as the central scientific test, mechanistic
evidence compressed into one multi-panel figure. **Not** seven equal
contributions.
