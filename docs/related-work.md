# Related work — base paper, target, and supporting literature

Positioning for the controlled-study / trade-off paper.
**Central research question:** under constrained inference budgets and increasing
compression, does adding an explicit frequency-domain branch to a spatial CNN
provide enough robustness improvement to justify its computational overhead?

All citations below were verified against dblp / CVF / MLR / arXiv on 2026-09-03/04.
Re-verify before final submission.

---

## Base / foundation paper (the method we implement and directly engage)

**Qian, Yin, Sheng, Chen, Shao. "Thinking in Frequency: Face Forgery Detection by
Mining Frequency-Aware Clues (F3-Net)." ECCV 2020.**

Two frequency streams (FAD: frequency-aware decomposition; LFS: local frequency
statistics) on an Xception backbone. Headline claim: beats spatial baselines on
FaceForensics++ **at every compression level, with the largest lead on low-quality
(c40) media**. This is *our* base paper because we don't just cite it — we
**implement its FAD architecture** (`src/models/f3net.py`) and **directly test its
central claim** with a matched-backbone c40 training run (C7 in
`docs/validation-plan.md`). Everything in our study is built to engage this one
paper's method and this one paper's headline result.

Antecedents (the deeper premise F3-Net itself rests on — cite, don't reproduce):

- **Frank, Eisenhofer, Schönherr, Fischer, Kolossa, Holz. "Leveraging Frequency
  Analysis for Deep Fake Image Recognition." ICML 2020.** arXiv:2003.08685 · code:
  RUB-SysSec/GANDCTAnalysis. Established DCT-domain analysis — the transform in our
  custom layer — as a detection tool; GAN images carry a regular grid artifact in
  the DCT spectrum. Our learnable per-coefficient DCT mask is a parametrised
  generalisation of their *fixed* DCT analysis.
- **Zhang, Karaman, Chang. "Detecting and Simulating Artifacts in GAN Fake Images."
  WIFS 2019.** arXiv:1907.06515. Up-sampling produces *spectral replication*
  artifacts; introduces the **AutoGAN** simulator — the direct ancestor of our
  Spectral Artifact Simulation (SAS).
- **Durall, Keuper, Keuper. "Watch Your Up-Convolution: CNN Based Generative Deep
  Neural Networks Are Failing to Reproduce Spectral Distributions." CVPR 2020.**
  arXiv:2003.01826. A spectral classifier detects the up-convolution mismatch "with
  up to 100% accuracy" on public benchmarks — the strong claim our data pushes back
  on.
- **Odena, Dumoulin, Olah. "Deconvolution and Checkerboard Artifacts." Distill,
  2016.** The mechanism: fixed-stride transposed conv leaves periodic checkerboard
  patterns.

---

## Recent target (the current state of the art we position against)

**Kashiani, Alipour Talemi, Afghah. "FreqDebias: Towards Generalizable Deepfake
Detection via Consistency-Driven Frequency Debiasing." CVPR 2025.**
pp. 8775–8785 · arXiv:2509.22412 ·
[CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Kashiani_FreqDebias_Towards_Generalizable_Deepfake_Detection_via_Consistency-Driven_Frequency_Debiasing_CVPR_2025_paper.html)

Defines **spectral bias**: detectors over-rely on specific frequency bands, which
*restricts* generalisation to unseen forgeries — a new failure mode diagnosed five
years after F3-Net. Fixes it with **Forgery Mixup (Fo-Mixup)** augmentation that
diversifies training-sample frequency characteristics, plus a **dual consistency
regulariser** (local, via class-activation maps; global, via a von Mises-Fisher
distribution over embeddings).

Why it's the right recent target — **a productive CONTRAST, not an ally** (wording
corrected 2026-09-04): it is the most current major statement in frequency-domain
face-forgery detection, closing the base-paper-recency gap (F3-Net 2020 →
FreqDebias 2025). But its conclusion is *not* the same as ours: FreqDebias argues
frequency modelling **can** help, provided its "spectral bias" failure mode is
explicitly corrected for (Fo-Mixup + dual consistency regularisation); we test a
*simpler, matched-backbone* frequency branch with no such correction and find it
adds nothing. The useful framing is not "FreqDebias agrees with us" but: their
paper's existence and engineering complexity is itself evidence that *naive*,
unregularised frequency reliance (which is what F3-Net's FAD, and our custom DCT
branch, are) does not straightforwardly pay off — real machinery was needed to make
theirs work. That is consistent with, but distinct from, our own finding.

**Scope note — do not fully reproduce.** Fo-Mixup + dual CR is materially heavier to
train than F3-Net's FAD. Engage it via (a) citation/positioning, and (b) their own
diagnostic lens applied cheaply to our data: the DCT band-ablation and
per-coefficient t-map we already compute (`src/spectra.py`, `results/analysis/`) is
a direct test of whether *our* frequency branch exhibits the spectral bias they
describe — same vocabulary, no need to reproduce their training pipeline.

Secondary/contemporaneous targets (frequency-domain face-forgery detectors, for the
"the design pattern is current, not a 2020 relic" argument — verify before citing):

- **Liu et al. "Spatial-Phase Shallow Learning (SPSL)." CVPR 2021.** Phase spectrum
  + shallow net.
- **Li et al. "Frequency-aware Discriminative Feature Learning (FDFL)." CVPR 2021.**
- **Doloriel, Cheung. "Frequency Masking for Universal Deepfake Detection."
  ICASSP 2024.** arXiv:2401.06506.
- TSFF-Net, WGN, DSTF-Net, FreqMamba, hybrid spatial-frequency EfficientNet —
  recent dual-domain architectures (unverified, confirm authors/venues before use).

---

## Supporting papers (independent evidence for the claim)

### A. Spatial / semantic representations generalise better than spectral ones

**Co-primary supports (updated 2026-09-04):**

- **Shiohara, Yamasaki. "Detecting Deepfakes with Self-Blended Images (SBI)."
  CVPR 2022 (oral).** arXiv:2204.08376. ~93% AUC on Celeb-DF with **no real
  deepfakes in training** — a purely spatial blending-artifact method that is still
  among the strongest cross-dataset results.
- **Dong, Wang, Ji, Liang, Fan, Ge. "Implicit Identity Leakage: The Stumbling Block
  to Improving Deepfake Detection (CADDM)." CVPR 2023**, pp. 3994–4004.
  code: megvii-research/CADDM. Diagnoses a *different* generalisation-killer than
  frequency fragility — binary classifiers spuriously learn identity (who) rather
  than forgery (what) — and shows a spatial artifact-detection module that avoids
  this beats prior SOTA in-dataset **and** cross-dataset, on the same FF++/Celeb-DF/
  DFDC suite we use. Strongest support paper for our claim: same task (not generic
  GAN-image detection like Ojha/Corvi below), a positive result (not just "X is
  fragile"), independent mechanism. **Verify their exact reported number against
  F3-Net specifically before quoting it** — not confirmed in our source search.

Both above are *on-task* (face-forgery video, our exact benchmark suite) and
*positive* (a working spatial method beating SOTA), which makes them stronger
support than a purely critical paper. Together with **FreqDebias** (recent target,
above) they give three independent mechanisms for the same directional finding:
we say the frequency signal is real but small and, per CKA, representationally
*distinct* from (not redundant with) what a spatial CNN learns — it's simply not
discriminative; CADDM says the real generalisation killer is identity leakage,
unrelated to frequency; FreqDebias says frequency reliance itself causes "spectral
bias." Three groups, three mechanisms, one direction — worth a paragraph in the
discussion section.

**Further support:**

- **Ojha, Li, Lee. "Towards Universal Fake Image Detectors that Generalize Across
  Generative Models." CVPR 2023.** arXiv:2302.10174. A *frozen* CLIP (semantic,
  non-spectral) feature space with nearest-neighbours generalises far better than
  trained spectral/artifact features (+15 mAP, +26% acc on unseen models). Generic
  GAN/diffusion-image detection, not face-forgery-video specific — cite as broader
  context, not a same-task result.
- **Yan et al. "UCF: Uncovering Common Features for Generalizable Deepfake
  Detection." ICCV 2023.** Spatial-domain disentanglement of content vs forgery,
  same task as CADDM.

### B. Frequency cues are fragile / biased — compression, post-processing, transfer

- **FreqDebias (see above)** — spectral bias as an intrinsic liability of
  unregularised frequency reliance.
- **Gragnaniello, Cozzolino, Marra, Poggi, Verdoliva. "Are GAN Generated Images
  Easy to Detect? A Critical Analysis of the State-of-the-Art." ICME 2021.**
  arXiv:2104.02617. Detectors — frequency ones included — collapse under realistic
  conditions. Methodological precedent for a controlled critical re-evaluation.
- **Corvi, Cozzolino, Zingarini, Poggi, Nagano, Verdoliva. "On the Detection of
  Synthetic Images Generated by Diffusion Models." ICASSP 2023.** GAN-era spectral
  fingerprints do **not** transfer to diffusion images — generator-specific, so a
  frequency detector's inductive bias is a liability out of distribution.
- **Grommelt, Weiss, Pfreundt, Keuper. "Fake or JPEG? Revealing Common Biases in
  Generated Image Detection Datasets." 2024 (arXiv:2403.17608).** Frequency
  detectors frequently learn JPEG/compression statistics, not generation artifacts.

### C. Controlled-benchmark methodology (how to run a fair comparison)

- **Yan, Zhang, Yao, Fu, Wu, Yang, et al. "DeepfakeBench: A Comprehensive Benchmark
  of Deepfake Detection." NeurIPS 2023 (Datasets & Benchmarks).** arXiv:2307.01426.
  Standardised preprocessing, splits, and backbones; our results should track its
  leaderboard for the detectors we re-implement.

### DeepfakeBench — what it standardises, checked 2026-09-11

**Why this matters more than a normal citation.** The paper's positioning rests on
what the field's reference benchmark does and does not standardise. That sentence
is load-bearing, so it is recorded here with its evidence level rather than
asserted.

**Standardises** (from the abstract, arXiv:2307.01426):

- "a unified data management system to ensure consistent input across all detectors"
- "an integrated framework for state-of-the-art methods implementation"
- "standardized evaluation metrics and protocols"

**Does not appear to standardise** (abstract silent; repository documentation
consulted): multiple random seeds, standard deviations, confidence intervals,
significance testing, or the statistical unit over which variance is computed.

**The sharper finding.** The repository documentation states the primary reported
metric is **frame-level AUC** (video-level AUC, ACC, EER, PR and AP are also
computed). That is precisely the estimand our V4 result shows changes the
architectural contrast in all five c40 seeds while preserving sign
(`results/analysis/aggregation/`).

So the positioning is a claim about a **concrete reported choice**, not merely
about an absence:

> The field's reference benchmark standardises data processing, implementations
> and metrics, and reports frame-level AUC as its headline metric — the estimand
> we show changes the estimated contrast — while specifying nothing about the unit
> over which uncertainty is computed.

✅ **CHECK 1 CONFIRMED 2026-09-11** from the paper's main results table
(within-domain and cross-domain evaluation, 15 detectors × 15 columns). **Every
cell is a bare four-decimal point value.** No standard deviation, error bar,
interval, or run count anywhere in the table or its caption. The positioning
sentence may be used in its strong form.

✅ **CHECK 2 CONFIRMED 2026-09-11** from the paper body, Evaluation and Analysis
Module. Verbatim:

> "we employ 4 widely used evaluation metrics: accuracy (ACC), the area under the
> ROC curve (AUC), average precision (AP), and equal error rate (EER) … it is
> notable that there is an inconsistency in the usage of these evaluation metrics
> in the community, some are at the frame level, while others are at the video
> level, leading to unfair comparisons. **Our benchmark currently adopts the frame
> level evaluation to build a fair basis for comparison among detectors.**"

Two things follow, and the second is the more important.

**Uncertainty.** Four metrics are enumerated — ACC, AUC, AP, EER. **None is a
measure of variability.** The gap is now confirmed from the paper body, not
inferred from silence in an abstract.

### The positioning this actually licenses — and it is better than the original one

The naive framing ("the field ignores the frame/video distinction") is **wrong and
would be unfair**. DeepfakeBench identifies the inconsistency explicitly, calls it
a source of "unfair comparisons", and resolves it deliberately.

What they do is standardise the choice **for comparability**. What our results show
is that the same choice has two further consequences they do not address:

1. **It changes the estimated contrast, not merely its comparability.** Our V4
   result moves the architectural estimate in all five c40 seeds while preserving
   sign (`results/analysis/aggregation/`). Frame-pooled and video-aggregated AUC
   are different estimands, so standardising on one makes comparisons consistent
   without making them equivalent to the other.
2. **Frame-level pooling is precisely what makes the units non-independent.**
   Reporting at the frame level is what licenses treating 3,000 correlated crops
   as 3,000 observations — the pseudoreplication our contribution 2 measures.

So the honest and much stronger claim:

> The field's reference benchmark **recognises** the frame-versus-video
> inconsistency and standardises on frame-level evaluation to make comparisons
> fair. Standardising the choice resolves comparability, but the choice also
> determines the estimand and the dependence structure of the evaluation units —
> neither of which is addressed, and neither of which a consistent choice fixes.

That framing credits them correctly, is verifiable from their own text, and leaves
our contribution precisely delineated: **consistency is not the same as correct
uncertainty modelling.**

### What the same table independently shows about frequency methods

Read from the within-domain columns of that table:

| detector | type | FF-c23 | FF-c40 | Δ vs Xception c40 |
|---|---|---|---|---|
| Xception | naive baseline | 0.9637 | 0.8261 | — |
| **F3Net** | frequency | 0.9635 | 0.8271 | **+0.0010** |
| SPSL | frequency | 0.9610 | 0.8174 | −0.0087 |
| SRM | frequency | 0.9576 | 0.8114 | −0.0147 |
| UCF | spatial | 0.9705 | 0.8399 | +0.0138 |

**Three findings, all directly relevant:**

1. **An independent standardised reimplementation of F3Net beats the Xception
   baseline by +0.0010 at c40** — about **14× smaller** than the +0.014 FAD
   ablation gain reported in the original paper, and −0.0002 (negative) at c23.
2. **Of three frequency detectors at c40, two are below the spatial baseline.**
   The best performer in the table is UCF, a spatial method, at +0.0138.
3. **Both +0.0010 and +0.014 lie inside our crossed interval [−0.0358, +0.0180].**
   Our study cannot separate them — which is exactly the resolution argument, now
   with two independent published estimates of the same quantity differing by an
   order of magnitude and neither reported with uncertainty.

**External validity.** Their Xception FF-c40 is 0.8261; our frame-pooled c40
Xception is 0.81347, a difference of 0.013 on a 300-sequence subset versus full
FF++. Close enough to indicate our pipeline is sound.

⚠️ **SCOPE IS AMBIGUOUS — paper and code disagree. Checked 2026-09-11.**

**Their paper** describes F3Net as the two-branch system and states how it was
built:

> "F3Net [32]: uses cross-attention two-stream networks to collaboratively learn
> frequency-aware clues from two branches: FAD and LFS … **The code for this
> detector is not publicly available, we re-implement it carefully following the
> instructions and settings in the original paper**"

**Their released code**, `training/detectors/f3net_detector.py`, states the
opposite on both counts:

> "We replicate the results by solely utilizing the **FAD branch**, following the
> **reference GitHub implementation**"

and contains `FAD_Head` with no `LFS_Head`, no `MixBlock`, and no branch-mode flag.

Two contradictions: **what was implemented** (two branches vs FAD alone) and
**what it was built from** (the paper, because no code existed, vs a reference
GitHub implementation).

⚠️ Both readings come from a single source each, read through a summarising model.
We cannot establish which corresponds to the code that produced the published
table, and the repository may have changed since publication.

### Why our use of the number survives the ambiguity

The comparison does not depend on resolving it:

| if their F3Net is… | original's claim | their measured effect | ratio |
|---|---|---|---|
| FAD only | +0.014 (FAD ablation) | +0.0010 | **14× smaller** |
| the full two-branch system | +0.040 (full system) | +0.0010 | **40× smaller** |

**Either way, an independent standardised reimplementation reports an effect an
order of magnitude or more below the original**, with no uncertainty attached to
either figure. That is the point we need, and it is robust to the ambiguity.

### How to cite it

Use the conservative form, and disclose the ambiguity in one clause rather than
making it a finding:

> An independent standardised benchmark reports F3Net at 0.8271 against an
> Xception baseline of 0.8261 on FF-c40, a difference of +0.0010 — an order of
> magnitude below the gain reported in the original ablation. (Their paper
> describes the two-branch method while their released implementation states the
> FAD branch alone; the comparison holds under either reading.)

Do **not** frame the paper/code discrepancy as a criticism of DeepfakeBench. It is
a disclosure that protects our comparison, not a result of ours, and treating it as
a gotcha would be both unfair and a distraction from the argument.

⚠️ **One implementation difference, in our favour to disclose.** Their `FAD_Head`
uses **four filters producing a 12-channel** input (bands at 0–1/16, 1/16–1/8,
1/8–1). Ours (`src/models/f3net.py`) uses **three radial bands producing
9 channels** (`_BANDS = ((0.0, 0.10), (0.10, 0.35), (0.35, 1.01))`, `in_chans=9`),
with hand-designed edges and a small learnable perturbation.

Both call themselves FAD. Neither is wrong — the original describes learnable
band-pass filters without fixing the partition — but it means **three
implementations of "the same component" differ in band count and edges**. Disclose
this when citing their number; it is also a minor point in favour of the paper's
thesis, since the component's definition is itself less determinate than a single
reported gain suggests.

⚠️ **Remaining caveats.** Their split and training protocol may differ from the
original paper's. And their numbers carry no uncertainty at all — with single point
values there is no way to tell whether +0.0010 is distinguishable from zero, or
from +0.014.

- **Yan et al. "DF40: Toward Next-Generation Deepfake Detection." NeurIPS 2024.**
  40 forgery methods incl. diffusion/editing — a modern cross-generator target.

### D. Nuance — a spatial method that exploits an upsampling (frequency) phenomenon

- **Tan, Wei, Yao, et al. "Rethinking the Up-Sampling Operations in CNN-based
  Generative Networks for Generalizable Deepfake Detection (NPR)." CVPR 2024.**
  arXiv:2312.10461. The useful signal from upsampling is captured best by a
  **pixel-domain** operator, not an explicit frequency transform — the phenomenon
  is real, the frequency *representation* is not the way to use it.

---

## Citations this file was missing (added 2026-09-11)

This document was written while the project was a **method** paper. The
contributions are now statistical, and three categories of essential citation were
absent entirely.

✅ **Verified 2026-09-15** against dblp / CVF / publisher pages — 15 of 22 entries
below now carry confirmed authors, venue, volume and pages, marked ✅ inline.
The 7 still marked ⚠️ remain written from memory. See `VERIFICATION-LEDGER.md` §6
for the full record, and **§ Two questions — answered** below, which settles both
open reading questions.

### Infrastructure — cannot publish without these

| citation | role |
|---|---|
| **✅ Rössler, Cozzolino, Verdoliva, Riess, Thies, Nießner. "FaceForensics++: Learning to Detect Manipulated Facial Images." ICCV 2019.** | the dataset. Also the source of the `target_source` naming our component clustering exploits. Check their official split sizes and how sequence pairs were formed |
| **✅ Chollet. "Xception: Deep Learning with Depthwise Separable Convolutions." CVPR 2017, pp. 1800–1807.** | the backbone for both arms, and for F3-Net. Why the comparison is matched |
| **⚠️ Lin, Goyal, Girshick, He, Dollár. "Focal Loss for Dense Object Detection." ICCV 2017.** | our training objective, γ=2.0, α from train class balance |
| **✅ Loshchilov, Hutter. "Decoupled Weight Decay Regularization." ICLR 2019.** | AdamW. Also the citation that makes our `no_decay_param_groups` fix legible — decoupled decay was being applied to a scalar gate |
| **✅ Loshchilov, Hutter. "SGDR: Stochastic Gradient Descent with Warm Restarts." ICLR 2017.** | cosine schedule |

### Statistical methodology — contribution 2 has no grounding without these

| citation | role |
|---|---|
| **✅ Hurlbert. "Pseudoreplication and the Design of Ecological Field Experiments." Ecological Monographs 54(2), 187–211, 1984.** | **names the error** and gives it a forty-year pedigree. Framing improves: not "we invented something" but "a known error has a large measurable cost here". Ours is closest to his *simple pseudoreplication* |
| **Efron, Tibshirani. "An Introduction to the Bootstrap." 1993.** / **Davison, Hinkley. "Bootstrap Methods and their Application." 1997.** | the percentile bootstrap. Davison & Hinkley is the better cite for **coverage**, which limitation 4 concerns |
| ✅ **Field, Welsh. "Bootstrapping Clustered Data." JRSS-B 69(3), 369–390, 2007.** / ✅ **Cameron, Gelbach, Miller. "Bootstrap-Based Improvements for Inference with Clustered Errors." REStat 90(3), 414–427, 2008.** | cluster bootstrap. **Answered 2026-09-15:** Cameron et al. state that standard asymptotic tests "can over-reject with few (five to thirty) clusters". **We have 29 — inside their stated regime.** Limitation 4 must cite this rather than admit ignorance, and it is the motivation for A1 and A6 |
| ✅ **DeLong, DeLong, Clarke-Pearson. Biometrics 44(3), 837–845, 1988.** / ✅ **Hanley, McNeil. Radiology 143(1), 29–36, 1982.** | the standard correlated-AUC comparison. **Needed to answer "why not DeLong?"** — its variance estimator assumes independent observations, which is the assumption the paper is about |
| **✅ Lakens. "Equivalence Tests: A Practical Primer for t Tests, Correlations, and Meta-Analyses." SPPS 8(4), 355–362, 2017.** | why "not significant" ≠ "no effect", and why margins must be set externally. Ours is externally anchored to +0.014, which is stronger than the arbitrary margins he warns against |

### ML evaluation, variance and leakage — where contributions 2 and 3 live

**The most important gap.** If we do not engage this literature, a reviewer will
say "already known" — and for parts of contribution 3 they would be partly right.

| citation | role |
|---|---|
| ✅ **Bouthillier, Delaunay, Bronzi, et al. "Accounting for Variance in Machine Learning Benchmarks." MLSys 2021.** | **contribution 3 in a general-ML setting.** The single most important addition. Position carefully: *they* establish the principle; *we* show it reverses a published-effect comparison, crossed with test-content clustering their setting does not require. Check whether clustering is among their variance sources — if not, that is precisely our addition |
| ✅ **Kapoor, Narayanan. "Leakage and the reproducibility crisis in machine-learning-based science." Patterns 4(9), 100804, 2023.** | **names and taxonomises contribution 1's error** across many fields. Find which category crop-randomised splitting falls into and use their term. Compare their documented leakage-induced inflations against our ~18 points |
| ✅ **Henderson et al. "Deep Reinforcement Learning that Matters." AAAI 2018, pp. 3207–3214.** / ✅ **Melis, Dyer, Blunsom. "On the State of the Art of Evaluation in Neural Language Models." ICLR 2018.** | seed variance overwhelming architectural differences, established in other subfields. **Cite two, not four** — Melis is closest to us (claimed architectural gains dissolving under careful evaluation) |
| Reimers, Gurevych. EMNLP 2017. / Dodge et al. EMNLP 2019. | same point, NLP. Corroborative only |
| ✅ **Pineau et al. "Improving Reproducibility in Machine Learning Research." JMLR 22, 1–20, 2021.** / ⚠️ **Gundersen, Kjensmo. AAAI 2018.** | reproducibility standards. Lets us frame our provenance apparatus as **meeting a standard** rather than as idiosyncratic diligence |
| Bengio, Grandvalet. JMLR 5, 2004. | variance estimation with dependent resamples is genuinely hard. Cite only if pressed on the 59/41 allocation |

### Two questions — ANSWERED 2026-09-15

Both were settled by full-text extraction of the source PDFs. Quotes are verbatim.

#### 1. Does Bouthillier et al. include test-content clustering? **No.**

They assume i.i.d. data explicitly:

> "These pairs are i.i.d. and sampled from an unknown data distribution D"

Their enumerated variance sources are **data sampling (bootstrap), data
augmentation, model initialization, dropout, and data visit order**, plus
hyperparameter optimisation. Test-content clustering is not among them.

**But they anticipate it in one sentence and then set it aside:**

> "If errors are correlated, not i.i.d., the degrees of freedom are smaller and the
> distribution is wider."

They move on because their binomial model matched their empirical bootstrap on
i.i.d. image benchmarks. **This is the precise gap our contribution 3 occupies** —
they name the possibility; in face-forgery evaluation the correlation is
structural (3,000 crops → 150 videos → 29 components), and we measure what it
costs. Our novelty claim survives, and is now anchored to a sentence in the very
paper that would otherwise pre-empt it.

**A second, unexpected link.** They also state:

> "these different contributions to the variance are not independent, the total
> variance cannot be obtained by simply adding them up"

We measured exactly that: `nonadditivity_ratio = var(crossed) / (var(component) +
var(seed)) = 1.697`. They state the principle; we supply a number for it in this
domain. Cite both together.

#### 2. Which Kapoor & Narayanan category? **[L3.2], verbatim.**

> **"[L3.2] Nonindependence between training and test samples.** Nonindependence
> between training and test samples constitutes leakage, unless the scientific
> claim is about a distribution that has the same dependence structure."

Their own example is ours in another domain:

> "In the extreme (but unfortunately common) case, training and test samples come
> from the same people or units."

Crops from the same video are the same unit. **And their prescribed remedy is what
we implemented:**

> "Methods such as 'block cross-validation' can partition the dataset strategically
> so that the performance evaluation does not suffer from data leakage"

That is L2 video-disjoint and L3 component-disjoint splitting. Use their term
**[L3.2] nonindependence** — one word, no hyphen — rather than inventing one.
They also note the general problem "is a hard problem", which supports our framing
rather than undercutting it.

## One-paragraph positioning (for the intro)

> ⛔ **STALE — DO NOT USE. Flagged 2026-09-15.**
>
> The paragraph below is **old-framing** and violates `ESSENCE.md` §11 in three
> ways: it says "we find it does not", which is the performance claim we never
> make; it rests on cross-dataset and robustness results that are **superseded or
> DO-NOT-CITE**; and it treats FAD as the subject of the paper rather than the case
> study. Retained only as a record of what the positioning used to be.
>
> A replacement must lead with the evaluation-methodology claim (`ESSENCE.md` §0)
> and cite Kapoor & Narayanan [L3.2] and Bouthillier et al. for contributions 1–3.

> F3-Net (ECCV 2020) established that injecting an explicit frequency-domain branch
> into a spatial CNN improves face-forgery detection, with its largest reported
> gains on heavily compressed video. Five years on, the most recent major work in
> this space, FreqDebias (CVPR 2025), diagnoses a new failure mode — *spectral
> bias*, where unregularised frequency reliance itself restricts generalisation —
> and proposes consistency-driven training to correct it. We ask a narrower,
> practitioner-facing question: under a matched backbone, training budget, and
> increasing compression, does an explicit frequency branch (F3-Net's FAD, and our
> own learnable DCT front-end) earn its computational cost at all? Across in-domain,
> per-manipulation, cross-dataset, and compression-robustness evaluation, we find it
> does not, consistent with the spectral bias FreqDebias identifies and with the
> generalisation-favours-spatial pattern reported by CADDM (Dong et al., CVPR 2023),
> Gragnaniello et al. (ICME 2021), Corvi et al. (ICASSP 2023), and Ojha et al.
> (CVPR 2023) — three independent groups, three different mechanisms (implicit
> identity leakage, generator-specific spectral fingerprints, unregularised
> spectral bias), converging on the same direction.
