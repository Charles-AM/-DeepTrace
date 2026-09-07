# The essence — what this project is, in one place

The north-star document. If any decision, experiment, or paragraph doesn't serve
what's below, it's drift. Read this first; everything else is detail.

Last updated 2026-09-05 (rev. 3 — **framing FROZEN**).

---

## 0. 🔒 FROZEN — the headline

**No further framing critique until every queued experiment has run.** Three rounds
of external review have converged; the last two corrected wording, not substance.
Further rounds trade execution time for polish. Re-open only if an experimental
result contradicts something below.

### Title
> **Evaluation Units Matter: A Cluster-Aware Assessment of Frequency-Domain
> Deepfake Detection**

### Thesis
> Evaluation design changes both what deepfake detectors appear to achieve and how
> much **evidential weight** an architectural comparison can carry. Using F3-Net's
> FAD component as a controlled case study, we quantify the crop-randomised versus
> video-disjoint protocol gap, separate clustered test-content uncertainty, and
> compare **fixed-split training-run variability** with **complete-pipeline
> variability** on FaceForensics++.
>
> *(After V2 isolates the seen-video effect, the first clause becomes "we isolate
> seen-video leakage".)*
>
> ⚠️ Revised 2026-09-06 under the §0 clause permitting re-opening when a result
> contradicts the framing. V8 showed the promised separation of "optimisation
> variability" is not supported at n=5: the two variances are not distinguishable,
> and the implied component changes sign between attempts. "Optimisation
> variability" was also the wrong category name — the measured quantity includes
> initialisation, data order, augmentation and runtime nondeterminism.

### The line the paper is built around
> **The conclusion was stable; its warrant was not.**

### The fact that unifies all three contributions
> **3,000 test crops. 30 independent videos.** Apparent precision came from frame
> count; real resolution comes from independent clusters. Everything else follows
> from that gap.

### Three contributions
1. **Protocol effect** — the seen-video vs unseen-video performance gap across
   matched detectors (~18 AUC points in our subset).
2. **Inference effect** — separating crop, test-content, and optimisation
   randomness, and reporting what resolution the design actually achieves.
3. **FAD case study** — consistently small point estimates across compression
   levels and protocols; **resolution pending cluster-aware inference (V1)**.

Everything below is supporting detail for these four boxes.

---

## 1. The thesis (expanded)

> Evaluation design changes both what deepfake detectors appear to achieve and how
> much **evidential weight** an architectural comparison can carry. Using F3-Net's
> FAD component as a controlled case study, we quantify the crop-randomised versus
> video-disjoint protocol gap, separate clustered test-content uncertainty, and
> compare **fixed-split training-run variability** with **complete-pipeline
> variability** on FaceForensics++.
>
> *(After V2 isolates the seen-video effect, the first clause becomes "we isolate
> seen-video leakage".)*
>
> ⚠️ Revised 2026-09-06 under the §0 clause permitting re-opening when a result
> contradicts the framing. V8 showed the promised separation of "optimisation
> variability" is not supported at n=5: the two variances are not distinguishable,
> and the implied component changes sign between attempts. "Optimisation
> variability" was also the wrong category name — the measured quantity includes
> initialisation, data order, augmentation and runtime nondeterminism.

Result-independent by construction — it holds whether FAD helps, doesn't, or stays
inconclusive.

**The strongest line available to us:**

> *The conclusion was stable; its warrant was not.*

At L1 (crop-randomised) and L2 (video-disjoint) the FAD-vs-Xception result was null
both times. What changed was not the answer but what the answer is worth. Be precise
about why — warrant is **three** things, and only two are established:

| dimension | at L1 | status |
|---|---|---|
| **Construct validity** — are we measuring unseen-video generalisation? | No: performance on unseen *crops from seen videos* is a different estimand | ✅ established |
| **Statistical validity** — are independent units treated as independent? | No: thousands of crops from ~30 videos are pseudoreplicated | ✅ established |
| **Practical resolution** — can the design exclude an effect large enough to matter? | Possibly compressed by the ceiling | ⬜ **hypothesis — must be demonstrated**, not assumed from a high AUC |

⚠️ **Do not claim L1 "could not have detected a difference."** Near-ceiling AUC does
not prove low power; with paired predictions and large n, small differences are
sometimes resolvable. That claim requires an interval or a detectable-effect
calculation.

⚠️ **Do not yet claim L2 is informative either.** Until the cluster-aware interval is
computed, the honest statement is *"its warrant changed and remains under
evaluation."* Upgrade only if the cluster-aware upper bound excludes the target
effect.

## 2. Title options

| | title | risk |
|---|---|---|
| **1 (recommended)** | *Evaluation Units Matter: A Cluster-Aware Assessment of Frequency-Domain Deepfake Detection* | low — states what's distinctive without implying we invented grouped splitting |
| 2 | *Protocol Before Architecture: Evaluation Design and the Assessment of Frequency Features* | fine **if** "before" means evaluation validity logically precedes architectural comparison — not that protocol reverses the conclusion |
| 3 | *What Does Frequency Add?* | weaker — the frequency null is semi-expected |
| ~~4~~ | ~~*Detectable but Not Complementary*~~ | **rejected** — commits to a result our intervals cannot support |

## 3. Motivation

Deepfake detection carries real stakes (video-call fraud, non-consensual imagery,
journalistic verification, video-KYC). Practitioners face a recurring architectural
choice: **add an explicit frequency branch, or spend that budget elsewhere?** The
literature has said yes since 2020, and the pattern is still active (Frequency
Masking ICASSP'24, FreqMamba, FreqDebias CVPR'25).

Answering that requires an evaluation design capable of resolving it. A convenient
but invalid crop-randomised protocol — **the one our own initial pipeline used** —
inflates FF++ AUC by ~18 points and measures the wrong estimand entirely.

*(Do not claim this protocol is widespread without a documented code/literature
audit. We know it is easy to fall into because we fell into it.)*

## 4. Application

| audience | what changes |
|---|---|
| Practitioners | Evidence on whether a frequency branch earns its cost: +31% FLOPs / +44% latency for a separate branch, ~+3% for FAD |
| Benchmarkers | Crop-randomised splits inflate ~18 points; grouped splits and cluster-aware intervals are necessary |
| Researchers running ablations | **How much resolution an FF++ ablation actually has** — plausibly less than the effects routinely claimed from it |

**Never claim:** knowledge of proprietary deployment stacks, or that published FF++
results are inflated (reputable work uses the official video-level splits).

## 5. Prior work — what they say vs. what we say

### Base — the method we implement and test
**Qian et al., "Thinking in Frequency" (F3-Net), ECCV 2020.** *They say:*
frequency-aware decomposition (FAD) + local frequency statistics (LFS) improve
face-forgery detection, most on low-quality media.
*We say:* under matched backbone, data and budget we assess whether **FAD alone**
delivers an advantage compatible with its reported magnitude.
⚠️ **We implement FAD only** — not LFS, not MixBlock. Configs are `Xception` and
`Xception + FAD`, **never "F3-Net"**. Avoid "replication failure": our subset,
resolution, epochs and implementation differ. The question is *compatibility with
the reported effect size*, not reproduction of the system.

### Recent reference point
**Kashiani et al., "FreqDebias," CVPR 2025.** *They say:* frequency reliance can
create spectral bias harming generalisation; augmentation + consistency
regularisation corrects it.
*We say:* a complementary perspective — frequency reliance can create
manipulation-specific shortcuts, so *explicitly adding frequency access does not
guarantee generalisable information*. **Not** "current SOTA" (that depends on
dataset/protocol/backbone/metric), and their result does **not** prove FAD is
unhelpful.

### Support
**Dong et al., "Implicit Identity Leakage" (CADDM), CVPR 2023.** *They say:*
detectors latch onto identity as a shortcut, harming generalisation.
*We say:* CADDM **motivates our identity-disjoint extension**. Video grouping
prevents frame overlap but may leave source-side overlap. We have quantified
**video-level** leakage; identity-level effects remain under evaluation.
⚠️ Our 18-point figure is same-video leakage, **not** identity leakage.

**Shiohara & Yamasaki, "SBI," CVPR 2022.** *They say / we say:* SBI demonstrates
that carefully synthesised spatial blending artifacts support strong cross-dataset
generalisation **without an explicit frequency branch**. (Do not inflate this into
"the field's strongest generalisation is spatial" — that's broader than SBI
establishes and reads as selective citation.)

Antecedents: Frank et al. ICML 2020; Zhang et al. WIFS 2019; Durall et al. CVPR
2020. Full list: `docs/related-work.md`.

## 6. Contributions (three, in order)

1. **Protocol effect** — quantify the seen-video vs unseen-video performance gap
   across matched detectors.
2. **Inference effect** — separate crop, test-content, and training randomness via
   paired cluster-aware evaluation; report what resolution the design actually has.
3. **FAD case study** — consistently small point estimates across compression
   levels and protocols, cost-accounted; whether they exceed prespecified
   thresholds is **pending cluster-aware inference (V1)**.

Complementarity (late fusion, α-sweep) is **explanatory evidence**, not a fourth
contribution.

## 7. Effect thresholds — anchored externally

Set from the literature, never from our own pilots.

| threshold | value | basis |
|---|---|---|
| Published FAD-specific gain | **+0.014 AUC** | ✅ **VERIFIED** — arXiv 2007.09355v2, Fig. 7(a) p.12 / Table 3 p.14: Xception 0.893 → Xception+FAD (learnable) 0.907 on FF++ LQ. Both Acc and AUC are reported; this is the AUC column. |
| Full-system gain (**do not use**) | +0.040 | full F3-Net (0.933); we implement neither LFS nor MixBlock |
| Practical threshold | 0.010, with sensitivity at 0.005 / 0.010 / 0.020 | a **stakeholder judgement**, not a principled constant — present it as such |

**Sobering implication.** Our seed-level CI half-width was ±0.043; a cluster-aware
interval over ~30 test videos is unlikely to be much tighter. That is **wider than
the +0.014 effect we are trying to adjudicate** — so the study as currently scoped
may not be able to answer its own question. This is not a failure; it *is*
contribution 2, and it makes scaling independent test videos **necessary rather
than optional**.

## 8. Status of findings

Updated 2026-09-06 after Tier 0 and V8.

| finding | status | evidence |
|---|---|---|
| Crop-randomised splits inflate FF++ AUC 17.6–19.2 pts across 3 architectures | **solid**, both compressions | `results/in_domain_c40_vid/`, `_c23_vid/` |
| Compression enlarges the protocol gap by 9.5–10.4 pts | **solid** — difference-in-differences | `results/in_domain_c23_vid/` §1 |
| One estimate, three uncertainty models, **two** conclusions (frame excludes zero; both clustered units include it) | **solid** | `results/analysis/cluster_boot/` |
| Component half-widths 0.037–0.062 = **2.6–4.5× the +0.014 reference effect**, five runs | **solid** | `results/analysis/resolution/` §1 |
| 4 of 5 intervals include +0.014; one excludes via its upper endpoint | **solid**, descriptive tally | `results/analysis/thresholds/` |
| c40 intervals 1.5–2.5× wider than c23 on **hash-verified identical test content** | **moderate** — 3 matched runs | `results/analysis/resolution/` §2 |
| Video-level aggregation moves the estimate in all 5 seeds, preserving sign | **solid** | `results/analysis/aggregation/` |
| Conclusions unchanged under a class-stratified bootstrap (0/10 verdict flips) | **solid** | `results/analysis/sensitivity/` |
| Fixed-split training-run sd 0.0145 vs complete-pipeline 0.0188 | **point estimate only** — not distinguishable (F=1.70, df 4,4); component changes sign between attempts | `results/in_domain_c40_fixedsplit/` §1 |
| A nominally identical seed-0 configuration did not repeat across sessions (Δ moved 0.0336 = 2.4× the reference effect) | **audit, n=1** — no causal attribution | `results/in_domain_c40_fixedsplit/` §2 |
| frequency_only strongest on Deepfakes, weakest on NeuralTextures (0.246 spread) | **descriptive** → supplementary; **replicates at L2** | `results/analysis/permanip_l2/` |
| Separate frequency branch +31% FLOPs / +44% latency; FAD ≈ +3% | **solid** | `results/analysis/efficiency/` |
| Fusion gate never leaves 0.5; weight-decay confound ruled out | **moderate** — needs α-sweep | `results/analysis/fusion_alpha.csv` |
| Real-vs-fake DCT gap small (d≈0.15), survives JPEG-q30 | **descriptive** — frame-pseudoreplicated; not affected by the L1 checkpoints | `results/analysis/spectra/` |
| ~~Late fusion adds nothing over spatial alone~~ | ⛔ **computed on L1 checkpoints — DO NOT CITE**; redo at c23 (G7) | `results/analysis/late_fusion/` |
| ~~Frequency branch representationally distinct~~ | ⛔ **L1 checkpoints — DO NOT CITE**; redo (G7) | `results/analysis/cka/` |

Scale: **81 training runs** (plus 10 discarded), 10 configurations, 66,000
committed predictions, ~45 GPU-hours (`docs/EXPERIMENTS.md`).

## 9. Validation required

**Essential**

| # | test | why |
|---|---|---|
| V1 | Paired bootstrap comparing **naive frame resampling → video-cluster → identity-component** resampling, plus hierarchical seed variability | Not three equally legitimate units: one naive analysis vs progressively defensible ones |
| V2 | Seen-video vs unseen-video matched test sets, one fixed trained model | Estimates the seen/unseen **generalisation gap** with the model held constant. Does *not* perfectly isolate leakage (different video content); match manipulation, compression, frame count, duration, sampling position, and repeat the matching |
| V3 | Identity-disjoint splits — **define identity explicitly**: sequence ID, source-target components, or face-embedding clusters. Retain a manipulated video only when target and source identities share a partition | FF++ sequence IDs are not verified human-identity labels |
| V4 | Video-level AUC **and** frame AUC on the same held-out videos | Separates aggregation effect from split effect |
| V5 | c23 and c40 under identical grouped splits and seeds | Enables the **interaction interval**, not two descriptive numbers |
| V6 | Thresholds above + sensitivity analysis | "Not significant" ≠ "no effect". Disclose that the practical margin was chosen after exploratory runs — **do not call it preregistration** |
| V7 | CI width vs independent-video count | Establishes how many videos an ablation needs to support a +0.014 claim — contribution 2's core figure |

**Recommended:** α-sweep 0→1 (stronger than "α didn't move"); spectral survival under
real H.264 c40; grouped learning curve (quantifies how performance and uncertainty
improve with independent-video count — it does *not* by itself separate leakage from
sample-size effects); prespecified early-stopping rule (epoch-3 peaks do **not**
license post-hoc budget cuts).

**Out of scope:** full FF++, multiple external datasets, many architectures, full
F3-Net with LFS+MixBlock, 10+ seeds.

## 10. Interpretation, precommitted

The compression statistic is the **interaction**, with its own interval:

$$(\mathrm{FAD}-\mathrm{Xception})_{c40} - (\mathrm{FAD}-\mathrm{Xception})_{c23}$$

Never infer an interaction because one level is significant and the other isn't.

| outcome | reading |
|---|---|
| Both intervals exclude the practical margin | No practically meaningful benefit at either compression, under tested conditions |
| Both centred near zero but wide | Inconclusive — report the resolution achieved |
| c23 positive, c40 near zero, **interaction excludes zero** | FAD benefit *decreases* under heavier compression — contrary to the expected direction |
| c23 positive, interaction includes zero | Insufficient evidence of compression-specific behaviour |
| Both exceed the practical margin | FAD earns its cost under both compressions |
| Frequency-only loses more L1→L2 than spatial | Possible greater sensitivity to seen-video leakage — requires a protocol × representation interaction to claim |

## 11. Claim discipline

| don't say | say |
|---|---|
| "FAD provides no benefit" | "No detectable benefit; our interval excludes effects larger than X" |
| "five independent measurements" | "five **related** estimates across compression levels, protocols and seed counts" — c40 n=3 is nested in n=5, frame/video share videos, c23/c40 are the same source content |
| "every estimate is null" | "no consistent directional advantage across the five conditions" — a null is a test result, not an observation |
| "the effect is unresolvable" | "resolution pending cluster-aware inference" — seed-level intervals measure optimisation variability only |
| "leakage-free evaluation" | "video-disjoint evaluation" |
| "the comparison is unanswerable" | "ceiling compression may limit resolution" (then demonstrate it) |
| "protocol changes the conclusion" | "protocol changes performance and warrant; the conclusion was stable" |
| "F3-Net" (our config) | "Xception + FAD" |
| "replication failure" | "not compatible with the reported FAD-ablation magnitude under our conditions" |
| "matched inference cost" | "cost-accounted" |
| "we quantify identity leakage" | "we quantify video-level leakage; identity effects under evaluation" |
| "leakage inflates AUC by 18 points" | "the **protocol gap** is ~18 points" — L1 vs L2 confounds leakage with partition difficulty until V2 isolates it |
| "the gap is 2.0–2.3× larger at c40" | "compression enlarged the gap by **9.5–10.4 AUC points**" — the difference-in-differences is tighter (9% spread) than the ratio (15%) |
| `f3net` in any table | **"Xception + FAD"** — use `config.display_name()` |

## 12. Target

4–5 page IEEE workshop / short paper. Structure: protocol result as motivation,
cluster-aware paired FAD result as the central test, **one** complementarity figure.
Main paper carries: protocol shift, cluster-aware uncertainty comparison, Xception ±
FAD at c23/c40, video-level metric, thresholds, efficiency, late fusion + α-sweep.
CKA, per-manipulation, gate training history and spectral inference → supplementary.
