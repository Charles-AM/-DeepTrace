# Contribution 2 — evidence map

Companion to `docs/contribution-1-evidence.md`, same discipline: the verification
ledger is organised by **source**; this file is organised by **claim we make**.

**Rule: if a sentence about contribution 2 is not derivable from a row below, it
does not go in the document.**

---

## The claim

> Dependence-aware evaluation changes test-content uncertainty and the statistical
> verdicts drawn from it. Architecture-level conclusions additionally require
> propagating training-run variation, not only test-content variation.

Contribution 1 concerns **which quantity is estimated**. Contribution 2 concerns
**how uncertain that estimate is**. Keep the two apart — see §"Boundary" below.

---

## 1. The core result — one estimate, three uncertainty models

c40, seed 0, `results/analysis/cluster_boot/`.

| # | resampling unit | n units | 95% CI | excludes zero | status |
|---|---|---|---|---|---|
| 1.1 | **frame** | 3000 | [−0.0325, −0.0079] | **yes** | ✅ |
| 1.2 | **video file** | 150 | [−0.0606, +0.0167] | no | ✅ |
| 1.3 | **source-target component** | 29 | [−0.0534, +0.0096] | no | ✅ |

Point estimate is **−0.0196 in all three rows** — identical data, identical
estimate, different dependence assumptions.

> ✅ **Independently recomputed 2026-09-17** with a separate implementation of AUC
> and of the cluster bootstrap (ledger §17). Point estimates and unit counts exact;
> interval endpoints agreed to the third decimal; **all three verdicts identical**.
> The finding is a property of the data, not of our code.

⚠️ Write **"three uncertainty models, two conclusions"** — video and component
agree with each other. Never "three different conclusions".

---

## 2. Resolution — what the design can actually detect

`results/analysis/resolution/`

| # | claim | value | status |
|---|---|---|---|
| 2.1 | Component half-widths across five c40 runs | **0.037–0.062** | ✅ |
| 2.2 | As a multiple of the +0.014 reference | **2.6–4.5×** | ✅ |
| 2.3 | Per-run detail | 0.0370, 0.0442, 0.0624, 0.0489, 0.0452 | ✅ |
| 2.4 | Four of five intervals include +0.014 | seed 0 excludes it via its upper endpoint (diff −0.0303, CI [−0.0702, +0.0039]) | ✅ descriptive tally |
| 2.5 | Video-file unit, secondary | 0.042–0.067 (3.0–4.8×) | ✅ **do not splice with 2.1** — different resampling units |
| 2.6 | Half-width floor, matched spatial-family comparisons | 0.023–0.038 = **1.7–2.7×** the reference | ✅ |

⚠️ Half-width is defined **h = (U − L)/2**. Percentile intervals are **not centred**
on the point estimate — measured offsets reach 0.0028, 20% of the reference effect.
**Report endpoints, never `estimate ± h`.**

⚠️ Five intervals over five different splits are **descriptive**. "One of five" is
not an exclusion *rate*, and these are not a combined test.

---

## 3. The crossed interval — the primary result

`results/analysis/crossed/`, `results/canonical.json`, V8 fixed-split.

| # | variation propagated | 95% CI | half-width | contains +0.014 |
|---|---|---|---|---|
| 3.1 | test components only, runs fixed | [−0.0224, +0.0042] | 0.0133 | **no** |
| 3.2 | training runs only, content fixed | [−0.0257, +0.0041] | 0.0149 | **no** |
| 3.3 | **both, crossed** | **[−0.0358, +0.0180]** | **0.0269** | **yes** |

| # | claim | value | status |
|---|---|---|---|
| 3.4 | Point estimate | **−0.0092** | ✅ |
| 3.5 | Crossed is wider than either conditional | **1.8×** | ✅ |
| 3.6 | **Either conditional alone supports a stronger claim than the data permit** | both exclude +0.014; the crossed interval does not | ✅ **this is the contribution in one line** |
| 3.7 | Exceedance above the reference | **4.07%** of 50,000 replicates | ✅ |
| 3.8 | Monte Carlo band on the upper endpoint | [0.0176, 0.0184] — entirely above +0.014 | ✅ |
| 3.9 | Non-additivity ratio | **1.697** = var(crossed) / (var(comp) + var(seed)) | ✅ exploratory |

> ✅ **Reproduced exactly 2026-09-17** by rerunning `src/crossed_boot.py` with the
> documented command — interval, exceedance, MC band and non-additivity all match
> (ledger §18.1).

⚠️ **3.9 is exploratory.** It quantifies, in this dataset, the principle Bouthillier
et al. state generally: *"these different contributions to the variance are not
independent, the total variance cannot be obtained by simply adding them up."*

---

## 4. Training-run variation

`results/in_domain_c40_fixedsplit/`

| # | claim | value | status |
|---|---|---|---|
| 4.1 | Fixed-split training-run sd | **0.0145** | ✅ point estimate |
| 4.2 | Complete-pipeline sd (varying split) | **0.0188** | ✅ |
| 4.3 | Implied additional split-related sd | 0.0121 | ⚠️ **illustrative**, under an additivity assumption |
| 4.4 | Illustrative allocation | 59% / 41% | ⚠️ **not an established variance partition** |
| 4.5 | The two sds are **not distinguishable** | F = 1.70, df 4,4 | ✅ |
| 4.6 | Run-to-run range with the split frozen | 0.034 = **2.4× the reference** | ✅ |

⚠️ **Naming is fixed.** 0.0145 is **"fixed-split training-run variability"** — it
contains initialisation, data order, augmentation and nondeterministic runtime
operations, and possibly session-level effects. Never "optimisation variability".
Never a "bound" — it is a point estimate.

### 4.7 The repeatability audit

Seed 0 is nominally identical across two campaigns — same verified split, same
seed, same code, same hyperparameters, both fp32.

| | Xception | + FAD | difference |
|---|---|---|---|
| varying-split run | 0.81347 | 0.79391 | −0.0196 |
| V8 run | 0.79705 | 0.81104 | **+0.0140** |
| movement | −0.0164 | +0.0171 | **+0.0336** |

The paired difference moved **0.0336 = 2.4× the reference** and **flipped sign**.
The two arms moved in **opposite directions**, so it is not a constant session
offset.

⚠️ **n = 1.** Only seed 0 is a repeated configuration. This is a **reproducibility
audit**, not an estimate of session-level variance.
⚠️ **No causal attribution.** cuDNN kernel selection, hardware and library versions
are all candidates; none was tested. Offering an untested explanation would be a
claim we cannot support.

---

## 5. Robustness of the inference

| # | check | result | status |
|---|---|---|---|
| 5.1 | Class-stratified bootstrap | **0 of 10 verdict flips** | ✅ `analysis/sensitivity/` |
| 5.2 | Target-group vs component clustering | half-width differs **0.0002**; 0 verdict changes in 8 comparisons | ✅ prespecified at `0639ba0` |
| 5.3 | **Why 5.2 is null** | in this split the two partitions nearly coincide — **30 target groups, 29 components**, because only one pair (251, 375) has both members in test | ✅ ledger §15 |
| 5.4 | Aggregation moves the estimate | all 5 runs, **sign preserved in all 5** | ✅ recomputed §18.2 |
| 5.5 | Aggregation **space** (logit vs probability) | conclusion unchanged; quantities move — see §7 | ✅ ledger §19 |

⚠️ **5.2 must always be written with 5.3.** Alone, the null reads as a convenient
robustness result. With the explanation it is a statement about **this split**, not
about clustering choices in general.

---

## 6. Calibration — what the pipeline can and cannot resolve

`results/analysis/generality/`, 18 pairwise comparisons, c23, component unit.

| observed \|diff\| | excluded zero |
|---|---|
| > 0.10 | **9 of 9** |
| 0.03 – 0.10 | 1 of 1 |
| < 0.03 | **0 of 8** |

⚠️ **Not a power analysis.** Power is defined against a *prespecified* true effect;
these are sorted **after the fact** by observed difference, and an observed
difference far larger than its half-width excludes zero close to by construction.

❌ "selectively powered" ❌ "positive control"
✅ **"informative for large architectural differences, insufficient resolution for
FAD-sized ones"** ✅ **"large-separation calibration model"**

---

## 7. Compression and resolution

`results/analysis/resolution/` §2. Three seeds evaluated at both compressions on
**hash-verified identical test content** (complete manipulation–target–source
membership).

| seed | c23 half-width | c40 half-width | ratio |
|---|---|---|---|
| 0 | 0.0152 | 0.0370 | 2.43× |
| 1 | 0.0292 | 0.0442 | 1.52× |
| 2 | 0.0255 | 0.0624 | 2.45× |

**c40 intervals were 1.5–2.5× wider than c23 on identical test content.** The same
seed yields the same partition at both compressions because `make_splits` groups on
`videos-([0-9]+)`, which is compression-independent.

⚠️ Write **"heavy compression is associated with wider cluster-aware intervals under
matched test content"**. Three matched runs support an **association**, not "c40 has
lower achievable resolution" as a universal property.

---

## 8. The aggregation-space sensitivity (post-hoc)

V8 fixed-split, **paired** — one set of 50,000 component × training-run resamples
applied to both rules. Ledger §19.

| | mean-logit (canonical) | mean-probability |
|---|---|---|
| point estimate | −0.0092 | −0.0060 |
| 95% CI | [−0.0360, +0.0175] | [−0.0379, +0.0238] |
| half-width | 0.0268 | 0.0308 (**+15%**) |
| P(> +0.014) | 0.0405 | **0.0825** |
| contains 0 / +0.014 | yes / yes | yes / yes |
| **verdict** | neither demonstrated nor excluded | **unchanged** |

Paired per-replicate difference: mean **+0.00286**, 95% of replicates in
[−0.0072, +0.0126] — **not a constant offset**. Under the *same* resample the two
rules disagree about exceeding +0.014 in **4.48%** of cases.

⚠️ **Post-hoc.** The discrepancy was observed before the analysis was specified.
Never present it as prespecified.
⚠️ **This belongs to contribution 1**, not contribution 2 — it defines the video
*score*, not the uncertainty. Listed here only because it was found while
validating contribution 2's numbers.
✅ Framing: **"robust conclusion, aggregation-sensitive quantities."**

---

## 9. Validations performed (2026-09-17)

| # | what was validated | method | outcome |
|---|---|---|---|
| 9.1 | Three-units result | independent reimplementation of AUC + cluster bootstrap | ✅ estimates and unit counts exact; all verdicts identical (§17) |
| 9.2 | Canonical crossed result | reran `src/crossed_boot.py`, documented command | ✅ **exact** — interval, exceedance, MC band, non-additivity (§18.1) |
| 9.3 | Per-seed frame-pooled diffs | independent implementation | ✅ exact, all five (§18.2) |
| 9.4 | Per-seed video-aggregated diffs | independent implementation, mean-logit | ✅ exact, all five (§18.2) |
| 9.5 | Aggregation-space contrast | paired, shared resamples, 50,000 replicates | ✅ conclusion robust, quantities move (§19) |
| 9.6 | Canonical test suite | `pytest tests/test_canonical.py` | ✅ **18 passed** |

⚠️ **"Exact reproduction" applies only to 9.2 and 9.6** — same code, same seed.
An independent implementation agrees **within Monte Carlo error**; the stored
resampling indices from the frozen run were not retained. The aggregation step is
deterministic; **the bootstrap summary is not** unless indices or RNG state are fixed.

---

## 10. Literature anchors

| # | source | role |
|---|---|---|
| 10.1 | **Hurlbert**, Ecol. Monogr. 54(2), 187–211, 1984 | **pseudoreplication** — treating correlated observations as independent *inferential* units. **This is contribution 2's anchor**, not contribution 1's |
| 10.2 | **Field & Welsh**, JRSS-B 69(3), 369–390, 2007 | the cluster bootstrap |
| 10.3 | **Cameron, Gelbach & Miller**, REStat 90(3), 414–427, 2008 | over-rejection with **"few (five to thirty) clusters"** — **we have 29**. Limitation 4 cites this rather than admitting ignorance |
| 10.4 | **DeLong et al.**, Biometrics 44(3), 837–845, 1988 | answers *"why not DeLong?"* — its variance estimator assumes independent observations, the very assumption under test |
| 10.5 | **Davison & Hinkley**, 1997 | the better cite for **coverage**, which limitation 4 concerns |
| 10.6 | **Lakens**, SPPS 8(4), 355–362, 2017 | why "not significant" ≠ "no effect"; margins must be set externally. Ours is anchored to +0.014, stronger than the arbitrary margins he warns against |
| 10.7 | **Bouthillier et al.**, MLSys 2021 | training-run variance in general ML. They assume i.i.d. test data, note *"if errors are correlated, not i.i.d., the degrees of freedom are smaller and the distribution is wider"*, and set it aside. **That is the gap we occupy.** They also state variance contributions are **not additive** — which 3.9 quantifies |
| 10.8 | **Henderson et al.** AAAI 2018 / **Melis et al.** ICLR 2018 | seed variance overwhelming architectural differences, established in other subfields |
| 10.9 | **DeepfakeBench** | **the boundary**: they standardise *which* estimate is computed; we examine *how uncertain* it is (ledger §16) |

---

## Boundary — contribution 1 vs contribution 2

| choice | defines | contribution |
|---|---|---|
| frame-pooled vs video-aggregated | **which quantity is estimated** | **1** |
| mean-logit vs mean-probability | **how the video score is formed** | **1** |
| frame / video / component resampling | **how uncertainty is calculated** | **2** |

The aggregation operator defines the video **score**; component resampling defines
how **uncertainty** is computed. Conflating them would undo the separation that
ledger §10 established between the splitting error (Kapoor & Narayanan) and the
inference error (Hurlbert).

---

## Claims we deliberately do NOT make

| ❌ never write | why |
|---|---|
| "three different conclusions" | **two.** Video and component agree with each other |
| "optimisation variability" | **"fixed-split training-run variability"** — the measured quantity includes initialisation, data order, augmentation and runtime nondeterminism |
| "a bound on training-run variability" | it is a **point estimate**, n=5, not distinguishable from the complete-pipeline sd (F=1.70) |
| "we partitioned the variance" | **not an established variance partition** — the 59/41 split is illustrative, under an additivity assumption |
| "selectively powered" | **not a power analysis** — sorted after the fact by observed difference |
| "positive control" | **"large-separation calibration model"** |
| "c40 has lower achievable resolution" | three matched runs support an **association**, not a universal property |
| any cause for the repeatability movement | **n = 1, no causal attribution.** cuDNN, hardware, library versions all untested |
| "clustering choice does not matter" (from 5.2) | in **this split** the two partitions nearly coincide (30 vs 29). Always write 5.2 with 5.3 |
| "exact reproduction" for an independent implementation | ✅ **"agrees within Monte Carlo error"** — stored resampling indices were not retained |
| "two of five runs flipped their verdict" | a per-seed threshold crossing is **not a statistical verdict**. ✅ "moved the point estimate from above to below the +0.014 reference magnitude" |
| "no paper reports the score space" | ✅ **"the papers in our documented audit did not state the score space used for video aggregation"** |
| "estimate ± h" | percentile intervals are **not centred**; offsets reach 0.0028. **Report endpoints** |

---

## Open items — experiments and documentation owed

### Experiments

| # | what | cost | what it closes |
|---|---|---|---|
| **A1** | Bootstrap coverage simulation | ~1.5 h CPU, **local** | **Limitation 4.** Converts "coverage at 29 clusters unverified" into a measured number. Cameron et al. put us inside the documented over-rejection regime, so either outcome is reportable |
| **A4** | Per-manipulation cluster-aware intervals | ~20 min CPU | `permanip_l2` is **descriptive only — it has no intervals at all** |
| **A5** | Resolution curves, remaining seeds + c23 component unit | ~30 min CPU | currently c40 seed-0 only at the component unit |
| **A6** | BCa sensitivity | ~30 min CPU | **no BCa anywhere in the repo.** Percentile intervals can be biased at 29 clusters |
| **new** | c23 replication of the aggregation-space contrast | ~20 min CPU | tests whether §8 generalises past c40 |
| **C2** | Repeat audit, 2 in-session repeats | ~2.4 h **GPU** | takes 4.7 from **n=1** to characterisable; separates run-level from session-level |

### Documentation

| # | what | blocked on |
|---|---|---|
| D1 | Name the estimand **and the averaging space** in `manuscript_sentence`; add an 11th test asserting it | **Charlie's approval** — approved wording |
| D2 | Limitations sentence: our reference effect may be measured at a different granularity than our estimate (ledger §11) | — |
| D3 | Results directory + README for the §8 sensitivity analysis, marked post-hoc | — |
| D4 | Limitation 4 rewritten to cite Cameron et al. rather than admit ignorance | A1 |
