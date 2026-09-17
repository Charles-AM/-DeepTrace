# Contribution 1 — evidence map

The verification ledger is organised by **source**. This file is organised by
**claim we make**, so that every sentence written about contribution 1 in the
proposal, report or paper can be traced to something checkable.

Rule: if a sentence about contribution 1 is not derivable from a row below, it
does not go in the document.

---

## The claim

> Protocol and estimand choices change apparent performance and the apparent size
> of architectural contrasts.

Note what this does **not** say: it makes no assertion about what other
researchers do. It states that the choice is consequential and measures by how
much. See "Claims we deliberately do not make" below.

---

## 1. Claims from our own experiments

| # | claim | value | evidence | status |
|---|---|---|---|---|
| 1.1 | Crop-randomised (L1) reports more than video-disjoint (L2) at c40 | +0.1921 / +0.1764 / +0.1817 (~18 pts) | `results/in_domain_c40_vid/README.md` §1 | ✅ |
| 1.2 | It holds across three architectures | baseline_spatial, xception, xception+FAD | same | ✅ |
| 1.3 | Compared on identical crops | same crop set as `results/in_domain_c40/` | same, header | ✅ |
| 1.4 | The L2 split was verified before training | *"300 video groups, 0 spanning more than one split"* | run log, quoted in README | ✅ |
| 1.5 | The gap is smaller at c23 | +0.0955 / +0.0814 / +0.0780 (~8 pts) | `results/in_domain_c23_vid/README.md` §1 | ✅ |
| 1.6 | Heavy compression enlarges the gap | **+0.0966 / +0.0950 / +0.1037** (difference-in-differences) | same | ✅ |
| 1.7 | Report the difference-in-differences, not the ratio | absolute spread 0.0087 vs ratio spread 0.32 | same | ✅ decided |
| 1.8 | `frequency_only` shows almost no protocol gap | **+0.0075**, an order of magnitude smaller | `results/in_domain_c23_vid/README.md` §2 | ✅ contrary to prediction |
| 1.9 | L1 compresses architectural differences | 0.6 pts apart at L1 vs 2.1 pts at L2 (**3.7×**) | derived from 1.1's absolutes | ✅ arithmetic |
| 1.10 | Unit structure of the subset | 3,000 crops → 150 videos → 30 target groups → **29 components** | `results/analysis/clusters/`, `component_stats.csv` | ✅ script-generated |

## 2. Claims about the reference effect

| # | claim | evidence | status |
|---|---|---|---|
| 2.1 | F3-Net's FAD ablation is **+0.014** (0.893 → 0.907) | Fig. 7(a) p.12 row 2 − row 1; Table 3 p.14 | ✅ |
| 2.2 | It is AUC, not accuracy | both reported side by side | ✅ |
| 2.3 | It is the **learnable** FAD variant | Table 3: fixed filters reach only 0.901 | ✅ |
| 2.4 | It is measured on the **LQ** task | Fig. 7(a) caption *"on the low quality task(LQ)"*; Table 3 caption | ✅ |
| 2.5 | **LQ = c40** | FF++ §3 p.5: HQ = quantization 23, *"Low quality videos (LQ) are produced using a quantization of 40"* | ✅ definitional |
| 2.6 | So their +0.014 and our primary result share one compression setting | 2.4 + 2.5 | ✅ |
| 2.9 | ⚠️ They may **not** share an estimand | F3-Net p.10: *"we also average the AUC scores of each frame in a video"* — reads as video-aggregated; ours is frame-pooled | ⚠️ ledger §11 — **qualify the 13× comparison** |
| 2.7 | The full-system +0.040 must **not** be used as our reference | we implement neither LFS nor MixBlock | ✅ decided, `ESSENCE.md` §7 |
| 2.8 | Using +0.040 would have given us a **stronger** claim | our CI upper endpoint is 0.0180 < 0.040, so it would be excluded | ✅ shows the threshold was not outcome-shopped |

## 3. Claims about what the literature reports

| # | claim | evidence | status |
|---|---|---|---|
| 3.1 | FF++ ships **official video-level splits**, 720/140/140 | FF++ dataset repository | ✅ |
| 3.2 | F3-Net reports **no measure of variability anywhere** | full-text search, ledger §4 | ✅ |
| 3.3 | DeepfakeBench reports **no measure of variability anywhere** | full-text search, ledger §4 | ✅ |
| 3.4 | DeepfakeBench pools **frames** for its metric | *"Our benchmark currently adopts the frame level evaluation..."* | ✅ verbatim |
| 3.5 | DeepfakeBench identifies the frame-vs-video **inconsistency** as causing unfair comparisons, and adopts frame-level pooling as its **solution** | same passage | ✅ — do **not** write that they call pooling itself a fairness error; that inverts them |
| 3.6 | The two published Xception figures differ by 0.067 | F3-Net 0.893 vs DeepfakeBench 0.8261 | ✅ **4.8× the effect under debate** |

## 4. Literature anchors for the error itself

| # | anchor | exact term | where |
|---|---|---|---|
| 4.1 | Kapoor & Narayanan, *Patterns* 4(9), 100804, 2023 | **[L3.2] Nonindependence between training and test samples** | free PDF, search "[L3.2]" |
| 4.2 | Their example matches ours | *"training and test samples come from the same people or units"* | same section |
| 4.3 | Their prescribed remedy is what we implemented | *"block cross-validation"*; also *"The train-test split should account for the dependencies in the data"* | same section |
| 4.7 | **Leakage is a property of the claim, not the split** | *"...constitutes leakage, **unless the scientific claim is about a distribution that has the same dependence structure**"* | ✅ **anchors our refusal to call crop-randomised splitting invalid** — ledger §13.1 |
| 4.8 | A published structural analogue in another field | histopathology: same **patient** in train and test, claim about **new patients** → *"a mismatch between the test set distribution and the scientific claim"* | ✅ use as the introductory analogy — ledger §13.2 |
| 4.4 | Scale of the problem across science | **leakage of all eight types** affects 294 papers across 17 fields | abstract — ⚠️ **not** 294 cases of [L3.2]; write "their broader review documented leakage across 294 papers in 17 fields" |
| 4.6 | **Precedent: correcting leakage dissolved an apparent advantage** | their civil-war prediction reproducibility study — *"When the errors are corrected, complex ML models do not perform substantively better than decades-old LR"* | ✅ **the closest published precedent for our kind of finding** |
| 4.5 | Hurlbert, *Ecological Monographs* 54(2), 187–211, 1984 | **pseudoreplication** | the founding reference |

---

## Claims we deliberately do NOT make

| ❌ never write | why |
|---|---|
| "recognising a face it has already seen" | **The shortcut's mechanism was never isolated.** Write "exploiting same-video content or acquisition cues" |
| "identical data / byte-identical images" (of the L1↔L2 comparison) | The crop **corpus** is the same; train/test **membership** is not, and cannot be. Say **only the partitioning rule changed** |
| "a property of the evaluation" (unqualified) | Say **"consistent in direction and approximate magnitude across the three tested architectures"** — n=3 |
| "FAD is worth +0.014" | **F3-Net reported a +0.014 AUC ablation gain under its setting.** Attribute it |
| "13× larger, therefore the FAD claim fails" | Two objections, both fatal to the strong reading. They answer different questions (absolute protocol shift vs matched architectural contrast), **and they may not share an estimand** — ours is frame-pooled, F3-Net's reads as video-aggregated (ledger §11). Use as **scale context across two evaluation designs**, never as disproof |
| "made the architectural question answerable at all" | Too categorical. **"Restored headroom and improved practical resolution"** |
| "the field has numbers but not error bars" | **Two papers audited.** Say "these two influential examples report point estimates without uncertainty" |
| "the standard setup falls short" | FF++ supplies fixed video-level splits. There is no deficient "standard setup" to indict |
| "crop-randomised splitting is invalid" | It is a valid estimator of a **different estimand** — performance on unseen frames of *known* videos. Say it **answers a different question** |
| "The field commonly splits by crop" | **No evidence.** FF++ ships video-level splits; a paper following the official protocol is not doing this |
| "Published numbers are inflated by 18 points" | Our L2 absolutes (~0.79–0.82) sit below F3-Net's published range for reasons that are **ours**: scoped subset, 128px inputs, 15 epochs, frame-pooled metric |
| "Leakage" (unqualified, for the L1→L2 gap) | The two protocols also induce different partitions. **"Protocol gap"** until V2 separates them |
| "Compression makes the model rely on memorised identity" | **Plausible mechanism, untested.** Label it as interpretation |
| "Frequency methods claim their largest gains under compression" | Only F3-Net checked. **Narrow to F3-Net** |
| "We found an error other researchers made" | We measured what a choice costs. That needs no claim about anyone's practice |

---

## The one open item

**B2 (V2), ~80 minutes GPU.** The protocol gap confounds two things: crops from one
video spanning the split (leakage), and the two protocols inducing different
partitions (difficulty). V2 holds test content fixed and varies only seen/unseen
status, isolating the leakage component.

Until it runs, "protocol gap" is the honest word, and it appears as a hedge in five
places in `ESSENCE.md`. This is the single highest-value GPU experiment remaining.
