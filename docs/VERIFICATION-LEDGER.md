# Verification ledger

One place to check what is verified, what is assumed, and what is still owed —
for **our own results** and for **every claim we make about other people's papers**.

Status key: ✅ verified from source · ⚠️ ambiguous, recorded as such · ❌ not yet
verified · 🔒 frozen

Last updated 2026-09-11.

---

## 1. Our experiments

| claim | evidence | status |
|---|---|---|
| Primary result −0.0092, 95% CI [−0.0358, +0.0180] | `results/canonical.json`, generated from `crossed_video_b50000_s0.csv`, hash-checked by `tests/test_canonical.py` | 🔒 |
| 50,000 replicates, reproducible on CPU | regenerated from committed dumps, numbers identical | ✅ |
| Protocol gap ~8.5 pts c23, ~18 pts c40 | `results/in_domain_c23_vid/`, `_c40_vid/` | 🔒 |
| Same estimate −0.0196, three units, two conclusions | `results/analysis/cluster_boot/` | 🔒 |
| Conditionals exclude +0.014, crossed does not | `crossed_video_b50000_s0.csv` | 🔒 |
| 9/9 above 0.10 exclude zero, 0/8 below 0.03 | `results/analysis/generality/` | 🔒 |
| c40 intervals 1.5–2.5× wider on identical membership | `results/analysis/resolution/` §2, membership SHA-256 verified | 🔒 |
| Run-to-run range 0.034 with split fixed | `results/in_domain_c40_fixedsplit/` | 🔒 |
| Repeatability audit: 0.0336 movement | same, **n = 1**, no cause isolated | ⚠️ |
| Variance allocation 59/41 | **not statistically distinguishable** (F = 1.70, df 4,4); sign flips between attempts | ⚠️ |
| Raw inputs committed | 22 + 10 prediction dumps, `SHA256SUMS.json` | ✅ |
| Test suite | 253 passed, 2 skipped (hardware-gated), under pytest | ✅ |
| Prespecifications precede their analyses | `86e6c98`, `0639ba0` — git history is the evidence | ✅ |

## 2. F3-Net (Qian et al., ECCV 2020) — arXiv:2007.09355v2

| claim | evidence | status |
|---|---|---|
| FAD ablation +0.014 AUC (0.893 → 0.907) | Fig. 7(a) p.12, Table 3 p.14; `docs/f3net-ablation-verified.md` | ✅ |
| The figure is **AUC**, not accuracy | both reported side by side; AUC column read | ✅ |
| +0.014 is the **learnable** variant (`f_base + f_w`) | Table 3: fixed filters reach only 0.901 | ✅ |
| Full system is +0.040 — **do not use** | Fig. 7(a) row 5 | ✅ |
| Their FAD matches ours structurally | three bands, learnable additive term, inverse-DCT per band | ✅ |
| In our documented full-text audit: **no confidence intervals, standard errors, standard deviations or repeated-run variability** accompany the reported detector comparisons | full-text search, see §4 | ✅ — scope to the audit; avoid the unqualified "anywhere" |
| **"LQ" = c40** | **CLOSED 2026-09-17.** FF++ (arXiv:1901.08971 §3, *Postprocessing – Video Quality*) defines the mapping itself: *"To generate high quality videos, we use a light compression denoted by HQ (constant rate quantization parameter equal to 23) ... Low quality videos (LQ) are produced using a quantization of 40."* F3-Net cites FF++ [50] and defines its labels identically (*"LQ indicates low quality (heavy compression), HQ indicates high quality (light compression) and RAW indicates raw videos without compression"*). HQ = quantization 23 = c23; LQ = quantization 40 = c40 | ✅ **verified** — definitional chain, no longer an inference. F3-Net never writes "c40", but uses FF++'s own labels in FF++'s own sense |
| **The +0.014 is measured on the LQ (heavy compression) task** | two independent captions — Fig. 7(a) p.12 *"Ablation study of the proposed F3-Net on the low quality task(LQ)"*; Table 3 p.14 *"...on FAD in FF++ low quality (LQ)"* | ✅ verified 2026-09-17 |
| **F3-Net is explicitly motivated by heavy compression** | abstract: *"especially wins a big lead upon low-quality media"*; intro: *"if the visual quality ... is tremendously degraded, such as compressed by JPEG or H.264 ... the forgery artifacts ... cannot be captured in RGB domain any more"*; contributions: *"significantly improves the performance over low-quality forgery media"* | ✅ verified 2026-09-17 |
| F3-Net's gain over Xception shrinks as quality rises: LQ +0.040, HQ +0.018, RAW +0.006 | Table 1 p.9 — Xception 0.893/0.963/0.992, F3-Net(Xception) 0.933/0.981/0.998 | ✅ **confirmed visually 2026-09-17**; the earlier text reconstruction matched exactly |
| **The full-system gain is backbone-dependent**: +0.040 on Xception vs **+0.022** on Slowfast at LQ | Table 1 p.9 — Slowfast 0.936 → F3-Net(Slowfast) 0.958; same pattern at HQ (+0.018 vs +0.011) and RAW (+0.006 vs +0.005) | ✅ their numbers; **our inference** from them |
| Table 1 reports **no AUC at all** for five of the twelve methods (Steg.Features, LD-CNN, Constrained Conv, CustomPooling CNN, MesoNet) and **no Acc** for Face X-ray | Table 1 p.9, dashes in the respective columns | ✅ observed directly |
| "**Frequency methods** claim their largest gains under compression" (plural, as a class) | **RESOLVED 2026-09-18.** FreqDebias (CVPR 2025) examined: it trains on FF++ **HQ** and frames its contribution as *generalisation*, not compression. It does **not** make the claim | ❌ **confirmed do-not-assert — the claim is F3-Net-specific** (`docs/split-policy-audit.md`) |

## 3. DeepfakeBench (Yan et al., NeurIPS 2023) — arXiv:2307.01426

| claim | evidence | status |
|---|---|---|
| Standardises data management, implementation framework, evaluation metrics/protocols | abstract, quoted | ✅ |
| Adopts **frame-level** evaluation | paper body: *"Our benchmark currently adopts the frame level evaluation to build a fair basis for comparison among detectors"* | ✅ |
| Identifies frame-vs-video **inconsistency** as causing "unfair comparisons" | *"there is an inconsistency in the usage of these evaluation metrics in the community, some are at the frame level, while others are at the video level, leading to unfair comparisons"* | ✅ |
| ⚠️ They do **NOT** call frame pooling itself a fairness error | pooling is their **solution** to the inconsistency, not their diagnosis. Never write "DeepfakeBench acknowledges frame pooling as a fairness problem" — it inverts them | ✅ precision fix 2026-09-17 |
| Correct PubMed record for Kapoor & Narayanan | **PMID 37720327**. (PMID 36913544 is a cobalt molybdenum sulfide catalysis paper — a miscitation seen in review feedback) | ✅ resolved via NCBI E-utilities |
| Four metrics: ACC, AUC, AP, EER — none a measure of variability | paper body, enumerated | ✅ |
| F3Net 0.8271 vs Xception 0.8261 at FF-c40 (+0.0010) | main results table, text-extracted | ✅ |
| In our documented full-text audit: **no confidence intervals, standard errors, standard deviations or repeated-run variability** accompany the reported detector comparisons | full-text search, see §4 | ✅ — scope to the audit; avoid the unqualified "anywhere" |
| Their F3Net scope | **paper** says two-branch FAD+LFS, re-implemented from the paper; **code** (`f3net_detector.py`) says FAD branch only, following a reference GitHub implementation | ⚠️ contradictory; our comparison holds under either reading (14× or 40×) |
| "Reproducibility" means consistent protocol/data | used of data pipelines and reported results, not of statistical stability | ✅ |

## 4. Full-text search for uncertainty reporting — 2026-09-11

Both PDFs extracted with `pdftotext -layout` and searched. **Extraction verified to
have captured the results tables** (0.8271/0.8261 and 0.907/0.893 all present as
text), so zero hits are genuine absences rather than extraction failures.

| term | F3-Net | DeepfakeBench |
|---|---|---|
| `±` | 0 | 0 |
| standard deviation | 0 | 0 |
| `std` | 1 — **in an email address** | 0 |
| variance | 4 — noise variances of a cited work, "shift-invariance", a reference title | 0 |
| confidence interval | 0 | 0 |
| error bar | 0 | 0 |
| random seed / seeds | 0 | 0 |
| repeated run(s) | 0 | 0 |
| significan* | 7 — **all colloquial** ("significantly outperforms") | 12 — **all colloquial** |

> **Neither paper reports any measure of variability, anywhere in its full text.
> Every use of "significant" is colloquial, not statistical.**

Reproduce:

```
pdftotext -layout 2007.09355v2.pdf f3net.txt
curl -sL -o dfb.pdf https://arxiv.org/pdf/2307.01426 && pdftotext -layout dfb.pdf dfb.txt
grep -ic -- "±" f3net.txt dfb.txt        # etc. for each term
```

⚠️ **Scope of this claim.** It is about **reporting**, not about internal practice.
We cannot know whether either group computed variability privately. Write *"neither
reports"*, never *"neither measured"*.

## 5. Still owed

| item | why it matters | cost |
|---|---|---|
| **7 citations in Groups A, G, H of `related-work.md`** | ~~20~~ → 7 after the 2026-09-15 pass (§6). Dataset paper and backbone now cleared; the rest still block submission | ~1 h |
| **Replacement intro positioning paragraph** | the existing one is quarantined as old-framing (§6) | short |
| DeepfakeBench F3Net numbers | read off a table image, then confirmed in extracted text — but worth one direct look at the PDF | minutes |
| Re-verify Groups B–F | last checked 2026-09-03/04 | an hour |

---

## 6. Citation verification pass — 2026-09-15

Checked against dblp, CVF open access, and publisher pages; the two papers with
open reading questions were extracted in full with `pdftotext` and quoted verbatim.

### Verified — 15

| citation | confirmed as |
|---|---|
| FaceForensics++ | Rössler, Cozzolino, Verdoliva, Riess, Thies, Nießner. ICCV 2019 |
| Xception | Chollet. CVPR 2017, pp. 1800–1807 |
| AdamW | Loshchilov, Hutter. "Decoupled Weight Decay Regularization." ICLR 2019 |
| SGDR | Loshchilov, Hutter. ICLR 2017 |
| Pseudoreplication | Hurlbert. Ecological Monographs **54(2), 187–211**, 1984 |
| Cluster bootstrap | Field, Welsh. JRSS-B **69(3), 369–390**, 2007 |
| Cluster bootstrap-t | Cameron, Gelbach, Miller. REStat **90(3), 414–427**, 2008 |
| Correlated AUC | DeLong, DeLong, Clarke-Pearson. Biometrics **44(3), 837–845**, 1988 |
| AUC meaning | Hanley, McNeil. Radiology **143(1), 29–36**, 1982 |
| Equivalence testing | Lakens. SPPS **8(4), 355–362**, 2017 |
| Benchmark variance | Bouthillier, Delaunay, Bronzi, et al. MLSys 2021 |
| Leakage taxonomy | Kapoor, Narayanan. Patterns **4(9), 100804**, 2023 |
| Seed variance (RL) | Henderson et al. AAAI 2018, **pp. 3207–3214** |
| Seed variance (LM) | Melis, Dyer, Blunsom. ICLR 2018 |
| Repro standards | Pineau et al. JMLR **22, 1–20**, 2021 |
| **Focal loss** | **Lin, Goyal, Girshick, He, Dollár. ICCV 2017, pp. 2999–3007** — ✅ verified 2026-09-21; our training objective (γ=2.0, α from train class balance) |

### Still unverified — 6

Efron & Tibshirani (1993) ·
Davison & Hinkley (1997) · Gundersen & Kjensmo (AAAI 2018) ·
Reimers & Gurevych (EMNLP 2017) · Dodge et al. (EMNLP 2019) ·
Bengio & Grandvalet (JMLR 5, 2004)

**Submission is still blocked until these are checked**, though the blocker is now
much smaller — and the dataset paper and backbone, the two that mattered most, are
cleared.

### Three findings that change what we can write

1. **Our 29 components sit inside a documented failure regime.** Cameron, Gelbach
   & Miller state that standard asymptotic tests "can over-reject with few (five to
   thirty) clusters". Limitation 4 stops being a vague admission of ignorance and
   becomes a citation — and A1 (coverage simulation) and A6 (BCa) stop being
   optional polish and become the natural response to a known problem.

2. **Bouthillier et al. name our gap and leave it open.** They assume i.i.d. test
   data, enumerate five variance sources, none of which is test-content clustering,
   and note in passing that "if errors are correlated, not i.i.d., the degrees of
   freedom are smaller and the distribution is wider" — then set it aside because
   their binomial model matched their bootstrap on i.i.d. benchmarks. Contribution 3
   is safe, and is now anchored to the paper that would otherwise pre-empt it.
   They also state variance contributions are **not additive**, which is the
   principle our measured `nonadditivity_ratio = 1.697` instantiates.

3. **Kapoor & Narayanan's term is [L3.2] "Nonindependence between training and test
   samples"**, and their prescribed remedy is **block cross-validation** — which is
   what L2 video-disjoint and L3 component-disjoint splitting are. Their example,
   "training and test samples come from the same people or units", is our case in
   another domain.

### One problem found

`related-work.md` still carried a **one-paragraph intro positioning from the
method-paper era** — it claimed "we find it does not", cited superseded and
DO-NOT-CITE results, and treated FAD as the subject rather than the case study.
Quarantined in place with a DO-NOT-USE banner rather than deleted. **A replacement
intro paragraph is now owed.**

---

## 7. FF++ official splits — verified 2026-09-17

**Verified.** FaceForensics++ provides official train/validation/test splits defined
at **video level**: **720 / 140 / 140**. Source: the FF++ dataset repository,
<https://github.com/ondyari/FaceForensics/tree/master/dataset> — *"We used 720
videos for train and 140 videos for validation as well as testing."*

This closes the open to-do on the FF++ entry in `related-work.md` ("check their
official split sizes").

### The correction it forces

A claim was made in conversation that crop-randomised splitting is what "a lot of
published work" does. **That claim is unsupported and is not made anywhere in this
repository.** It should never enter the proposal, report or paper.

What is actually established:

| claim | status |
|---|---|
| FF++ official splits are video-level, 720/140/140 | ✅ verified (above) |
| DeepfakeBench pools **frames** for the metric | ✅ verified, verbatim (§3) |
| **We** used crop-randomised splits before 2026-09-05 | ✅ our own logs, `results/in_domain_c40/` vs `_c40_vid/` |
| The field commonly splits by crop | ❌ **no evidence — do not assert** |

Because FF++ ships video-level splits, a paper following the official protocol is
**not** crop-randomised at the split level. Contribution 1 must therefore be stated
as it already is in `STATE-OF-PLAY.md` — *protocol and estimand choices change
apparent performance* — which asserts that the choice is consequential **without
claiming anyone made the wrong choice**. Our evidence is a controlled
demonstration built from the same crop corpus with partition membership differing
by the splitting rule; it needs no claim about others' practice. (Never "identical
crops" — train/test membership necessarily differs when the split rule is what changed.)

### Keep the two errors apart

- **Splitting** by crop → leakage (Kapoor & Narayanan [L3.2]). Demonstrated by us;
  not attributed to anyone else.
- **Pooling frames** for the metric → the estimand problem. **This one is
  documented in the field**, with a verified DeepfakeBench quote.

Conflating them overstates the first and wastes the second.

---

## 8. Ceiling effects — our own interpretation, flagged as such

F3-Net's advantage shrinks from +0.040 (LQ) to +0.006 (RAW). **Part of that is
almost certainly mechanical**: at RAW every method scores ~0.99, so there is
little room left to gain. A shrinking advantage as quality rises is therefore
**not by itself evidence that frequency contributes more under compression**.

We can say this from our own data rather than speculating about theirs. Under L1
our three architectures sat **0.6 AUC points** apart (0.9878 / 0.9933 / 0.9935);
under L2 the same three sat **2.1 points** apart (0.7957 / 0.8169 / 0.8117) — a
3.7× expansion. Ceilings compress architectural differences.

Status: **our interpretation, computed from verified numbers.** It is a legitimate
observation about measurement, not a claim about F3-Net's mechanism, and must not
be written as one.

---

## 9. Backbone dependence in F3-Net's own Table 1 — 2026-09-17

Confirmed from the table image. Adding the **same** full F3-Net system to two
different backbones produces materially different gains:

| backbone | LQ baseline | LQ with F3-Net | gain |
|---|---|---|---|
| Xception | 0.893 | 0.933 | **+0.040** |
| Slowfast | 0.936 | 0.958 | **+0.022** |

The same ordering holds at HQ (+0.018 vs +0.011) and RAW (+0.006 vs +0.005).

**Why this matters to us.** It is evidence *from the proposing paper's own results
table* that the benefit of the frequency system is **not a stable constant** — it
depends on what it is attached to, varying by ~1.8× across two backbones at LQ.
That is directly supportive of contribution 3: an architectural gain measured once,
on one backbone, without any uncertainty estimate, is a weaker piece of evidence
than its presentation implies.

⚠️ **Two honest caveats, state both.**

1. **Ceiling.** Slowfast starts higher (0.936 vs 0.893), so it has less headroom.
   Part of the smaller gain is mechanical, exactly as in §8.
2. **n = 1 each.** These are two single measurements with no repetition and no
   interval — the same limitation we identify everywhere else. We cannot say the
   difference between +0.040 and +0.022 is larger than the noise, because neither
   paper reports noise. **Use it to illustrate the problem, never as a measurement.**

### A second observation about the table

Five of the twelve methods report **no AUC at all**, and Face X-ray reports **no
Acc**. The comparison table therefore compares different methods on different
metrics, with gaps. Recorded as an observation about reporting practice; it is
**not** a criticism we need to make in the paper, and it is not evidence for any
of our three contributions.

---

## 10. Two errors, two anchors — keep them apart (2026-09-17)

From external review, and adopted. These had been conflated under "the same error
with two names". They are two distinct errors, and they map onto two contributions.

| error | what it is | anchor | our contribution |
|---|---|---|---|
| **Nonindependence between train and test** | crops from one video appear on both sides of the split | Kapoor & Narayanan, *Patterns* 4(9), 100804, 2023 (**PMID 37720327**) — [L3.2]; 294 papers, 17 fields | **1** |
| **Pseudoreplication** | treating correlated crops as independent *inferential* units when computing uncertainty | Hurlbert, *Ecological Monographs* 54(2), 187–211, 1984 | **2** |

The first is a **splitting** error; the second is an **inference** error. A study can
commit either without the other. Citing Hurlbert for the splitting problem, or
Kapoor & Narayanan for the uncertainty problem, misattributes both.

### Relationship to DeepfakeBench

DeepfakeBench standardises the **estimand** (frame-level) so that detectors are
compared on a common basis. That addresses inconsistency, not dependence. The
pseudoreplication problem survives their standardisation untouched. Our
contribution 2 is therefore **complementary to** their effort, not a correction of
it — write it that way.

---

## 11. Estimand mismatch — F3-Net's AUC granularity (2026-09-17)

Found while checking whether the "13×" comparison is sound.

**F3-Net, Evaluation Metrics, p.10, verbatim:**

> "(2) AUC. Following face X-ray [40], we use AUC score as another evaluation
> metric. **For single-frame methods, we also average the AUC scores of each frame
> in a video.**"

And immediately above, for accuracy:

> "for single-frame methods, we average the accuracy scores of each frame in a
> video."

### What this means

The sentence is **loose as written**: an AUC cannot be computed for a single frame,
since AUC requires both positive and negative examples. Read with the Acc sentence,
the intended procedure is most plausibly **average frame scores to a video-level
score, then compute AUC across videos** — i.e. **video-aggregated**, not
frame-pooled.

| source | estimand | status |
|---|---|---|
| **Ours** | **frame-pooled** AUC — every crop one row | ✅ known, `ESSENCE.md` §8a |
| **DeepfakeBench** | **frame-level** | ✅ verbatim, §3 |
| **F3-Net** | reads as **video-aggregated**; wording is ambiguous | ⚠️ **inference from an unclear sentence — do not state as fact** |

### Consequences — both must be written into the paper

1. **The 13× comparison may mix estimands.** Our 18 points is frame-pooled;
   F3-Net's +0.014 may not be. Qualify it: it is **scale context across two
   evaluation designs**, not a like-for-like ratio. Our own
   `results/analysis/aggregation/` shows switching estimands moves the estimate in
   all five seeds, so this is not a hypothetical concern.
2. **A candidate mechanical explanation for the 0.067 Xception discrepancy.**
   F3-Net reports 0.893, DeepfakeBench 0.8261. DeepfakeBench pools frames
   (verified); F3-Net appears not to. Estimand difference is a **plausible
   contributor**. ⚠️ **Hypothesis only** — the two also differ in subset, crops,
   training budget and implementation. Do not present it as the explanation.

### Why this strengthens rather than weakens the paper

Contribution 1 is that **protocol and estimand choices change apparent
performance**. Finding that two influential papers may report AUC at different
granularities — one stated plainly, one stated ambiguously — is a direct instance
of the thesis, not a problem for it. It also explains why we report both
frame-pooled and video-aggregated results rather than picking one.

**Owed:** a sentence in the limitations noting that our reference effect may be
measured at a different granularity than our own estimate.

---

## 12. Kapoor & Narayanan — two precision points (2026-09-17)

### 12.1 The 294 figure is about leakage in general

Verbatim, abstract:

> "we find 17 fields where leakage has been found, **collectively affecting 294
> papers** ... Based on our survey, we introduce a detailed taxonomy of **eight
> types of leakage**"

So 294 papers span **all eight types**, not [L3.2] nonindependence alone. Figure 1
confirms: *"Survey of 22 papers that identify pitfalls ... across 17 fields,
collectively affecting 294 papers."*

❌ Never write: "Kapoor & Narayanan classify nonindependence ... documented across
294 papers in 17 fields" — it implies all 294 are nonindependence cases.
✅ Write: "Kapoor and Narayanan identify nonindependence between training and test
samples as a form of leakage; **their broader review** documented leakage across
294 papers in 17 fields."

### 12.2 Their civil-war study is a direct precedent for our finding

Also in the abstract, and we had not been using it:

> "we conduct a reproducibility study of civil war prediction, where complex ML
> models are believed to vastly outperform traditional statistical models such as
> logistic regression (LR). **When the errors are corrected, complex ML models do
> not perform substantively better than decades-old LR**"

This is the closest published precedent to our own result: an apparent
architectural advantage that **did not survive correction of the evaluation**.

**Use it in the introduction.** It establishes that our kind of finding has a
precedent in a peer-reviewed venue, which pre-empts "this is just a null result".
It also positions us as the deepfake-detection instance of a documented pattern
rather than an isolated negative finding.

⚠️ Their correction dissolved the advantage outright; **ours does not** — our
interval contains zero *and* +0.014. Cite it as a precedent for the *pattern*, not
as a parallel result.

---

## 13. [L3.2] in full — two clauses we had missed (2026-09-17)

Independently confirmed by Charlie against the PDF. The three quotes already in §6
are verbatim. The fuller passage adds two things that materially improve our
framing.

### 13.1 The conditional clause licenses our careful wording

> "Nonindependence between training and test samples constitutes leakage,
> **unless the scientific claim is about a distribution that has the same
> dependence structure.**"

This is not a hedge on their part — it is a definition. Leakage is a property of
**the claim**, not of the split. Crop-randomised splitting is leakage *for a claim
about unseen videos*, and is not leakage *for a claim about unseen frames of known
videos*.

**Why this matters.** Under external review we adopted the wording "crop-randomised
splitting is not invalid in itself — it estimates a different quantity, performance
on unseen frames of known videos." That was adopted as a matter of caution. It
turns out to be **what the source literature actually says**. Cite this clause
directly when making that point; it converts a defensive hedge into a positive
claim with an anchor.

### 13.2 Their histopathology example is structurally identical to ours

> "a recent study on histopathology uses **different observations of the same
> patient** in the training and test sets. In this case, the scientific claim is
> being made about the ability to predict gene mutations in **new patients**;
> however, it is evaluated on data from **old patients** ... leading to a
> **mismatch between the test set distribution and the scientific claim**."

| their case | ours |
|---|---|
| different observations of the same **patient** | different crops of the same **video** |
| claim is about **new patients** | claim is about **unseen videos** |
| evaluated on **old patients** | evaluated on **seen videos** |

Use this as the introductory analogy. It is a published, peer-reviewed instance of
our exact structure in a different field, which makes the error legible to a reader
who knows nothing about deepfakes.

**Adopt their phrasing: "a mismatch between the test set distribution and the
scientific claim."** It is more precise than "leakage" and more informative than
"protocol gap", because it names *what* is mismatched. It also sidesteps the
attribution problem entirely — a mismatch is a property of a study design, not an
accusation about a researcher.

They also state the remedy as a requirement:

> "The train-test split **should account for the dependencies in the data** to
> ensure correct performance evaluation."

### 13.3 They state the general problem is hard — and we are in the tractable case

> "Handling nonindependence between the training and test sets **in general** (i.e.,
> without any assumptions about independence in the data) **is a hard problem**,
> because **we might not know the underlying dependency structure** of the task in
> many cases."

This sentence does two jobs for us, in opposite directions. Use both.

#### It answers "isn't this obvious? just split by video"

The reviewer objection contribution 1 is most exposed to is that video-disjoint
splitting is common sense. Kapoor & Narayanan say the general problem is **hard**,
precisely because the dependency structure is usually unknown.

FaceForensics++ is the fortunate case: the structure is **recoverable from the
metadata**. Video ids are explicit, and the `target_source` sequence naming lets us
recover the pair graph and run union-find over it — yielding 150 components, all of
size 2, `video_groups_per_component = 2.0` (`results/analysis/clusters/`).

So our claim is not "we thought of splitting by video". It is: *this is a domain
where the dependency structure can be recovered, so the cost of ignoring it can be
measured rather than merely asserted.* That is why the study is possible here and
hard elsewhere.

#### It also legitimises our own limitation 1

We do **not** have the true dependency structure either. Our components are
**source-target components, not verified human identities** — FF++ sequence ids are
not identity labels (`results/analysis/clusters/README.md`). The same actor could
appear in two unconnected components and we would not know.

That is exactly the situation they describe: *we might not know the underlying
dependency structure*. Limitation 1 in `ESSENCE.md` §8a is therefore not an
admission of sloppiness — it is **an instance of a documented open problem**, and
should be written that way, citing this sentence.

**Net effect:** the same sentence defends the contribution against "too obvious"
and defends the limitation against "you didn't go far enough". Cite it in both
places.

---

## 14. DeepfakeBench's evaluation module, enumerated in full (2026-09-17)

Confirmed by Charlie against p.5 of the PDF. The passage matches §3 verbatim, and
the **complete** paragraph lets us state the no-uncertainty finding more strongly.

### What the module contains, by their own enumeration

| category | items |
|---|---|
| metrics | ACC, AUC, AP, EER |
| performance visualisations | ROC-AUC curve, radar chart, histogram |
| analysis tools | Grad-CAM, t-SNE, per-detector custom visualisations |

### Why this is a stronger claim than what we had

§4 established "no measure of variability is reported anywhere" by **full-text
search** — an absence argument, which invites the reviewer response *"perhaps you
missed it."*

This paragraph is the authors **enumerating their own evaluation apparatus**, and
no element of it estimates uncertainty in a metric. Four point-estimate metrics,
three visualisations of point performance, two interpretability tools.

Upgrade the wording from *"we searched and found none"* to:

> **DeepfakeBench enumerates its evaluation module as four metrics (ACC, AUC, AP,
> EER) together with performance and interpretability visualisations; none of these
> estimates variability in a reported metric.**

That is a claim about what they say they do, not about what we failed to find.

⚠️ **One qualification, state it.** A histogram *is* a distribution — but a
distribution over examples or prediction scores, not an uncertainty estimate for
the metric. The enumeration does not say what the histogram plots. Do not claim it
cannot be one; claim only that **no item is described as a variability estimate**.
This is **our reading of their enumeration**, not their statement.

### Bearing on contribution 2

Their module standardises *which* number is computed and makes it comparable
across detectors. It does not address how much that number would move under
resampling or retraining. That gap is contribution 2's subject, and this
enumeration is the cleanest evidence that the gap is real rather than alleged.

---

## 15. Our own splitting claims, verified from the raw dumps (2026-09-17)

Prompted by a draft paragraph describing our protocol. Rather than trusting the
READMEs, these were recomputed from
`results/predictions_v8/ffpp_c40_vid_xception_seed0_test.csv` (3,000 rows).

| claim | result | status |
|---|---|---|
| The grouping key is the **target** video | `video_id == target_seq` in **3000/3000** rows | ✅ |
| A group holds the real video **and all four manipulated variants** | **30/30** groups carry Deepfakes, Face2Face, FaceSwap, NeuralTextures **and** real; **30/30** carry both labels | ✅ |
| Test target groups | **30** | ✅ |
| Components among them | **29** | ✅ matches `STATE-OF-PLAY.md` |
| Test targets whose partner is in train/val | **28 of 30 = 93.3%** | ✅ confirms the "~93%" in `REMAINING-WORK.md` C4 |
| Pairs with both members in test | **exactly one** — (251, 375) | ✅ explains the 30 → 29 collapse |

### Wording this licenses

> "We partition by target-video group, preventing crops and manipulated variants
> associated with the same target video from crossing the training–test boundary.
> This removes same-video overlap but not every dependency: source–target
> relationships connect otherwise distinct target groups across partitions — in our
> test split, **28 of 30 target groups have their manipulation partner in
> training**."

Use the number. "Can connect otherwise distinct groups" is abstract enough that a
reader cannot tell whether it is a technicality or a hole; 93% settles it and shows
the residual was measured rather than gestured at.

### The limitation and the null result are one fact

Because only one pair co-occurs in test, **target groups and components are nearly
the same partition here** (30 vs 29). That is precisely why
`results/analysis/unit_sensitivity/` found the choice between them almost
immaterial — half-width differing by 0.0002, no verdict changes across 8
conditional comparisons.

So the unit-sensitivity null is **not** evidence that clustering choice never
matters. It is evidence that *in this split* the two candidate units nearly
coincide. State it that way, and it stops looking like a convenient result.

---

## 16. Settled related-work wording for DeepfakeBench (2026-09-17)

Agreed text. Every clause traces to §3, §14 or the verified quote on p.5.

### The one-line formulation

> **DeepfakeBench standardises which performance estimate is computed; our study
> examines how uncertain that estimate is.**

Use this wherever the relationship needs stating in a sentence. It draws the
estimand/inference boundary without positioning us as correcting them.

### The related-work paragraph

> DeepfakeBench identifies inconsistent frame- and video-level evaluation as an
> obstacle to fair comparison and adopts frame-level metrics as a common basis. Its
> evaluation module enumerates four performance metrics — ACC, AUC, AP and EER —
> alongside ROC-AUC curve, radar chart, histogram, Grad-CAM, t-SNE and
> detector-specific visualisations. **None of these items is described as
> estimating sampling uncertainty or training-run variability in the reported
> performance metrics.** Thus DeepfakeBench standardises the estimand and
> evaluation pipeline across detectors, whereas our work examines the uncertainty
> that remains after standardisation, when test observations are clustered and
> trained models vary across runs.

### Why each clause is safe

| clause | grounding |
|---|---|
| "identifies **inconsistent** ... as an obstacle" | attributes unfairness to the inconsistency, not to pooling — the inversion to avoid (§3) |
| "enumerates ... " | their own list, p.5 (§14) |
| "**is described as**" | a claim about their description, not about what a histogram could in principle be |
| "the uncertainty that **remains after** standardisation" | positions us downstream, not in opposition |

### The histogram qualification — keep it

A histogram visualises a distribution, but **unless it is built from resampled
metric estimates or repeated training runs, it is not an uncertainty estimate for
AUC**. Their text does not say what theirs plots. This names the exact condition
under which it would count, which is why the "described as" phrasing is sufficient
and no stronger claim is needed.

### ⚠️ Attached to any citation of their +0.0010

DeepfakeBench's F3Net figure (0.8271 vs Xception 0.8261) carries the unresolved
scope ambiguity recorded in §3 as **E3**: their **paper** describes F3Net as
two-branch FAD+LFS; their **code** (`f3net_detector.py`) implements the FAD branch
only. Our comparison holds under either reading (14× or 40×), but **the sentence
citing +0.0010 needs a footnote saying which reading is assumed.**

### This paragraph is the template

It describes what the cited work did accurately, states the boundary of that
contribution, and claims only the remainder. Use the same shape when writing about
F3-Net: no adversarial framing is needed when the boundary is drawn precisely.

---

## 17. Contribution 2's headline, independently recomputed (2026-09-17)

The three-units result is contribution 2's signature finding. It was recomputed
from the raw dumps with a **separate implementation** of AUC and of the cluster
bootstrap — not by re-reading `results/analysis/cluster_boot/`.

Inputs: `results/predictions/ffpp_c40_vid_xception_seed0_test.csv` and
`..._f3net_seed0_test.csv`. Row alignment asserted on `path` before comparing.

| quantity | recomputed | stored | |
|---|---|---|---|
| AUC Xception | 0.8135 | 0.8135 | ✅ exact |
| AUC Xception+FAD | 0.7939 | 0.7939 | ✅ exact |
| difference | −0.0196 | −0.0196 | ✅ exact |
| units: frame / video / component | 3000 / 150 / 29 | 3000 / 150 / 29 | ✅ exact |
| **frame** 95% CI | [−0.0326, −0.0074] | [−0.0325, −0.0079] | ✅ **excludes zero** |
| **video** 95% CI | [−0.0616, +0.0161] | [−0.0606, +0.0167] | ✅ contains zero |
| **component** 95% CI | [−0.0522, +0.0095] | [−0.0534, +0.0096] | ✅ contains zero |

Endpoints differ in the third decimal, as expected from independent RNG draws at
4,000 replicates. **All three verdicts are identical.** The finding is a property
of the data, not of our implementation.

Definitions used, for reproducibility: the **video** unit is
`video_id + manipulation` (150 = 30 targets × 5 variants including real); the
**component** unit is union-find over the `target_seq`/`source_seq` pair graph,
giving 29.

### ⚠️ Naming trap — fix before drafting

`results/analysis/cluster_boot/ALL.csv` names the third unit **`identity`**.

It is **not** identity. `results/analysis/clusters/README.md` states plainly:
*"These are source-target components, not verified human identities. FF++ sequence
ids are not identity labels."*

The column name contradicts the documentation. Anyone reading the CSV without the
README will write "identity-level clustering" into the paper, which would be a
false claim about what we controlled for — and it collides directly with
limitation 1, which is precisely that we did **not** achieve identity-disjointness.

**Never write "identity" for this unit.** Write **"source-target component"**. The
column name is left as-is because the frozen results depend on it; this entry
exists so the name is not trusted.

---

## 18. Contribution 2's remaining numbers, recomputed (2026-09-17)

### 18.1 The canonical result reproduces exactly

`src/crossed_boot.py` was run with the documented command from `STATE-OF-PLAY.md`
§7. Output, unmodified:

```
seeds=[0,1,2,3,4]  items=150  components=29  aggregate=video
  crossed         95% CI [-0.0358, +0.0180]  hw 0.0269  P(>0.014)=0.0407  upper MC[+0.0176,+0.0184]
  non-additivity ratio 1.697
```

Every figure in `canonical.json` reproduced: interval, exceedance rate, Monte Carlo
band, non-additivity ratio. ✅

### 18.2 Frame-pooled and video-aggregated estimates both match

Independent reimplementation (own AUC, own bootstrap) against
`results/predictions/ffpp_c40_vid_*`:

| estimand | seeds 0–4 | match |
|---|---|---|
| frame-pooled | −0.0196, −0.0101, +0.0142, +0.0253, +0.0145 | ✅ exact, all five |
| video-aggregated (mean `logit_margin`) | −0.0303, −0.0208, +0.0203, +0.0306, +0.0164 | ✅ exact, all five |

Aggregation moves the estimate in **all five** runs and preserves sign in all five
— confirming `results/analysis/aggregation/`.

### 18.3 ⚠️ SUPERSEDED BY §19 — WRONG CAMPAIGN. Read §19 instead.

> **This table is the varying-split campaign (`results/predictions/`), not the
> V8 fixed-split runs behind the primary result.** Its mean is +0.00324, not the
> canonical −0.0092. The conceptual point stands; the claim that it bears on the
> primary result did not. Corrected in §19.

### 18.3 (as originally written — the aggregation *space* as a third estimand choice)

"Video-aggregated" is under-specified. Frames can be averaged in **logit space**
(what `src/cluster_boot.py` does) or in **probability space**. Both are defensible;
neither is conventionally reported.

| seed | mean-logit | mean-prob | Δ | exceeds +0.014? |
|---|---|---|---|---|
| 0 | −0.0303 | −0.0325 | +0.0022 | no → no |
| 1 | −0.0208 | −0.0192 | −0.0017 | no → no |
| 2 | **+0.0203** | **+0.0100** | **+0.0103** | **yes → NO** |
| 3 | +0.0306 | +0.0339 | −0.0033 | yes → yes |
| 4 | **+0.0164** | **+0.0056** | **+0.0108** | **yes → NO** |

- Mean |Δ| = **0.0057**, 41% of the reference effect
- Max |Δ| = **0.0108** = **0.8× the reference effect**
- **2 of 5 runs change whether they appear to exceed +0.014**, on nothing but the
  space in which frames were averaged

Sign is preserved in all five, so this changes magnitude and verdict, not
direction. Say so.

**Why it matters.** Contribution 1 shows protocol choice matters and contribution 2
shows the uncertainty unit matters. This is a **third layer**: a sub-choice inside
one estimand, invisible in every paper we have read, that moves the estimate by up
to 0.8× the effect under debate. It costs no GPU — a deterministic recomputation on
committed predictions — and it is the cheapest new result available.

⚠️ Scope: five runs, one comparison, c40, descriptive. Not a general claim about
aggregation spaces. **Owed:** replicate at c23 before it goes in the paper.

### 18.4 Documentation gap in the approved sentence

`canonical.json` carries
`estimand = "video-aggregated AUC (frames averaged to one score per video)"`, but
the **`manuscript_sentence` does not state it** — it says only "the mean FAD −
Xception difference across five training runs was −0.0092 AUC".

In a paper whose first contribution is that estimand choice changes the answer,
quoting the headline without naming its estimand is indefensible. And per §18.3 it
is not even fully specified by "video-aggregated" — the averaging space needs
naming too.

**Owed:** revise `manuscript_sentence` to name the estimand, and re-run
`tests/test_canonical.py`.

---

## 19. Aggregation space, done on the right campaign (2026-09-17)

§18.3 used `results/predictions/` — the **varying-split** campaign, which has no
`split_seed` column. The primary result comes from `results/predictions_v8/`,
where every file carries `split_seed=0`. The mean of the §18.3 table is **+0.00324**
against the canonical **−0.0092**: different experiments. Caught in review.

### 19.1 Two different things — do not conflate them

⚠️ **"Exact reproduction" applies to one of these and not the other.**

| what | result | status |
|---|---|---|
| Rerunning `src/crossed_boot.py` with the documented command | [−0.0358, +0.0180], P=0.0407 | ✅ **exact** — same code, same seed, same draws |
| An **independent reimplementation** under mean-logit | [−0.0360, +0.0175], P=0.0405 | ✅ **agrees within Monte Carlo error** — *not* exact reproduction |

The stored resampling indices from the frozen 50,000-replicate run were not
retained, so an independent implementation cannot reproduce it exactly and must not
claim to. The aggregation step is deterministic; the **bootstrap summary is not**
unless the resampling indices or RNG state are fixed.

**The frozen primary result stands unchanged.** Nothing in this section revises it.

### 19.2 The result — conclusion robust, quantities not

V8 fixed-split, 5 runs × 29 components, 50,000 replicates:

| | mean-logit (canonical) | mean-probability | shift |
|---|---|---|---|
| point estimate | **−0.0092** | **−0.0060** | +0.0032 (23% of reference) |
| 95% CI | [−0.0360, +0.0175] | [−0.0379, +0.0238] | — |
| half-width | 0.0268 | 0.0308 | **+15% wider** |
| **P(> +0.014)** | **0.0405** | **0.0825** | **doubles** |
| contains 0 | yes | yes | — |
| contains +0.014 | yes | yes | — |
| **verdict** | neither demonstrated nor excluded | **unchanged** | — |

**The primary conclusion is robust to the aggregation space.** Both intervals
contain zero and +0.014.

**But the quantities it is built from are not.** The exceedance rate **doubles**
(4.05% → 8.25%), the interval widens 15%, and per-seed estimates move by up to
**0.0125 = 0.89× the reference effect** (per-seed deltas +0.0053, +0.0125, −0.0114,
+0.0044, +0.0050).

This is the stronger version of the finding: the headline survives, and a number
quoted *in the approved manuscript sentence* doubles under a choice that our
documented audit found unstated.

### 19.3 Status — post-hoc, label it

⚠️ **This is a post-hoc estimand-sensitivity analysis.** The discrepancy was
observed before the analysis was specified. It is **not** prespecified and must
never be presented as if it were.

### 19.4 The conceptual decomposition — three distinct choices

Video-level AUC does not define an estimand. Three choices must each be stated:

1. **What is the evaluation unit?** (frame, video file, target group, component)
2. **Per frame, or summarised per video?**
3. **If summarised — in what score space, with which operator?**

Mean logit margin, $s_v = \frac{1}{n_v}\sum_i \operatorname{logit}(p_{vi})$, and
mean probability, $s_v = \frac{1}{n_v}\sum_i p_{vi}$, **do not commute with the
sigmoid**, so they can rank videos differently and yield different AUCs. Neither is
incorrect; they encode different aggregation rules.

### 19.5 Wording — required

❌ "two of five runs flip their verdict"
✅ **"In two of five varying-split runs, changing from mean-logit to
mean-probability aggregation moved the point estimate from above to below the
+0.014 reference magnitude."** A per-seed threshold crossing is **not** a
statistical verdict.

❌ "no paper specifies the averaging space"
✅ **"The papers in our documented audit did not state the score space used for
video aggregation."** Our audit did not systematically check methods sections,
supplements and evaluation code across the field.

### 19.6 Consequence for §18.4

The approved sentence quotes **4.07%**. Under mean-probability the same analysis
gives **8.25%**. A quoted figure that doubles under an unstated choice makes naming
the estimand — **and the averaging space** — mandatory, not stylistic.

### 19.7 The paired contrast — both arms on identical resamples

The comparison above is **paired**: one set of 50,000 component × training-run
resamples was generated and applied to *both* aggregation rules, so the contrast
carries no Monte Carlo noise between arms.

| | mean-logit | mean-probability |
|---|---|---|
| point estimate | −0.0092 | −0.0060 |
| 95% CI | [−0.0360, +0.0175] | [−0.0379, +0.0238] |
| half-width | 0.0268 | 0.0308 (+15%) |
| P(> +0.014) | 0.0405 | 0.0825 |
| contains 0 / +0.014 | yes / yes | yes / yes |

**Per-replicate paired difference (probability − logit):** mean **+0.00286**, with
95% of replicates in **[−0.0072, +0.0126]**.

So the aggregation rule is **not a constant offset** — it moves individual
resamples in both directions. Under the *same* resample, the two rules disagree
about whether that replicate exceeds +0.014 in **4.48%** of cases.

### 19.8 Framing — "robust conclusion, aggregation-sensitive quantities"

Use that phrase. **Not** "fragile quantities" — more precise and less sensational.

> The conclusion that this evaluation neither demonstrates nor excludes the
> reference gain survived both mean-logit and mean-probability video aggregation.
> Meanwhile the point estimate moved by 0.0032, the interval half-width increased
> by approximately 15%, and the bootstrap exceedance rate rose from approximately
> 4.1% to 8.3%.

### 19.9 Where it belongs — contribution 1, not contribution 2

| | defines | contribution |
|---|---|---|
| frame-pooled vs video-aggregated | **which quantity is estimated** | **1** (estimand) |
| mean-logit vs mean-probability | **how the video score is formed** | **1** (estimand) |
| frame / video / component resampling | **how uncertainty is calculated** | **2** (inference) |

The aggregation operator defines the video **score**; component resampling defines
how **uncertainty** is computed. Keep them apart — conflating them would merge the
two contributions that §10 just took the trouble to separate.

### 19.10 Manuscript wording — agreed

> As a post-hoc sensitivity analysis, we replaced mean-logit video aggregation with
> mean-probability aggregation. The estimated FAD–Xception difference shifted from
> −0.0092 to −0.0060 AUC, the crossed interval widened by approximately 15%, and
> the bootstrap exceedance rate above +0.014 increased from approximately 4.1% to
> 8.3%. Nevertheless, both aggregation rules produced the same substantive
> conclusion: neither demonstrated a FAD benefit nor excluded the reference-sized
> gain. Thus, the conclusion was robust, but its numerical warrant depended on an
> aggregation choice that must be reported explicitly.

---

## 20. Contribution 3's evidence, audited (2026-09-17)

### 20.1 What C3 rests on — all present and validated

| element | evidence | status |
|---|---|---|
| Primary data | 10 V8 prediction dumps, `split_seed=0`, identical test items | ✅ committed |
| Crossed result −0.0092, [−0.0358, +0.0180] | `results/analysis/crossed/`, `canonical.json` | ✅ **exact reproduction** (§18.1) |
| Conditionals: component-only, seed-only | same run | ✅ both exclude +0.014 |
| Exceedance 4.07%, MC band [0.0176, 0.0184] | same | ✅ |
| Canonical test suite | `tests/test_canonical.py` | ✅ **18 passed** |
| Reference effect +0.014 | F3-Net Fig. 7(a) p.12, Table 3 p.14 | ✅ verified from source |
| LQ = c40 | FF++ §3 p.5, quantization 40 | ✅ definitional (E2 closed) |
| Aggregation-space robustness | paired, 50,000 replicates | ✅ verdict unchanged (§19) |

**No experiment is missing for contribution 3.** Its evidence is the most complete
of the three.

### 20.2 Cost accounting — verified, with a caveat on which figure

From `results/analysis/efficiency/params_flops_latency.csv`:

| xception → f3net (Xception+FAD) | change |
|---|---|
| params | 20.81 → 20.86 M = **+0.24%** |
| GFLOPs | 1.48 → 1.52 = **+2.70%** |
| batch-32 latency | 40.7 → 42.0 ms = **+3.19%** |

✅ "FAD ≈ **+3%**" is accurate **for GFLOPs and latency**. It is **not** true of
parameters (+0.24%). Say which quantity when quoting it.

### 20.3 ⚠️ DISCREPANCY — the +44% latency figure does not match its own CSV

`results/analysis/efficiency/README.md` states the separate frequency branch costs
**"+31% FLOPs / +44% latency"**, and `ESSENCE.md` and `STATE-OF-PLAY.md` repeat it.

Recomputed from the generated CSV (baseline_spatial → full):

| | CSV | README |
|---|---|---|
| GFLOPs | 0.59 → 0.77 = **+30.5%** | +31% ✅ rounding |
| latency | 10.2 → 14.9 ms = **+46.1%** | **+44%** ❌ **2.1 points out** |

The CSV is the generated artifact and should win. **Do not quote +44% until this is
resolved** — either the README predates a regeneration, or a different pair was used.
The FLOPs figure is fine.

⚠️ This affects a **secondary** claim about our own `full` configuration, not the
FAD cost and not the primary result.

### 20.4 Test suite — what can and cannot run locally

With numpy, scipy, pandas, scikit-learn and pytest installed in a scratch venv:

```
107 passed, 13 errors
```

The 13 errors are **collection failures on torch-dependent modules** — `test_dct`,
`test_detector`, `test_engine`, `test_f3net`, `test_freq_dropout`,
`test_frequency_mask`, `test_sas`, `test_robustness`, `test_utils` and others.
They are **not failures**; torch is not installed locally and there is no GPU.

| | |
|---|---|
| run and pass locally | **107** |
| require Kaggle (torch) | the remainder of the ~253 |

**The "253 passed, 2 skipped" figure in `STATE-OF-PLAY.md` is a Kaggle figure** and
should be labelled as such — it cannot be reproduced on this machine.

⚠️ The venv lives in the **session scratchpad and is cleared between sessions**.
Reinstall with:
`pip install numpy scipy pandas scikit-learn pytest`

---

## 21. Test suite verified on Kaggle — 2026-09-17

Run by Charlie in a **CPU-only** Kaggle session (no GPU quota consumed), on a fresh
clone of the repo at `0f464e2`:

```
python -m pytest tests/ -q
```

| | |
|---|---|
| **pytest exit code** | **0** — authoritative: pytest exits non-zero on any failure or error |
| failures / errors | **none** — no `F` or `E` in 257 progress characters |
| skipped | **2** |
| approximate total | **~257** — inferred from the 28/56/84/100% progress markers |
| torch-dependent modules | **all ran and passed**, including the 13 that cannot even collect locally |

⚠️ **The total is approximate.** Kaggle truncated the summary line across three
attempts, including with `-p no:warnings`, `--tb=no` and `subprocess` capture. The
**pass/fail status is certain** (exit 0); the **count is inferred** from progress
markers and is recorded as approximate rather than invented precisely.

### What this replaces

`STATE-OF-PLAY.md` carried **"253 passed, 2 skipped"**, a figure from an earlier
Kaggle run that had not been re-verified. It is now known to be **stale** — the
suite has grown since. Do not quote 253.

### What it confirms

- Nothing broke across the 21 commits of 2026-09-17
- The 13 modules invisible to local runs — `test_dct`, `test_detector`,
  `test_engine`, `test_f3net`, `test_freq_dropout`, `test_frequency_mask`,
  `test_sas`, `test_robustness`, `test_utils` and others — are healthy
- The warnings observed are harmless: matplotlib pyparsing deprecations, and a
  `pin_memory` notice consistent with a CPU-only session

### How to re-run it

CPU accelerator, two cells, no installs (Kaggle ships torch, numpy, scipy,
scikit-learn, pandas and pytest):

```
!git clone https://github.com/Charles-AM/-DeepTrace.git /kaggle/working/dt
!cd /kaggle/working/dt && python -m pytest tests/ -q
```

⚠️ Never `pip install -r requirements.txt` on Kaggle — it breaks the pre-installed
torch build (`docs/REPRODUCIBILITY.md`).

---

## 22. F3-Net's splitting procedure — verified, and it is video-level (2026-09-18)

Asked directly: is there any reference showing a target paper used crop-randomised
splitting? **No. And the target paper states the opposite.**

**F3-Net, §4.1 Setting, p.10, verbatim:**

> "FaceForensics++ is a face forgery detection video dataset containing 1,000 real
> videos, in which **720 videos are used for training, 140 videos are reserved for
> validation and 140 videos for testing**."

That is **video-level splitting**, matching FF++'s official partition exactly
(§7). F3-Net did not crop-randomise.

### What this settles

| claim | status |
|---|---|
| FF++ ships official video-level splits, 720/140/140 | ✅ §7 |
| **F3-Net uses that video-level split** | ✅ **verified, stated in their own words** |
| Any paper uses crop-randomised splitting | ❌ **no evidence, and none sought since** |
| **We** used crop-randomised splitting before 2026-09-05 | ✅ our own logs |

### Consequences — three, all binding

**1. Contribution 1 cannot be framed as correcting F3-Net's protocol.** Their
protocol is the one we recommend. Any sentence implying otherwise is false, not
merely unsupported.

**2. F3-Net's +0.014 is not inflated by crop-randomisation.** We should never have
left that ambiguous. Their reference effect was measured under video-disjoint
evaluation, as ours is — the two are **protocol-matched**, which strengthens the
comparison rather than weakening it.

**3. The motivating case for contribution 1 is our own pipeline.** The honest
framing, already adopted: *we measured what the choice costs, using the alternative
our own early pipeline used.* No claim about anyone else's practice is made, needed,
or available.

### Why the demonstration is still worth making

A reasonable challenge follows: if the target paper split correctly and we have no
evidence anyone splits by crop, why measure the cost?

- It **quantifies the cost of a decision**, which is useful independent of who makes
  it — and nobody had quantified it in this domain.
- It **establishes the scale** against which a +0.014 claim must be judged: a single
  protocol decision moves the score by ~18 points.
- Kapoor & Narayanan document nonindependence across **294 papers in 17 fields**
  (§12.1), so the error class is demonstrably live in science generally — just not
  demonstrated here, by us, about anyone.
- **Our own pipeline made it**, which is the honest motivating case and requires no
  speculation about others.

⚠️ **Not checked, and not to be asserted:** whether papers in this area generally
*state* their splitting procedure. F3-Net does. We have examined one paper's
methods section on this point and must not generalise from it.

---

## 23. Focal loss — where our settings match the source and where they do not (2026-09-21)

Verified against Lin, Goyal, Girshick, He, Dollár, *Focal Loss for Dense Object
Detection*, ICCV 2017, pp. 2999–3007 (arXiv:1708.02002).

| | paper | ours | |
|---|---|---|---|
| formula | FL(p_t) = −α_t(1 − p_t)^γ log(p_t) | identical, `src/losses.py` | ✅ **matches** |
| **γ** | *"we found γ = 2 to work best"*; *"γ = 2 (our default setting)"* | **2.0** | ✅ **matches** |
| **α** | *"best α's ranged in just [.25, .75] (we tested α ∈ [.01, .999]). **We use γ = 2.0 with α = .25** for all experiments"* | **0.200** | ⚠️ **below their best range** |

### The α difference, stated precisely

Our α = 0.200 is **derived from the training class balance** (4,800 real : 19,200
fake), not tuned. It falls **just below** the [0.25, 0.75] interval Lin et al.
report as best.

Their optimum was found under RetinaNet's foreground/background imbalance of
roughly 1:1000; ours is 1:4, so their range is not obviously binding here. **But
that is an argument for why our value is reasonable, not for why it matches** —
different claims, and only the first is available to us.

### ✅ Required methods wording

> We train with focal loss (Lin et al., 2017) using γ = 2.0, the value that work
> reports as best. The class-weighting term α = 0.200 is set from the training
> class balance rather than tuned; it falls just below the [0.25, 0.75] range
> reported as best in that paper, whose class imbalance differs substantially from
> ours. No focal-loss hyperparameter was searched.

❌ Never write *"following Lin et al."* unqualified — true of γ, false of α.

### Why this matters beyond one citation

It is the **only place our training recipe departs from a cited source**, and it is
now stated rather than discovered. It also makes the no-hyperparameter-search
disclosure concrete: we can now say exactly where our settings sit relative to the
source's own recommendations.
