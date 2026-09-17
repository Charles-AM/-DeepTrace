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
| **Reports no measure of variability anywhere** | full-text search, see §4 | ✅ |
| **"LQ" = c40** | **CLOSED 2026-09-17.** FF++ (arXiv:1901.08971 §3, *Postprocessing – Video Quality*) defines the mapping itself: *"To generate high quality videos, we use a light compression denoted by HQ (constant rate quantization parameter equal to 23) ... Low quality videos (LQ) are produced using a quantization of 40."* F3-Net cites FF++ [50] and defines its labels identically (*"LQ indicates low quality (heavy compression), HQ indicates high quality (light compression) and RAW indicates raw videos without compression"*). HQ = quantization 23 = c23; LQ = quantization 40 = c40 | ✅ **verified** — definitional chain, no longer an inference. F3-Net never writes "c40", but uses FF++'s own labels in FF++'s own sense |
| **The +0.014 is measured on the LQ (heavy compression) task** | two independent captions — Fig. 7(a) p.12 *"Ablation study of the proposed F3-Net on the low quality task(LQ)"*; Table 3 p.14 *"...on FAD in FF++ low quality (LQ)"* | ✅ verified 2026-09-17 |
| **F3-Net is explicitly motivated by heavy compression** | abstract: *"especially wins a big lead upon low-quality media"*; intro: *"if the visual quality ... is tremendously degraded, such as compressed by JPEG or H.264 ... the forgery artifacts ... cannot be captured in RGB domain any more"*; contributions: *"significantly improves the performance over low-quality forgery media"* | ✅ verified 2026-09-17 |
| F3-Net's gain over Xception shrinks as quality rises: LQ +0.040, HQ +0.018, RAW +0.006 | Table 1 p.9 — Xception 0.893/0.963/0.992, F3-Net(Xception) 0.933/0.981/0.998 | ✅ **confirmed visually 2026-09-17**; the earlier text reconstruction matched exactly |
| **The full-system gain is backbone-dependent**: +0.040 on Xception vs **+0.022** on Slowfast at LQ | Table 1 p.9 — Slowfast 0.936 → F3-Net(Slowfast) 0.958; same pattern at HQ (+0.018 vs +0.011) and RAW (+0.006 vs +0.005) | ✅ their numbers; **our inference** from them |
| Table 1 reports **no AUC at all** for five of the twelve methods (Steg.Features, LD-CNN, Constrained Conv, CustomPooling CNN, MesoNet) and **no Acc** for Face X-ray | Table 1 p.9, dashes in the respective columns | ✅ observed directly |
| "**Frequency methods** claim their largest gains under compression" (plural, as a class) | only F3-Net checked. FreqDebias and others **not** examined | ❌ **do not assert — narrow to F3-Net** |

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
| **Reports no measure of variability anywhere** | full-text search, see §4 | ✅ |
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

### Still unverified — 7

Focal Loss (Lin et al., ICCV 2017) · Efron & Tibshirani (1993) ·
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
demonstration on byte-identical crops; it needs no claim about others' practice.

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
