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
| "LQ" = c40 | p.429/996: *"LQ indicates low quality (heavy compression), HQ … light compression, RAW … without compression"*. The **c-notation is never used**; the mapping follows from FF++ shipping exactly these three levels | ⚠️ inference, not a stated equivalence |

## 3. DeepfakeBench (Yan et al., NeurIPS 2023) — arXiv:2307.01426

| claim | evidence | status |
|---|---|---|
| Standardises data management, implementation framework, evaluation metrics/protocols | abstract, quoted | ✅ |
| Adopts **frame-level** evaluation | paper body: *"Our benchmark currently adopts the frame level evaluation to build a fair basis for comparison among detectors"* | ✅ |
| Identifies frame-vs-video inconsistency as causing "unfair comparisons" | same passage — **they recognise it; they do not ignore it** | ✅ |
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
| **~20 citations in Groups A, G, H of `related-work.md`** | written from memory, never checked against dblp/CVF/arXiv. **Includes the dataset paper and the backbone** — the paper cannot be submitted without these verified | a few hours |
| Bouthillier et al.: do they include test-content clustering among variance sources? | if not, that is precisely our addition; if so, our novelty narrows to the domain demonstration | read one paper |
| Kapoor & Narayanan: which leakage category covers crop-randomised splitting? | use their taxonomy term rather than inventing one | read one paper |
| DeepfakeBench F3Net numbers | read off a table image, then confirmed in extracted text — but worth one direct look at the PDF | minutes |
| Re-verify Groups B–F | last checked 2026-09-03/04 | an hour |
