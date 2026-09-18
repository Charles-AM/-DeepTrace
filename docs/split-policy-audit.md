# Split-policy audit — do FF++ papers state how they partitioned?

**Status: COMPLETE, 2026-09-18. Seven papers examined; four independently confirmed by Charlie the same day.**

## The question

Not *"do papers split by crop?"* — we have no evidence of that and F3-Net
explicitly splits by video (ledger §22). The answerable question is:

> **When a paper trains and evaluates on FaceForensics++, does it state its
> train/test partitioning procedure clearly enough for a reader to know whether
> videos span the split?**

That is checkable by reading methods sections, and either answer is reportable.

## Criteria — fixed before reading, so the result is not shaped by it

| verdict | means |
|---|---|
| **EXPLICIT** | states video counts (e.g. 720/140/140), or says "official split", or names the FF++ split files |
| **IMPLICIT** | says "following [FF++]" or "standard protocol" without numbers — recoverable but not stated |
| **UNSTATED** | describes only frames/crops sampled, with no partition description. A reader cannot tell whether videos span the split |
| **CROP-RANDOMISED** | explicitly shuffles images before splitting |

Record the **verbatim sentence** in every case, or "none found" with the sections
searched.

## Where to look, in order

1. **"Experimental Setup" / "Setting" / "Implementation Details"** — usually §4.1
2. **"Dataset"** subsection
3. **Supplementary material**, if any
4. **The released code** — `dataset.py`, `split`, `train_list`, `.json` split files

## Search terms — Ctrl+F

`split` · `720` · `140` · `training set` · `test set` · `partition` · `official`
· `protocol` · `videos are used`

One paper is already done and shows what EXPLICIT looks like:

> **F3-Net (ECCV 2020) §4.1 p.10 — EXPLICIT.** *"FaceForensics++ is a face forgery
> detection video dataset containing 1,000 real videos, in which 720 videos are
> used for training, 140 videos are reserved for validation and 140 videos for
> testing."*

## Recommended papers — all FF++-trained, top venue, open access

| # | paper | venue | why this one |
|---|---|---|---|
| 1 | **CADDM — Dong et al., "Implicit Identity Leakage"** | **CVPR 2023**, pp. 3994–4004 | **Start here.** It is *about* identity leakage in this dataset, so it is the paper most likely to describe partitioning carefully — and the most informative if it does not |
| 2 | **SBI — Shiohara & Yamasaki, "Self-Blended Images"** | **CVPR 2022 (oral)**, arXiv:2204.08376 | Highly cited, FF++-trained, oral — a high-visibility example |
| 3 | **FreqDebias — Kashiani et al.** | **CVPR 2025**, pp. 8775–8785 | **Two-for-one.** Also answers our open question of whether frequency methods *generally* claim their largest gains under compression (currently marked do-not-assert, ledger §2) |
| 4 | **Ojha et al., "Towards Universal Fake Image Detectors"** | **CVPR 2023**, arXiv:2302.10174 | Different framing (frozen CLIP), so a useful contrast |
| 5 | **UCF — Yan et al.** | **ICCV 2023** | Recent, FF++, same group as DeepfakeBench |
| 6 | **DF40 — Yan et al.** | **NeurIPS 2024** | A benchmark paper; benchmarks are where protocol is most likely stated |

Already done: **F3-Net** (EXPLICIT) · **DeepfakeBench** (standardises evaluation,
does not split per-paper — ledger §3)

## Findings — 2026-09-18

All PDFs fetched from arXiv and searched with `pdftotext` + grep.

| paper | venue | verdict | evidence |
|---|---|---|---|
| **F3-Net** | ECCV 2020 | **EXPLICIT** | §4.1 p.10: *"720 videos are used for training, 140 videos are reserved for validation and 140 videos for testing"* |
| **SBI** | CVPR 2022 | **EXPLICIT** (by naming) ✅✅ | *"We follow the official train/test splits for all datasets except FFIW where we use the original validation set as our test set because the official test set has not been released yet"* — names the protocol **and** states its one deviation |
| **CADDM** | CVPR 2023 | **EXPLICIT** (by counts) ✅✅ | §5.1: trained on *"720 original videos"*; evaluated on FF++ *"which contains 140 original videos and 700 fake videos"*. ⚠️ **Never says "official split"** and never mentions the validation set — 720 + 140 = 860 of 1,000. Explicit enough that a reader knows videos do not span the split, but weaker than naming the protocol |
| **DF40** | NeurIPS 2024 | **EXPLICIT** (naming + counts) ✅✅ | **The strongest of the four.** *"**We adhere to the official data split method**, which uses 720 selected videos for training and 140 for testing and validation."* Names the protocol and gives the counts |
| **FreqDebias** | CVPR 2025 | **IMPLICIT** ✅✅ | *"For preprocessing and training, we adhere to the configurations outlined in DeepFakeBench [64] to maintain a fair comparison"* — defers rather than states. Backbone ResNet-34, 256×256, 50 epochs — all differ from ours |
| **UCF** | ICCV 2023 | ⚠️ **UNSTATED** | **Zero occurrences** of split, partition, 720, 140, train set, training videos, held-out or divided in the entire paper. Its Datasets section names the datasets and compression level only |
| Ojha et al. | CVPR 2023 | **n/a** | Does not use FF++ — one mention, in the reference list. It targets GAN/diffusion image detection. Out of scope |

### The headline

**Zero of six crop-randomise.** Four state their partition explicitly, one defers
to a benchmark, one does not state it.

### Two findings beyond the question asked

**1. FreqDebias does NOT claim its largest gains under compression.** It trains on
FF++ **HQ** (c23) and frames its contribution as *generalisation* across datasets,
not compression robustness. Compression appears only in a descriptive sentence
listing FF++'s three levels.

→ **This settles the do-not-assert item in ledger §2.** The claim *"frequency
methods report their largest gains under heavy compression"* is **specific to
F3-Net** and must never be pluralised. The most recent frequency paper in our
bibliography does not make it.

**2. The estimand choice propagates — and is described as the convention.**
FreqDebias states it directly:

> *"To benchmark our method, **we follow the deepfake detection studies [8, 62, 64,
> 65] and adopt the frame-level area-under-the-curve (AUC)**, and Equal Error Rate
> (EER) metrics."*

They adopt frame-level AUC and **characterise it as following four prior studies**.
UCF states no metric granularity at all.

→ Frame pooling is not one benchmark's isolated choice. It is inherited by papers
adopting that benchmark, and at least one of them describes it as the prevailing
convention, citing four works.

⚠️ **That is FreqDebias's characterisation, not our finding.** Write *"FreqDebias
describes frame-level AUC as the convention it follows, citing four studies."*
**Never** *"frame-level AUC is the field norm"* — we have not surveyed that.

### How to verify each — 5 minutes

| paper | arXiv | where | search for |
|---|---|---|---|
| F3-Net | 2007.09355 | §4.1 Setting, p.10 | `720 videos are used` |
| SBI | 2204.08376 | Experiments → Datasets | `official train/test splits` |
| CADDM | 2210.14457 | §5.1 Experiment Setting → Datesets *(their typo)* | `720 original videos` |
| DF40 | 2406.13495 | appendix, protocol section | `720 selected videos` |
| FreqDebias | 2509.22412 | Implementation detail | `adhere to the configurations` |
| UCF | 2304.13949 | §Datasets | `split` — **expect zero hits** |
| Ojha | 2302.10174 | anywhere | `FaceForensics` — one hit, in references |

Reproduce the whole audit:

```
curl -sL -o p.pdf https://arxiv.org/pdf/<ID> && pdftotext p.pdf p.txt
grep -n -i "split\|partition\|720\|140 videos\|official" p.txt
```

## ⚠️ What may be concluded

This is a **small, non-random, convenience sample** chosen from our own related
work. Whatever it shows:

✅ *"Of the six FF++ papers whose methods sections we examined, four stated their
partitioning procedure explicitly, one deferred to a benchmark's configuration, and
one did not state it. **None described crop-randomised splitting.**"*

❌ Any statement about "the field", "most papers", or a prevalence rate.

If most state it explicitly, **that is a fine result** — it means protocol
reporting in this area is better than the general ML picture Kapoor & Narayanan
describe, and contribution 1 stands on the controlled demonstration alone, which
is where it already stands.
