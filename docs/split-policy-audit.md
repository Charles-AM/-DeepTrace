# Split-policy audit — do FF++ papers state how they partitioned?

**Status: template. No papers examined yet beyond F3-Net.**

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

## Findings

Fill in as you go. **Verbatim quotes only.**

| paper | verdict | quote / sections searched | date |
|---|---|---|---|
| F3-Net, ECCV 2020 | **EXPLICIT** | §4.1 p.10, quoted above | 2026-09-18 |
| CADDM, CVPR 2023 | | | |
| SBI, CVPR 2022 | | | |
| FreqDebias, CVPR 2025 | | | |
| Ojha et al., CVPR 2023 | | | |
| UCF, ICCV 2023 | | | |
| DF40, NeurIPS 2024 | | | |

## ⚠️ What may be concluded

This is a **small, non-random, convenience sample** chosen from our own related
work. Whatever it shows:

✅ *"Of the N FF++ papers whose methods sections we examined, M stated their
partitioning procedure explicitly."*

❌ Any statement about "the field", "most papers", or a prevalence rate.

If most state it explicitly, **that is a fine result** — it means protocol
reporting in this area is better than the general ML picture Kapoor & Narayanan
describe, and contribution 1 stands on the controlled demonstration alone, which
is where it already stands.
