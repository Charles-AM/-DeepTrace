# Split-policy audit — independent check sheet

I ran this audit on 2026-09-18 (`docs/split-policy-audit.md`). **This sheet exists
so you can confirm or contradict it**, not take my word. My verdict and the quote I
found are given for each — if yours differs, mine is wrong and I want to know.

Open each PDF, go to the section named, Ctrl+F the term given.

---

## 1. CADDM — "Implicit Identity Leakage" · CVPR 2023
**PDF:** https://arxiv.org/pdf/2210.14457
**Section:** §5.1 "Experiment Setting" → **"Datesets"** *(their typo — search that spelling)*
**Ctrl+F:** `720 original videos`

> **My verdict: EXPLICIT.** *"We trained our models on the widely-used dataset
> FaceForensics++ (FF++). FF++ contains 4320 videos, i.e. 720 original videos
> collected from YouTube..."* and *"We evaluated our approach performance on the
> following datasets: (1) FF++, which contains 140 original videos and 700 fake
> videos."*

**Why this one matters most:** the paper is *about* identity leakage in this
dataset. If anyone describes partitioning carefully it should be them.

---

## 2. SBI — "Self-Blended Images" · CVPR 2022 (oral)
**PDF:** https://arxiv.org/pdf/2204.08376
**Section:** Experiments → Datasets
**Ctrl+F:** `official train/test splits`

> **My verdict: EXPLICIT.** *"We follow the official train/test splits for all
> datasets except FFIW where we use the original validation set as our test set
> because the official test set has not been released yet."*

Note they state the **exception** too — that's unusually good practice, worth
noting if you write about reporting quality.

---

## 3. DF40 · NeurIPS 2024
**PDF:** https://arxiv.org/pdf/2406.13495
**Section:** appendix, protocol description (it's a long paper — search rather than scroll)
**Ctrl+F:** `720 selected videos`

> **My verdict: EXPLICIT.** *"uses 720 selected videos for training and 140 for
> testing and validation... we use the 720 corresponding fake videos for training
> and the original 720 real videos as real samples."*

---

## 4. FreqDebias · CVPR 2025
**PDF:** https://arxiv.org/pdf/2509.22412
**Section:** "Implementation detail"
**Ctrl+F:** `adhere to the configurations`

> **My verdict: IMPLICIT.** *"For preprocessing and training, we adhere to the
> configurations outlined in DeepFakeBench [64] to maintain a fair comparison."*
> It defers to a benchmark rather than stating its own partition.

**Also check two things here** — both matter beyond this audit:

- **Ctrl+F `frame-level`** → I found *"evaluated on other datasets using the
  frame-level AUC metric"*. Confirms the estimand choice propagates from
  DeepfakeBench to papers adopting it.
- **Ctrl+F `compress`** → I found only a descriptive sentence listing FF++'s three
  quality levels. **They do NOT claim their largest gains under compression**, and
  they train on **HQ**, not LQ. This is what killed the "frequency methods" plural
  — if you find otherwise, that claim needs reopening.

---

## 5. UCF · ICCV 2023
**PDF:** https://arxiv.org/pdf/2304.13949
**Section:** §Datasets
**Ctrl+F:** `split` — then `partition`, `720`, `140`, `train set`

> **My verdict: UNSTATED.** I found **zero occurrences** of any of those terms.
> Its Datasets section names the four datasets and the compression level, and says
> *"Following previous works, the HQ version of FF++ is adopted by default"* — about
> which data, not how it was divided. I also found **no** statement of metric
> granularity (frame-level or video-level).

**This is the one most worth double-checking**, because a null result is the
easiest to get wrong. If you find a partition statement I missed, say so — it
changes the audit's headline.

---

## 6. Ojha et al. — "Towards Universal Fake Image Detectors" · CVPR 2023
**PDF:** https://arxiv.org/pdf/2302.10174
**Ctrl+F:** `FaceForensics`

> **My verdict: NOT APPLICABLE.** One occurrence, in the reference list. The paper
> targets GAN/diffusion-generated images, not FF++ video. I recommended it before
> checking; it's out of scope.

---

## Already done, no need to recheck

**F3-Net** (ECCV 2020) — EXPLICIT, §4.1 p.10: *"720 videos are used for training,
140 videos are reserved for validation and 140 videos for testing."*
**DeepfakeBench** (NeurIPS 2023) — standardises evaluation; does not split
per-paper (ledger §3).

---

## The headline to confirm or break

> **Zero of six crop-randomise.** Four explicit, one implicit, one unstated.

If that holds, contribution 1 stays a controlled demonstration of what a choice
costs — never a critique of anyone's practice. If you find a paper that *does*
crop-randomise, tell me immediately: it would change the framing of the entire
first contribution.

## What may be written afterwards

✅ *"Of the six FF++ papers whose methods sections we examined, four stated their
partitioning explicitly, one deferred to a benchmark, and one did not state it.
None described crop-randomised splitting."*

❌ Anything about "the field", "most papers", or a prevalence rate. Six papers
chosen from our own bibliography is a convenience sample, not a survey.
