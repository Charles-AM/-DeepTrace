# Seed-variance audit — do recent FF++ papers propagate training-run variation?

**Status: COMPLETE, 2026-09-26. Eight papers examined, ECCV 2020 – CVPR 2025.
All PDFs fetched from arXiv, extracted with `pdftotext`, searched with `grep`.**

Companion to `docs/split-policy-audit.md`, which asked the same kind of question
about partitioning. This one supplies the motivation for **contribution 3**.

## The question

> When a recent top-tier FF++ deepfake-detection paper compares two methods, does
> it propagate variation across training runs into that comparison?

## Criteria — fixed before reading

| verdict | means |
|---|---|
| **PROPAGATED** | headline comparison carries run-to-run uncertainty (±sd, error bars, CI over seeds) |
| **REPORTED-NOT-PROPAGATED** | multiple runs / sd mentioned, but headline numbers are single point estimates |
| **SINGLE-RUN** | one number per config; states or implies a single training run |
| **UNSTATED** | no mention of seeds, repetition, or run variance anywhere |

Search terms: `seed` · `standard deviation` · `std` · `variance` · `deviation` ·
`error bar` · `confidence interval` · `averaged over` · `three runs` · `five runs`
· `repeated` · `trials` · `±` · `randomly` · `reproduc` · `initializ`

## Findings

| paper | venue | verdict | evidence |
|---|---|---|---|
| **DF40** — Yan et al. | **NeurIPS 2024** | **SINGLE-RUN (disclosed)** ⭐ | NeurIPS checklist, verbatim: *"Did you report error bars (e.g., with respect to the random seed after running experiments multiple times)? **[No]** We use the fixed seed that is the same as DeepfakeBench [90] to align the settings and facilitate fair comparison."* |
| **DeepfakeBench** — Yan et al. | NeurIPS 2023 | **UNSTATED** | Zero hits for seed / sd / error bar / variance. Its 12 uses of "reproduc*" all concern **code availability and unified protocol**, never run variance. "three time*" refers to duplicating the UADFV dataset |
| **FreqDebias** — Kashiani et al. | CVPR 2025 | **UNSTATED** | Zero hits. "average of" ×2 = spectrum averaging and cross-domain result averaging |
| **CADDM** — Dong et al. | CVPR 2023 | **UNSTATED** | Zero hits. "random*" ×6 = minibatch sampling, MFS window scale, t-SNE sampling |
| **UCF** — Yan et al. | ICCV 2023 | **UNSTATED** | "variance" ×2 = AdaIN feature normalisation; "deviation" ×1 = deviation from the default FF++ quality setting |
| **SBI** — Shiohara & Yamasaki | CVPR 2022 | **UNSTATED** | Only hit is an **aggregation** statement (see finding 2) |
| **NPR** | CVPR 2024 | **UNSTATED** | All three "variance" hits are the substring in *translation invariance* |
| **F3-Net** — Qian et al. | ECCV 2020 | **UNSTATED** | "variance" = hand-crafted *noise variance* features; one "std" is an email domain (`@std.uestc.edu.cn`) |

### The headline

**Zero of eight propagate training-run variation.** Seven never raise it. One —
DF40, NeurIPS 2024 — is asked directly by the venue's own checklist and answers
**No**, giving a fixed seed matched to DeepfakeBench as the reason.

## Finding 1 — what DF40's answer does and does not show

DF40's reasoning is legitimate as far as it goes: a shared fixed seed removes a
confound *between* methods, so the comparison is matched. ✅ State that.

What it does not do is make the resulting difference generalisable beyond that one
draw. The interval is conditional on a single training run per method, so
run-to-run variation is **excluded from the comparison rather than accounted for
in it**. That substitution — fixing a variance source and treating the result as
fair — is exactly what contribution 3 measures.

❌ Never write that DF40 was careless or that fixing the seed is an error. It is a
design choice with a stated rationale, and the paper disclosed it.
✅ Write: *"DF40 states that it did not report error bars with respect to the random
seed, using a fixed seed matched to DeepfakeBench to align settings."*

## Finding 2 — two CVPR papers call **opposite** estimands the convention

Unplanned, and it strengthens **contribution 1** more than anything we had.

> **SBI, CVPR 2022:** *"We report the **video-level** area under the receiver
> operating characteristic curve (AUC) to compare with prior works. **Typically**,
> frame-level predictions are averaged over video frames."*

> **FreqDebias, CVPR 2025:** *"we follow the deepfake detection studies [8, 62, 64,
> 65] and adopt the **frame-level** area-under-the-curve (AUC)"*

Both present their choice as following prior work. They choose **different
estimands**. So the aggregation convention is not one thing the field agrees on —
it is two, each described as conventional, three years apart, both at CVPR.

This is the field's own evidence for contribution 1, not our inference.

## Limits of this audit — what must NOT be claimed

⚠️ **Eight papers is not a survey of the field.**

- ❌ *"No deepfake detection paper propagates seed variance."* Not established.
- ✅ *"Among eight FF++ deepfake-detection papers from ECCV 2020 to CVPR 2025, none
  propagate training-run variation into their reported comparisons."*
- ❌ *"Frame-level AUC is the field norm."* Still unsurveyed (ledger §2 discipline).
- ✅ *"FreqDebias describes frame-level AUC as the convention it follows, citing four
  studies; SBI describes video-level AUC as what it reports to compare with prior
  works, calling frame averaging typical."*

Absence of a term in a PDF is evidence the paper does not discuss it, **not**
evidence the authors ran only one seed. `UNSTATED` means unstated.

## How to verify each — 5 minutes

| paper | arXiv | search for | expect |
|---|---|---|---|
| DF40 | 2406.13495 | `Did you report error bars` | the verbatim **[No]** |
| DeepfakeBench | 2307.01426 | `seed` | zero hits |
| FreqDebias | 2509.22412 | `seed`, `frame-level area` | zero; the metric sentence |
| CADDM | 2210.14457 | `seed`, `deviation` | zero |
| UCF | 2304.13949 | `seed` | zero |
| SBI | 2204.08376 | `video-level area`, `Typically` | the aggregation sentence |
| NPR | 2312.10461 | `variance` | only *translation invariance* |
| F3-Net | 2007.09355 | `seed` | zero |
