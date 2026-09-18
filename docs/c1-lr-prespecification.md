# Prespecification — C1, learning-rate sensitivity

**Written and committed before the runs.** Git history is the evidence. In a paper
about evidential warrant, fixing the decision rule after seeing the numbers it
governs would be the exact failure being criticised.

⚠️ **Timing, stated precisely.** The crossed result already exists and is frozen
(`results/canonical.json`, −0.0092, CI [−0.0358, +0.0180]). This is **not**
preregistration before any result. The accurate description is:

> Having observed a null FAD result under the reference learning rate, we
> prospectively specified the design and decision rule for a sensitivity probe
> before running it.

## 1. The question

The strongest available objection to the case study is: **"your null may simply be
an untuned FAD."** FAD changes the network input from 3 to 9 channels, so a
different optimal learning rate is plausible.

This probe asks whether the FAD − Xception difference at c40 is materially
sensitive to the learning rate, **within a symmetric range around the reference
value used by every existing run.**

## 2. Design — fixed before running

| | |
|---|---|
| Learning rates | **1e-4, 3e-4, 1e-3** — the reference value `3e-4` (from `results/reference/train_args_c40_vid.json`) plus one decade either side |
| Configs | **`xception` and `xception_fad`** — **both arms at every lr** |
| Cells | 3 × 2 = **6 runs** |
| Split | fixed, `--split-seed 0`, verified against `results/reference/ffpp_c40_vid_seed0_test_split.json` before training |
| Training seed | **0** for all cells |
| Epochs | **5** |
| Everything else | exactly `results/reference/train_args_c40_vid.json`. **No `--amp`.** |

### Why 5 epochs, decided now

c40 best-epoch across the existing campaign is **≤ 11 in every run, mean ≈ 5**
(`results/in_domain_c40_vid/`). Five epochs covers the region where c40 runs
actually peak. This is **not** a post-hoc budget cut: it is recorded here before
any C1 run exists, and it is the reason the probe costs ~1.2 h rather than ~3.6 h.

⚠️ If any cell's best epoch is **4** (the last), that cell was still improving and
**the cell is inconclusive**, not evidence of a low result. Report it as such. This
rule is fixed now precisely because the c23 campaign hit exactly this problem.

### What makes this a fair test

**Both arms are swept.** Tuning only FAD, or only Xception, would build the answer
into the design — the same asymmetry the paper criticises elsewhere.

## 3. The primary quantity

**FAD − Xception, frame-pooled test AUC, at each learning rate**, computed from
committed prediction dumps by the same code path as every other comparison.

## 4. Decision rule — fixed before seeing any result

| observation | conclusion |
|---|---|
| At **every** lr, FAD − Xception lies **inside** the frozen crossed interval [−0.0358, +0.0180] | The null is **not attributable to the learning-rate choice** within this range. The objection is answered. |
| At **any** lr, FAD − Xception **exceeds +0.0180** (the crossed upper endpoint) | The result **is** lr-sensitive. This must be reported prominently, and the case-study conclusion must be qualified as conditional on the reference lr. |
| Cells disagree in sign but all lie inside the crossed interval | Consistent with the resolution finding — the design cannot separate these. Report as further evidence for contribution 2, **not** as evidence about FAD. |
| Any cell's best epoch is the final epoch | That cell is **inconclusive** and is excluded from the above, with the exclusion stated. |

## 5. What this probe is NOT

- **Not a tuning exercise.** We are not selecting a better lr for either arm, and
  no result here licenses re-running the headline at a different lr.
- **Not powered.** One training seed per cell. A single run per cell cannot
  separate lr effects from training-run variability, which we measured at
  sd **0.0145** with the split frozen. **Any observed difference smaller than
  ~0.0145 is uninterpretable** and must be reported as such.
- **Not a replacement for the primary result**, which remains frozen regardless of
  what this returns.

## 6. Prohibited after seeing results

- Selecting a "best" lr for one arm and comparing it against the other at a
  different lr
- Extending the epoch budget for cells that look unfinished
- Adding lr values to fill in a pattern
- Re-running the frozen headline at any lr found here

Any of these converts a sensitivity probe into optional stopping.
