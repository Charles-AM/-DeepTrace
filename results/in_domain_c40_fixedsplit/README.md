# V8 attempt 1 — INVALID, superseded (2026-09-06)

Kept as a record, not as evidence. **Do not cite these numbers.**

10 runs, c40, `--split-seed 0`, training seeds 0–4, 15 epochs, 155 min. All
trained, none failed, and the split verified against
`results/reference/ffpp_c40_vid_seed0_test_split.json`.

## Why it is invalid

The driver passed `--amp`; `src/run_ablation.py`, which produced every
varying-split c40 run, never does. So the comparison arm is fp32 and this arm is
mixed precision. Every other setting matches — lr 3e-4, batch 64, weight decay
0.05, 15 epochs, image 128, 2 workers, and the `no_decay_param_groups` fix
predates both (`7bb4e11` is an ancestor of `e68b559`).

V8 exists so its sd can be **subtracted** from the varying-split sd. That
subtraction requires an identical training procedure. Precision mode is part of
the procedure, so as it stands the difference confounds precision with split
policy and the decomposition cannot be computed.

## What it produced

FAD − Xception, frame-pooled test AUC:

| seed | varying split | this run (fixed split, AMP) |
|---|---|---|
| 0 | −0.0196 | **+0.0395** |
| 1 | −0.0101 | −0.0049 |
| 2 | +0.0142 | −0.0046 |
| 3 | +0.0253 | −0.0096 |
| 4 | +0.0145 | −0.0114 |
| **sd** | **0.0188** | **0.0213** |

The naive subtraction gives a *negative* variance component (−0.000098), which is
not a possible value for a variance and is the expected symptom of comparing two
noisy sd estimates at n=5 across a confound.

## The seed-0 signal

Seed 0 is nominally the same configuration in both runs — same verified split,
same seed, same code, same epochs — yet:

| | Xception | + FAD | difference |
|---|---|---|---|
| varying-split run | 0.81347 | 0.79391 | −0.0196 |
| this run | 0.78158 | 0.82113 | **+0.0395** |

Individual AUCs move by ~0.03 and the paired difference flips sign and moves by
0.059. Mixed precision alone is a sufficient explanation — it changes the
optimisation trajectory from the first step — but it is not an isolated one here,
so this is a flag for the re-run, not a finding.

## The re-run

`--amp` removed from `docs/v8_run.py`. Expect ~3.5–4 h in fp32 rather than 2.6 h.

Seed 0 now serves as a built-in reproducibility control: it should return
Xception ≈ 0.81347 and +FAD ≈ 0.79391. If it does, cross-session environment
drift is negligible and the sd comparison is sound. If it does not, the
decomposition cannot be performed across sessions and that becomes the reportable
result.
