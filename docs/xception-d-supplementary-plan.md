# Plan — Xception + D supplementary campaign (NOT YET RUN)

**Written 2026-10-30, before any run.** Git history is the evidence.
Supersedes the full-UCF scope in `docs/ucf-supplementary-prespecification.md`,
whose isolation rules, fidelity checks and sanity band all still apply.

## Purpose

Test whether two of our evaluation findings hold for a **second architecture**, so
the conclusions are not specific to one frequency component.

FAD remains the primary case study throughout. This is supplementary. A failed or
unstable result stays supplementary and does not weaken the FAD findings.

## 🚨 What "D" actually is — read before scoping anything

From `training/detectors/ucf_detector.py`:

```python
self.encoder_f = self.build_backbone(config)   # fingerprint encoder
self.encoder_c = self.build_backbone(config)   # content encoder
self.con_gan   = Conditional_UNet()            # decoder
```

**Two full Xception backbones plus a U-Net decoder.** D is separable — the four
loss terms are `cls`, `spe`, `con`, `rec`, and their ablation maps D to the
disentanglement framework with `rec`, M to `spe`, C to `con`. So we keep `cls` and
`rec`, drop `spe` and `con`.

⚠️ **D is not a component in the sense FAD is.** FAD adds 0.24% to the parameter
count. D roughly doubles the backbone and adds a decoder.

❌ Do **not** describe these as two matched ablations. Anyone opening the file sees
otherwise.
✅ Describe them as **two case studies of different kinds**: a structurally tiny
change with a published increment of +0.014, and a structurally large one with a
published increment of +0.009. A change costing ~100× more parameters reports a
smaller gain — that contrast is worth stating on its own terms.

## Scope — one campaign, two experiments

Five seeds of **Xception + D** on the **fixed c40 video-disjoint split**
(`split_seed 0`), prediction dumps saved. Paired against the five Xception seeds
already committed on that exact split.

| experiment | covered | purpose |
|---|---|---|
| **3 — aggregation** | ✅ | does the frame-to-video scoring rule move the difference for a second architecture |
| **4 — units, conditional vs crossed** | ✅ | does the resampling unit still flip the verdict; does propagating both variance sources still change the conclusion |
| 1 — splitting | ❌ | needs a second campaign on the crop-randomised split; that side has the artifact-retention problem even for Xception |
| 2 — prior exposure | ❌ | needs the V2 manifests, uncommitted (recovery runbook gap 1) |

The two covered are the two strongest contributions. The two excluded are the two
least worth extending.

## What stays identical

Same split, same seeds, same recipe, same analysis code, same decision rule. The
only change is the second arm: `Xception+FAD − Xception` becomes
`Xception+D − Xception`, paired by seed in both cases.

That is what makes the two comparable **as analyses**, even though the
architectural changes are not comparable in size.

## Reference effect

UCF's Table 4 reports **Xception 0.986 → Xception + D 0.995**, a published
increment of **+0.009**.

⚠️ That is at **c23, near ceiling**, on full FF++. Ours runs at **c40** on 150
pairs at 128 px. ❌ **Never present it as a reproduction target** and never compare
our number against their 0.986.

## Cost

Two backbones plus a U-Net decoder per forward pass: roughly 2–3× Xception's
per-run cost. Five seeds lands near **10–15 GPU-hours**, not the ~5 estimated for
the full-UCF scope.

## Interpretation rule — fixed before running

| outcome | what we report |
|---|---|
| Both findings appear for D | The effects are **not architecture-specific**, and they survive a change of very different size. Report alongside the primary, labelled supplementary |
| They do not appear | **Architecture-dependence is a finding.** It bounds the generality claim and is reported as such |
| D fails to converge | Report plainly. A structurally large method failing at 128 px under inherited loss weights is a limitation of our study, not a hidden result |

❌ No outcome changes `canonical.json` or the primary FAD conclusion.

## Leading risk

The `rec` loss carries a weight set for UCF's batch size and resolution. At 128 px
with batch 64, a U-Net decoder reconstructing from half-dimension codes is the most
likely thing to fail.

**We will not tune it.** Tuning inherited hyperparameters is the one move this
paper cannot make. Run their published weight, report what happens, state it as a
limitation.

## Carried over from the UCF prespecification

Isolation (branch, new directories, `src/train_ucf.py` so `src/train.py` stays
byte-identical, additive `config.py`, empty `git diff` plus 16/16 before merging),
the known-answer fidelity test (our own Xception through the new entry point with D
disabled must return **0.81347**), the sanity band (≈0.5 broken, 0.75–0.90
plausible, ≈0.99 leak), and the run guards (dump every run, distinct `--out-dir`,
5 distinct md5s, verify the split, no `--amp`).
