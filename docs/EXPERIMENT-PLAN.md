# Experiment plan — what is left, who runs it, and why it counts

Companion to `docs/REMAINING-WORK.md`, which has the full reasoning. This file is
the **execution view**: where each experiment runs, what it would change, and the
literature it answers to.

Results are frozen at `results-frozen-v2`. **Nothing here can move a number in
`results/canonical.json`.** Every item closes a gap, strengthens a claim, or
retires a limitation.

Citation status is from the verification pass of **2026-09-15** — see
`docs/VERIFICATION-LEDGER.md` §6.

---

## Legend

| where | meaning |
|---|---|
| **local** | runs on this machine in bash. numpy 2.4.3 + scipy 1.17.1 present; no torch, no GPU |
| **Kaggle** | needs GPU quota. Charlie launches; notebook cells prepared in advance |
| **web** | literature verification, no compute |

---

## Tier A — free, no quota, no GPU

| # | experiment | where | what it changes | literature it answers to |
|---|---|---|---|---|
| **A2** | Verify remaining citations | web | **Unblocks submission.** 7 still unchecked after the 2026-09-15 pass | — (is the literature) |
| **A1** | Bootstrap coverage simulation | local | **Retires limitation 4.** Turns "coverage unknown at 29 clusters" into a measured number | **Cameron, Gelbach & Miller (2008)** put the over-rejection regime at **5–30 clusters — we have 29**. Also Field & Welsh (2007); Davison & Hinkley (1997) for coverage |
| **A3** | Three publication figures | local¹ | Forest plot, protocol gap, resolution curve. **None exist** — all 8 images in the repo are DO-NOT-CITE or superseded | — |
| **A4** | Per-manipulation cluster-aware intervals | local | `permanip_l2` is descriptive only; gives it uncertainty | Field & Welsh (2007), cluster bootstrap |
| **A5** | Resolution curves, remaining seeds + c23 component unit | local | Extends V7 beyond c40 seed-0 | Bouthillier et al. (2021): test-set variance is "well explained by the limited statistical power in the test set" — our curve measures that directly |
| **A6** | BCa sensitivity | local | Shows whether the percentile choice matters at n=29 | Efron & Tibshirani (1993); Cameron et al. (2008) |

¹ A3 needs matplotlib, which is absent. Install into a scratchpad venv (~100 MB of
3.8 GB free), or hand-write SVG. The venv keeps the script re-runnable.

**Order: A2 → A4 → A6 → A5 → A1 → A3.** A2 is the only submission blocker; A1 is
the long one and the only one that retires a limitation.

---

## Tier B — cheap GPU, under 2 hours

| # | experiment | where | what it changes | literature it answers to |
|---|---|---|---|---|
| **B1** | `band_ablation.py` smoke test | Kaggle, ~5 min | Closes **G3**, the last open item in the gap register. Supports no claim — but an unrun script contradicts the reproducibility posture | Pineau et al. (JMLR 22, 2021) — the standard we claim to meet |
| **B2** | **V2 — seen vs unseen, matched** | Kaggle, ~80 min | **The best GPU spend available.** Converts "protocol gap" into a measured leakage effect, removing a hedge from five places in ESSENCE including the headline. Code built and tested, never run | **Kapoor & Narayanan [L3.2] "Nonindependence between training and test samples"** — their own remedy is *block cross-validation*, which is what our L2/L3 splits are |

---

## Tier C — moderate GPU, 2 to 6 hours

⚠️ **Every item here needs a prespecification committed before the run**, per
standing practice. C1 and C3 could move a headline.

| # | experiment | where | what it changes | literature it answers to |
|---|---|---|---|---|
| **C1** | Learning-rate sensitivity, **both arms** | Kaggle, ~1.2 h at 5 epochs | Defuses the strongest objection available: *"your null is just an untuned FAD."* FAD takes the input from 3 to 9 channels, so a different lr is plausible | **Melis, Dyer & Blunsom (ICLR 2018)** — architectural gains dissolving under proper tuning is the exact failure mode. Bouthillier et al. treat hyperparameter choice as a first-class variance source |
| **C2** | Repeat audit, 2 in-session repeats | Kaggle, ~2.4 h | Takes the reproducibility finding from **n=1** to characterisable; separates run-level from session-level | Henderson et al. (AAAI 2018); Bouthillier et al. (2021) |
| **C3** | **c23 fixed-split campaign** | Kaggle, ~6 h | **Retires limitation 3**, and has the highest ceiling of anything left. c23 resolution is better (half-widths 0.015–0.029 vs 0.037–0.062), so it may resolve +0.014 where c40 cannot | F3-Net claims its *largest* gains on low quality. Testing where the claim is strongest is the sharpest available test |
| **C4** | L3 component-disjoint splits | Kaggle, ~3.6 h | A third protocol rung. ~93% of test targets currently have their partner in training | Kapoor & Narayanan [L3.2]; CADDM / implicit identity leakage (CVPR 2023) |

### Why C3 is the interesting one

If the c23 crossed interval **excludes** +0.014 while c40 does not, the statement
becomes: *the same comparison is resolvable at high quality and not at low
quality* — precisely the regime where the original claim was made. Strongest single
result available. Also the highest risk: it can come back inconclusive.

---

## Not doing — old framing

Band ablation beyond the smoke test, robustness under perturbation, cross-dataset
transfer, late fusion, capacity control, α-sweep, spectral survival under H.264.

All were designed when this was a method paper. Under the evaluation framing they
add space and disconnected hypotheses without serving any of the three
contributions. Reasoning in `REMAINING-WORK.md` §D.

---

## Standing traps

- **Never pass `--amp`.** `run_ablation.py` never did; all comparison runs are fp32.
  Cost a 155-minute run once. Build commands from
  `results/reference/train_args_c40_vid.json`
- **Commit prediction CSVs immediately.** Kaggle outputs vanish; the V1 dumps were
  recovered only from a `~/Downloads` tarball
- **Save & Run All is container-isolated** — its `/kaggle/working` is invisible to
  an interactive session
- **Adding runs after seeing a result is optional stopping.** Five was set by
  matching the existing campaign under budget, a reason that predates the result
- **pytest and sklearn are absent locally**; the shim lives in the session
  scratchpad and is cleared between sessions. Re-run the suite on Kaggle
