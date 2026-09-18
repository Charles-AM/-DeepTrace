# State of play — resume here

Last updated **2026-09-17**. Read this first; everything else is detail.

**Next session starts at `docs/EXPERIMENT-PLAN.md`.**

---

## 1. What the project is

An **evaluation-methodology study** of frequency-aware deepfake detection, using
F3-Net's FAD component as a controlled case study on FaceForensics++.

> Evaluation design changes both what deepfake detectors appear to achieve and how
> much evidential weight an architectural comparison can carry.

**FAD is the case study, not the subject. No method contribution is claimed.**
Framing is frozen in `docs/ESSENCE.md` §0; claim discipline is §11.

### Three contributions

1. **Protocol and estimand** choices change apparent performance and architectural
   contrasts
2. **Dependence-aware evaluation** changes test-content uncertainty and statistical
   verdicts
3. **Architecture-level conclusions** require propagating training-run variation in
   addition to test-content variation

---

## 2. The primary result

Quote `results/canonical.json` → `manuscript_sentence`. Do not paraphrase; a test
checks its figures against the data.

> On the fixed c40 split, the mean FAD − Xception difference across five training
> runs was **−0.0092 AUC**. A crossed bootstrap propagating training-run and
> source-target-component variation produced a 95% interval of
> **[−0.0358, +0.0180]**. The Monte Carlo uncertainty band for its upper endpoint
> was [0.0176, 0.0184], entirely above the +0.014 reference; **4.07%** of
> crossed-bootstrap replicates exceeded that reference. This fixed-split c40
> evaluation therefore neither demonstrated a FAD advantage nor excluded a gain of
> the published magnitude.

### The other headline numbers

| finding | value |
|---|---|
| Unit structure | 3,000 crops → 150 videos → 30 target groups → **29 components** |
| Protocol gap | ~8.5 pts c23, ~18 pts c40; difference-in-differences +9.5 to +10.4 |
| **V2 seen-video advantage** | **+0.1998 frame-pooled, CI [+0.1422, +0.2542]** — one model, matched disjoint-group sets. Also +0.1731 (video mean-logit) and +0.1778 (mean-probability); **all exclude zero** |
| **V2 estimand spread** | **0.0267 = 1.9× the reference effect** — aggregation choice alone |
| **Resolution calibration** | V2 effect is **3.6×** its half-width (excludes zero); FAD effect is **0.34×** its half-width (cannot resolve) |
| Three units, one estimate (−0.0196) | frame [−0.0325, −0.0079] **excludes zero**; video and component do not |
| Conditional vs crossed | both conditionals exclude +0.014; crossed does not |
| Resolution | component half-widths 0.037–0.062 = **2.6–4.5×** the effect |
| Calibration | >0.10 excludes zero 9/9; <0.03 does 0/8 |
| Compression | c40 intervals **1.5–2.5×** wider than c23 on hash-verified identical test content |
| Run-to-run | range 0.034 with the split frozen = 2.4× the effect |
| Repeatability audit | 0.0336 movement, **n = 1**, no cause isolated |
| Cost | FAD ≈ **+3%** (GFLOPs +2.70%, latency +3.19%; params only +0.24% — say which) |

### External comparison

| source | Xception | +FAD | difference | uncertainty |
|---|---|---|---|---|
| F3-Net (ECCV 2020) | 0.8930 | 0.9070 | +0.0140 | **none** |
| DeepfakeBench (NeurIPS 2023) | 0.8261 | 0.8271 | +0.0010 | **none** |
| ours (5 runs) | 0.8314 | 0.8223 | −0.0092 | [−0.0358, +0.0180] |

Two published measurements differ by 14×; our interval contains both and zero.
**Neither published paper reports any measure of variability anywhere** —
established by full-text search, see `docs/VERIFICATION-LEDGER.md` §4.

---

## 3. Status

**Experimental programme closed.** Frozen at tag `results-frozen-v2`.

| | |
|---|---|
| training runs | 81 (+10 discarded AMP, marked) |
| committed predictions | 66,000 across 32 dumps |
| analysis directories | 15, all with READMEs |
| result CSVs | 74 |
| tests | ✅ **suite passes clean on Kaggle, 2026-09-17** — `pytest` exit code **0**, 2 skipped, ~257 tests. Locally only **107** run (the rest need torch) — ledger §21 |
| citations verified | **15 of 22** in the previously-unverified section |
| verification ledger | **16 sections** — every external claim traced to a quote |
| contribution 1 | **dissected and evidence-mapped** (2026-09-17); wording settled |
| contribution 2 | **positioning settled, dissection NOT started** — no evidence map yet |
| contribution 3 | dissected earlier (the ablation); result frozen |
| prespecifications | 2, committed before the analyses they govern |
| tags | 3, immutable (`results/PROVENANCE.md`) |
| manuscript | **not started** |

---

## 4. What to do next

**`docs/EXPERIMENT-PLAN.md`** is the execution view — where each experiment runs
(local bash / Kaggle GPU / web), what it changes, and the paper it answers to.
**`docs/REMAINING-WORK.md`** has the underlying reasoning.

**Free (no quota), all runnable locally — numpy and scipy are present:**
A2 verify the **remaining 7** citations (~1 h, **blocks submission**; was ~20
before the 2026-09-15 pass) · A1 coverage simulation (~1.5 h, **narrows** limitation 4 — it cannot retire it,
and Cameron et al. put 29 clusters inside the documented over-rejection regime) ·
A3 three figures (~1 h, needs a matplotlib venv) · A4 per-manipulation intervals ·
A5 more resolution curves · A6 BCa sensitivity

**Cheap GPU:** B1 `band_ablation` smoke test (~5 min, closes G3) ·
**B2 = V2 seen/unseen (~80 min — the best GPU spend, it changes what you can claim)**

**Moderate GPU:** C1 lr sweep (~1.2 h at 5 epochs) · C2 repeat audit (~2.4 h) ·
**C3 c23 fixed-split campaign (~6 h — highest ceiling, could remove limitation 3)** ·
C4 L3 component-disjoint (~3.6 h)

**Deliberately not doing:** band ablation beyond a smoke test, robustness,
cross-dataset, late fusion, capacity control, α-sweep. All old-framing.

⚠️ Everything in section C needs a prespecification **committed first**.

---

## 5. Traps that have already cost time

- **No `--amp`.** `run_ablation.py` never passed it, so every comparison run is
  fp32. It cost a 155-minute run once. Commands now build from
  `results/reference/train_args_c40_vid.json`
- **Kaggle outputs vanish.** The V1 dumps were lost and recovered only from a
  `~/Downloads` tarball. Commit prediction CSVs every time
- **Save & Run All is container-isolated** — its `/kaggle/working` is invisible to
  an interactive session. Results come from the version output
- **DO NOT CITE**: `analysis/late_fusion/`, `analysis/cka/`, `analysis/permanip/`
  (L1 checkpoints). `analysis/spectra/` is fine — it loads no checkpoint
- **pytest and sklearn are absent locally.** Local runs use a shim in the session
  scratchpad, which is **cleared between sessions**. Re-run the suite on Kaggle
  after touching tests
- **Adding runs after seeing a result is optional stopping.** Five runs was set by
  matching the existing campaign under budget — a reason that predates the result

---

## 6. Wording that is settled

| never write | write |
|---|---|
| "FAD provides no benefit" | "neither demonstrated nor excluded a gain of the published magnitude" |
| "three different conclusions" | "three uncertainty models, two conclusions" — video and component agree |
| "leakage" (for the L1→L2 gap) | **"protocol gap"** — V2 has not isolated the cause |
| "neither measured uncertainty" | **"neither reports"** — we can only see the page |
| "optimisation variability" | **"fixed-split training-run variability"** |
| "selectively powered" | "informative for large differences, insufficient resolution for FAD-sized ones" |
| "positive control" | **"large-separation calibration model"** |
| "we beat / outperform" | **nothing** — no performance claim is made anywhere |
| "three published measurements" | **two published, plus ours** |
| "leakage" (unqualified, in related work) | **"[L3.2] nonindependence"** — Kapoor & Narayanan's own term |
| "no one has accounted for this" | Bouthillier et al. **name** correlated errors and set them aside; we fill that gap |

Full table: `ESSENCE.md` §11. Variance vocabulary: §0b. Limitations: §8a (six).

---

## 7. Key documents

| file | what |
|---|---|
| `results/canonical.json` | the primary result + approved sentence |
| `docs/ESSENCE.md` | frozen framing, contributions, limitations, recommendations |
| `docs/VERIFICATION-LEDGER.md` | what is verified, assumed, or owed |
| `docs/EXPERIMENT-PLAN.md` | **where each experiment runs, and its literature anchor** |
| `docs/contribution-1-evidence.md` | **contribution 1 indexed by claim** — draft from this, not from memory. 15 do-not-write entries |
| `docs/REMAINING-WORK.md` | every candidate experiment with compute time |
| `docs/related-work.md` | reference papers + the citations still unverified |
| `docs/REPRODUCIBILITY.md` | methods, hyperparameters, why no tuning |
| `docs/f3net-ablation-verified.md` | the +0.014, verified from source |
| `results/PROVENANCE.md` | tag history, prespecification timing |
| `results/analysis/crossed/STABILITY.md` | the high-replicate stability check |

**Reproduce every headline number, one command, CPU, ~3 minutes:**

```
python -m src.reproduce
```

11 checks — unit structure, the canonical crossed result, and the aggregation-space
contrast — each compared against a value recorded in the repository. It also prints
what **cannot** be reproduced from committed data and why, rather than skipping it.

Reproduce only the primary result:

```
python -m src.crossed_boot --a-glob 'results/predictions_v8/ffpp_c40_vid_xception_seed*_test.csv' --b-glob 'results/predictions_v8/ffpp_c40_vid_xception_fad_seed*_test.csv' --margins 0.014 --n-boot 50000 --boot-seed 0 --out-dir /tmp/check
```

---

## 8. Course deliverables

Sequence: **proposal → literature review → full report → paper**, written in
reverse from frozen results. User supplies the spec for each.

Decided: the project is presented **as it stands**, not as it evolved. No pivot
narrative.

Two writing skills are installed (`~/.claude/skills/`): `academic-paper` and
`academic-paper-reviewer`. Hooks deliberately **not** installed — the plugin's
PreToolUse guard would intercept every write and Bash call.

⚠️ Anything those skills generate must be reconciled against
`VERIFICATION-LEDGER.md` for citations, and `ESSENCE.md` §11 for claim discipline.

---

## 9. Where contribution 2 actually stands (2026-09-17)

**Positioning is settled. The dissection is not started.** Do not mistake one for
the other.

### Settled

| against | our boundary | ledger |
|---|---|---|
| **DeepfakeBench** | they standardise *which* estimate is computed; we examine *how uncertain* it is | §16 |
| **Bouthillier et al.** | they enumerate training-run variance under an explicit i.i.d. assumption and flag correlated errors in one sentence without modelling them; we handle the clustered case | §6 |
| **Hurlbert** | pseudoreplication — an **inference** error, distinct from contribution 1's **splitting** error | §10 |
| **Cameron, Gelbach & Miller** | over-rejection documented at 5–30 clusters; **we have 29**, so limitation 4 is a citation rather than an admission | §6 |

### Not done

- **No `docs/contribution-2-evidence.md`.** Contribution 1 has one; contribution 2
  does not. That file is what made contribution 1 safe to draft from
- **Its own numbers have not been re-verified from raw dumps** the way contribution
  1's were in §15 — the cluster bootstrap, the three-units result, the crossed
  interval and the resolution curves are all recorded but not independently
  recomputed in a verification pass
- **A1 (coverage simulation) has not run** — ~1.5 h CPU, free, and it is the
  experiment that **narrows** limitation 4 (it cannot retire it — see below)
- **The estimand-mismatch limitation sentence is owed** (§11)

### Contribution 2 — DONE 2026-09-17

Dissected, all numbers independently validated (ledger §17–19), evidence map
written, external review applied. `docs/contribution-2-evidence.md`.

---

## 10. Next session — contribution 3, then the tests

**Charlie's plan, 2026-09-17:** dissect contribution 3 in conversation first — the
way contributions 1 and 2 were done — and run the remaining tests alongside it.
Do **not** write `contribution-3-evidence.md` before that conversation; the map is
the output of the dissection, not a substitute for it.

### What contribution 3 already has

| | |
|---|---|
| result | **−0.0092, 95% CI [−0.0358, +0.0180]**, frozen in `results/canonical.json` |
| reproduction | ✅ **exact** — `src/crossed_boot.py`, documented command (ledger §18.1) |
| test suite | ✅ 18 passed |
| reference effect | +0.014, verified from source, LQ = c40 (ledger §2, E2 closed) |
| ablation reasoning | `docs/f3net-ablation-verified.md` — why +0.014 and not +0.040, LFS vs FAD, sub-additivity |
| cost accounting | FAD ≈ +3% compute |

### What the dissection still needs to settle

- Whether C3 is framed as a **case-study finding** rather than a contribution in
  the methodological sense (suggested in review, not yet decided)
- The **estimand sentence** in `canonical.json` — blocked on Charlie's approval
- The §11 **limitations sentence**: our reference effect may be measured at a
  different granularity than our estimate
- DeepfakeBench's +0.0010 as **between-study evidence only**, never a threshold
- A do-not-write table, which C3 does not yet have

### Tests to run alongside

All free and local (numpy, scipy and pytest are installed): A4 per-manipulation
intervals (~20 min) · A6 BCa (~30 min) · A5 more resolution curves (~30 min) ·
c23 replication of the aggregation contrast (~20 min) · A1 coverage simulation
(~1.5 h). A3 figures (~1 h) needs matplotlib added to the venv.

⚠️ All optional follow-up — the frozen result cannot change. A3 is the one with
real consequences for a course submission, since **no usable figures exist**.

⚠️ **A1 does not retire limitation 4** — a simulation measures coverage under a *chosen* data-generating process, not for the unknown real FF++ population. It **narrows** the limitation to a measured statement under stated assumptions.
