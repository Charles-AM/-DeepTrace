# Remaining work — full audit, 2026-09-15

Every gap found by sweeping the repo: unrun code, unvalidated claims, weak
findings, stale documentation, and experiments that would strengthen the paper.
Ordered by value per unit of compute, not by topic.

**The paper is publishable without any of this.** Results are frozen at
`results-frozen-v2`; nothing below can move a number in the canonical block. This
is about closing gaps a reviewer or marker could reasonably probe.

---

## A. Free — no GPU, no quota

| # | item | why | cost |
|---|---|---|---|
| **A1** | **Bootstrap coverage simulation** | **Closes limitation 4.** We currently state that percentile-bootstrap coverage with 29 components is *unverified*. A simulation study settles it: estimate component- and run-level variance from our own data, simulate many datasets with a known true effect, run the crossed bootstrap on each, and measure how often the 95% interval actually contains it. | ~1.5 h CPU |
| **A2** | Verify ~20 citations in Groups A/G/H | **Blocks submission.** Written from memory, never checked — including the dataset paper and the backbone | ~1 h |
| **A3** | Three publication figures | Needed for every document. Two-panel forest plot, protocol gap, resolution curve | ~1 h |
| **A4** | Per-manipulation cluster-aware intervals | `permanip_l2` is currently descriptive only, with no uncertainty. Computable from committed dumps | ~20 min |
| **A5** | V7 resolution curves for the remaining seeds and at c23 component unit | Currently c40 seed-0 only at the component unit; c23 video only | ~30 min |
| **A6** | BCa interval sensitivity | Percentile intervals can be biased with few clusters. A BCa comparison at full n would show whether the choice matters | ~30 min |
| **A7** | Document the c23 epoch-budget caveat | **Found during this audit, not recorded anywhere.** At c23, `f3net`, `frequency_only` and `full` each have runs peaking at epoch 13–14 of 15 — they may still have been improving. c40 is unaffected (all peak by epoch 11, mean ~5) | 10 min |

### A1 in more detail — the highest-value free item

Limitation 4 currently says: *"the high-replicate analysis addresses Monte Carlo
error in the estimated endpoints, not the frequentist coverage of the bootstrap
procedure itself. Numerical stability is demonstrated; small-sample coverage
remains unknown."*

A simulation converts "unknown" into a measured number. If coverage is ~95%, the
limitation shrinks to a sentence. If it is 88%, that is a **finding** — and one
that strengthens the paper, since under-coverage means the field's intervals are
even more optimistic than we showed.

Either outcome is reportable. Prespecify before running.

---

## B. Cheap GPU — under 2 hours

| # | item | why | cost |
|---|---|---|---|
| **B1** | `band_ablation.py` smoke test | **Last open item in the gap register (G3).** The script has never been executed. It supports no claim in the paper, but an unrun script in a repo that claims reproducibility is a bad look | ~5 min |
| **B2** | **V2 — seen vs unseen, matched** | Turns "protocol gap" into a **leakage measurement**. Removes a hedge from five places in ESSENCE including the headline. Code built and tested (`src/seen_unseen.py`, 8 tests) | ~80 min |

---

## C. Moderate GPU — 2 to 6 hours

| # | item | why | cost |
|---|---|---|---|
| **C1** | **Learning-rate sensitivity, both arms** | Defuses the strongest available objection to the case study: *"your null result may just be an untuned FAD."* FAD changes the input from 3 to 9 channels, so a different lr is plausible. Must sweep **both** arms or it recreates the asymmetry we avoid | ~3.6 h (15 epochs) or **~1.2 h at 5 epochs** — defensible because c40 best-epoch is ~5 |
| **C2** | Repeat audit, 2 in-session repeats | Takes the reproducibility finding from n=1 to something characterisable, and separates run-level from session-level. Code built and dry-run (`docs/repeat_audit.py`) | ~2.4 h |
| **C3** | **c23 fixed-split campaign** | **Removes limitation 3.** A crossed interval currently exists only at c40, because c23 runs used varying splits. Ten runs at c23 with `--split-seed 0` would make the crossed analysis possible at both compressions — and c23 has *better* resolution (half-widths 0.015–0.029 vs 0.037–0.062), so it might resolve +0.014 where c40 cannot | ~6 h |
| **C4** | L3 component-disjoint splits | A third protocol rung. Our subset is partner-closed (150 components of size 2, `video_groups_per_component = 2.0`), so ~93% of test targets currently have their partner in training — a real leakage channel, and the one CADDM is about | ~3.6 h |

### C3 is the most interesting of these

If the c23 crossed interval **excludes** +0.014 while c40 does not, that is a much
sharper statement than either alone: *the same comparison is resolvable at high
quality and not at low quality* — which is precisely the regime where the original
claim was made. That would be the strongest single result in the paper.

It is also the highest-risk: it could equally come back inconclusive.

---

## D. Deliberately not doing — old framing

These were designed when the project was a method paper. Under the evaluation
framing they add space and disconnected hypotheses without supporting any of the
three contributions.

| item | why not |
|---|---|
| C4/C4b full band ablation (spectral bias) | A method question. Engages FreqDebias, but as a claim about frequency behaviour, not about evaluation |
| C5 robustness under perturbation | Method question; existing results are pre-protocol-fix |
| C6 cross-dataset transfer | Method question; our cross-dataset work is superseded and used a non-official sample |
| C1b late fusion, C1c capacity control | **Removed from the paper** (ESSENCE §12) |
| α-sweep | Gate behaviour — a method question |
| Spectral survival under real H.264 | Method question |

---

## E. Documentation and verification gaps

| # | item | status |
|---|---|---|
| **E1** | ~20 citations in Groups A/G/H unverified | **blocks submission** |
| E2 | F3-Net "LQ" = c40 | inference from FF++ convention; they never use the c-notation |
| E3 | DeepfakeBench F3Net scope | paper says FAD+LFS, code says FAD only — recorded as ambiguous |
| E4 | Re-verify Groups B–F | last checked 2026-09-03/04 |
| E5 | c23 epoch-budget caveat | **newly found, not yet recorded** (see A7) |
| E6 | Bouthillier: do they include test-content clustering? | changes how we word contribution 3's novelty |
| E7 | Kapoor & Narayanan: which leakage category? | use their taxonomy term |
| E8 | DO-NOT-CITE directories still present | `analysis/cka`, `analysis/permanip`, `analysis/late_fusion` — marked in place, intentionally retained |

---

## F. Suggested order

**If you have no quota:** A7 → A2 → A3 → A1. That closes the submission blocker,
produces the figures, and converts limitation 4 from unknown to measured.

**If you have ~2 hours of GPU:** add B1 and B2. V2 is the single best GPU spend —
it changes what you can *claim*.

**If you have ~6 hours:** add C1 at the 5-epoch setting (~1.2 h) and C3 (~6 h is
too much alongside; pick one). C3 has the higher ceiling.

**Everything in C requires a prespecification committed first**, per the standing
practice — especially C1 and C3, where the result could change a headline.
