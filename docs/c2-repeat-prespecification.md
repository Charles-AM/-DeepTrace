# Prespecification — C2, repeat audit

**Written and committed before the runs.**

⚠️ **Timing.** The single cross-session movement this addresses is already
observed and reported (`results/in_domain_c40_fixedsplit/` §2: a nominally
identical seed-0 configuration moved **0.0336** and flipped sign). This
prespecification fixes what the follow-up may conclude, before it is run.

## 1. The question

The repeatability finding is currently **n = 1**. One nominally identical
configuration, re-run in a different session, produced a paired difference
0.0336 away from the original — **2.4× the reference effect** — with the two arms
moving in **opposite** directions.

We do not know whether that is:

- **(a)** ordinary run-to-run variation, which would place it alongside the
  measured fixed-split training-run sd of 0.0145; or
- **(b)** a session-level effect — something that differs between Kaggle sessions
  and is shared by runs within one.

n = 1 cannot distinguish these. This experiment adds **within-session** repeats so
the two can be compared.

## 2. Design — fixed before running

| | |
|---|---|
| Configuration | exactly the seed-0 cell: `xception` and `xception_fad`, `--split-seed 0`, training seed 0 |
| Repeats | **2 additional**, both arms, **in a single session** |
| Runs | 2 repeats × 2 configs = **4** |
| Epochs | **15** — the reference budget. Not reduced; this must match the runs it is compared against |
| Everything else | exactly `results/reference/train_args_c40_vid.json`. **No `--amp`.** |
| Split verification | abort on any divergence from `results/reference/ffpp_c40_vid_seed0_test_split.json` |

Driver: `docs/repeat_audit.py`, already written and dry-run.

## 3. The primary quantity

**The spread of FAD − Xception across repeats of a nominally identical
configuration**, separated into:

- **within-session spread** — across the 2 new repeats plus, if the session is the
  same, any run sharing it
- **across-session movement** — the existing 0.0336

## 4. Decision rule — fixed before seeing any result

| observation | conclusion |
|---|---|
| Within-session spread is **comparable to 0.0336** | The earlier movement is **consistent with ordinary run-to-run variation**. No session-level effect is indicated. The repeatability finding becomes a variance statement, not an anomaly. |
| Within-session spread is **much smaller than 0.0336** | A **session-level component is indicated**. ⚠️ Indicated, not established — see §5. |
| Within-session spread is **larger than 0.0336** | Run-to-run variability at this scale is larger than previously estimated; the fixed-split sd of 0.0145 understates it, and that must be reported. |

## 5. Hard limits on interpretation — fixed now

- **n remains small.** Three observations of one configuration is not an estimate
  of session-level variance and must never be presented as one.
- **No causal attribution, under any outcome.** cuDNN kernel selection, hardware
  allocation, driver and library versions are all candidates. **None is tested by
  this design.** The empirical finding does not require an explanation, and
  offering an untested one would be a claim we cannot support. This rule stands
  even if the result looks like an obvious explanation.
- **This is an audit, not an estimate.** The vocabulary is fixed:
  "reproducibility audit", never "session variance component".
- **The frozen result does not change** whatever this returns. V8's canonical
  interval is computed from its own five runs; these repeats are not added to it.

## 6. Prohibited after seeing results

- Adding further repeats because the first two looked interesting
- Attributing the movement to any specific cause
- Pooling these repeats into the V8 five-run set
- Re-describing the existing 0.0336 as an outlier on the basis of the new runs
