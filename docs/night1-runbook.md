# Night-1 runbook — B1, B2, C1, C2 (Save & Run All)

**Batch mode. Close the laptop.** Written for *Save Version → Save & Run All*.

| | |
|---|---|
| Accelerator | **GPU T4 ×2** (never P100 — `CUDA error: no kernel image available`) |
| Internet | **On** (clone) |
| Estimated | **~5.1 h** (session limit 12 h) |
| Driver | `docs/night1_run.py`, dry-run validated 2026-09-18 |
| Governed by | `docs/c1-lr-prespecification.md`, `docs/c2-repeat-prespecification.md` — both committed **before** this runs |

## Stages, in execution order

Ordered **cheapest-and-highest-value first**, so a session death loses least.

| stage | what | time | value |
|---|---|---|---|
| **B1** | `band_ablation` smoke test | ~5 min | closes G3, the last gap-register item |
| **B2** | **V2 seen vs unseen** | ~80 min | **the one that changes what we can claim** — converts "protocol gap" into a measured leakage effect |
| **C1** | lr sweep, 3 rates × 2 arms, 5 epochs | ~1.2 h | answers *"is your null just an untuned FAD?"* |
| **C2** | repeat audit, 2 in-session repeats × 2 arms, 15 epochs | ~2.4 h | takes repeatability from **n=1** to characterisable |

If the session dies after B2 you still have the most valuable result. That is the
reason for this order.

## Inputs to attach

**Required — one:** notebook output **`c40-run`** (under `charlesappiahmanu`),
which contains `ffpp_c40_crops/`. Add Input → Notebook Output → search `c40-run`.

The driver searches three paths and aborts with a clear message if it finds none.

## The cells

**Cell 1 — clone**

```
!git clone https://github.com/Charles-AM/-DeepTrace.git /kaggle/working/-DeepTrace
```

**Cell 2 — dry run first. Always.**

```
!cd /kaggle/working/-DeepTrace && python docs/night1_run.py --dry-run
```

Read the printed commands. Confirm: **no `--amp`**, lr values 1e-4/3e-4/1e-3,
epochs 5 for C1 and 15 for C2, crops path resolved to a real directory. If any of
that is wrong, stop — do not start the batch.

**Cell 3 — the batch**

```
!cd /kaggle/working/-DeepTrace && python docs/night1_run.py
```

To run a subset: `--stages B2,C1`.

**Cell 4 — package the outputs**

```
!cd /kaggle/working && tar czf night1_results.tar.gz night1/
!ls -la /kaggle/working/night1_results.tar.gz
```

## Guard rails built into the driver

- **Every training command is constructed from `results/reference/train_args_c40_vid.json`.**
  A reference parameter the script does not pass is a **FATAL error**, not a silent
  omission — the failure mode that cost a 155-minute run.
- **Explicit `--amp` check.** The command is inspected and the run aborts if the
  flag is present, regardless of what the reference says.
- **Stage independence.** Each stage records its return code and continues; one
  failure does not stop the rest.
- **`night1_summary.json`** records every stage's outcome.

## ⚠️ The retrieval trap — read this before starting

**Save & Run All is container-isolated.** Its `/kaggle/working` is **invisible** to
an interactive session. Results come from the **version output** — open the
completed version, go to Output, download `night1_results.tar.gz`.

This has already cost this project one set of dumps. **Do not assume the files will
be there when you reopen the notebook interactively.** They will not be.

## What to send back

1. The tail of the log (the `=== SUMMARY ===` block)
2. `night1_results.tar.gz`, or the `results/predictions/` CSVs inside it

Prediction CSVs get committed to the repo on arrival — outputs that live only on
Kaggle have been lost before.

## Known uncertainties in this runbook

Stated rather than discovered at 3 a.m.:

- **B1's `--runs` argument** is a best guess at `band_ablation.py`'s expected run
  naming. It has never been executed. It is a 5-minute smoke test that supports no
  claim — **if it fails, ignore it** and let the batch continue.
- **B2's training stage** depends on `src.seen_unseen` writing a manifest that
  `src.train` then resolves by `--dataset-name ffpp_c40_v2seen`. The manifest
  builder is tested (8 tests); the hand-off to training is not.
- Timings are estimates from comparable runs, not measurements of these exact
  configurations.

Nothing here can change `results/canonical.json`. All four stages are additive.
