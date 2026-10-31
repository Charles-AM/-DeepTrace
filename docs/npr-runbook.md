# NPR supplementary runbook — ten runs, counterbalanced, one session

Prespecification: `docs/npr-supplementary-prespecification.md`
Campaign constants: `results/reference/npr_campaign.json`
Rehearsal findings: `docs/npr-campaign-rehearsal.md`

| | |
|---|---|
| Accelerator | **GPU T4 ×2** (never P100 — no kernels in Kaggle's torch build) |
| Internet | **On** (clone) |
| Persistence | **No persistence** — see the warning below |
| Runs | 10 = 2 configs × 5 training seeds, c40, `--split-seed 0`, 15 epochs |
| Estimated | 4–6 h, ⚠️ **unverified** — see the cost note below |

## Cost — what is measured and what is not

⚠️ **No observed duration for V8 is recorded anywhere in this repository.** The
"~6 h" in `docs/v8-runbook.md` is that runbook's *Estimated* row, not a
measurement, and the per-run JSONs store `best_epoch` but no wall time.

The nearest **observed** figure is in `docs/EXPERIMENTS.md`: the c40 video-level
campaign of **nine runs took ~3.9 h**, i.e. roughly **26 min per run** at 15
epochs, or ~1.7 min per epoch. Runs within a session are sequential
(`docs/v8_run.py` loops `subprocess.run`); the "2 parallel" in that table refers
to two *concurrent Kaggle sessions* (`docs/overnight-plan.md`), not two GPUs
inside one session.

Extrapolated to ten sequential runs: **~4.3 h**, plus whatever NPR's two
interpolations per forward pass add. That is what `--time-one-epoch` measures.

❌ Do not quote 4.3 h as this campaign's cost. It is an extrapolation from a
different config, and the NPR overhead is exactly the unmeasured quantity.

## 🚨 Use a NEW notebook, and turn persistence OFF

`docs/kaggle-setup.md` recommends *Persistence: Files only* so `/kaggle/working`
survives between sessions. **For this campaign that setting is a hazard.**

The runner skips any run whose `best.pt` already exists. A checkpoint left in
`/kaggle/working` by an earlier session would be silently adopted into a campaign
it was not trained in — the exact cross-session mixing this study exists to avoid.
The runner defends against it by stamping such a run with a unique
`UNKNOWN-<run>` session id, so `src.campaign_gate` **fails** rather than accepting
it. That is the correct outcome, but it wastes the session.

A new notebook with persistence off starts with an empty `/kaggle/working`, so the
question does not arise.

## Input to attach

**One input:** Notebook Output → **`c40-run`** (under `charlesappiahmanu`). It
contains `ffpp_c40_crops/`, where every c40 result's crops live.

Optionally also attach the c40 video-level output for
`results/manifests/ffpp_c40_vid_seed0_sz128.csv`. Without it the script
regenerates the split deterministically. **Either way the split is verified
against `results/reference/ffpp_c40_vid_seed0_test_split.json` and the run aborts
on any difference.**

## Do NOT pip install

Per `docs/kaggle-setup.md`, do not run `pip install -r requirements.txt` on
Kaggle — reinstalling torch can pull a build that breaks the GPU. The base image
already has what training needs, and V8 trained successfully there, so
`tensorboard` is present. The dependencies added to `requirements.txt` in `aa67e18`
are for building a working environment *elsewhere*; they change nothing on Kaggle.

---

## Cell 1 — clone

```python
!cd /kaggle/working && rm -rf ./-DeepTrace && git clone -q https://github.com/Charles-AM/-DeepTrace.git ./-DeepTrace
!cd /kaggle/working/-DeepTrace && git log --oneline -1
```

Confirm the commit includes the NPR campaign work (`1703a72` or later).

## Cell 2 — the timing gate, FIRST, interactively

```python
!cd /kaggle/working/-DeepTrace && python docs/npr_run.py --time-one-epoch
```

Prints the measured per-epoch cost and the projected ten-run total. **Record that
number in the prespecification before launching the campaign.** One NPR epoch is
written to a separate `dataset_name` and `out_root`, so it cannot enter the
campaign's results.

⚠️ If the projection exceeds the session budget, stop and reduce scope. Do not
start a campaign the session cannot finish — a half-finished pair is excluded by
the session rule anyway.

## Cell 3 — the campaign (Save Version → Save & Run All)

```python
!cd /kaggle/working/-DeepTrace && python docs/npr_run.py
```

Runs both arms per seed in the committed counterbalanced order, dumps predictions
after each pair, writes the run manifest, and finishes by running
`src.campaign_gate`. It does **not** run the bootstrap.

## Cell 4 — package

```python
!cd /kaggle/working && tar czf npr_results.tar.gz results preds_npr && du -h npr_results.tar.gz
```

🚨 **Everything must land in `/kaggle/working`.** Per `docs/ARTIFACT-POLICY.md`,
Kaggle version outputs are not permanent, and the V1 notebook's prediction dumps
were lost this way. This cell is what makes the run recoverable.

## After it finishes

1. Check the gate verdict in the Cell 3 log. **If it failed, nothing may be
   estimated** — read which check failed and why.
2. Commit `preds_npr/runs.json` with the dumps, so the session record travels with
   the data.
3. Only then run the crossed bootstrap, pointing it at the campaign directory
   alone.

The baseline for the effect estimate is the **five new xception runs from this
campaign**. The committed V8 xception runs are used only for the repeatability
audit, which is five paired seed configurations across **one** pair of sessions —
never described as a session sample size of five.
