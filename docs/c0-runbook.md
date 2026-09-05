# C0 runbook — video-level splits (L2), the leakage fix

**Goal:** re-run the c40 headline comparison with **video-level** splits so no
video's crops appear in both train and test. Answers the question that currently
blocks the paper: does `f3net ≈ xception` survive once the ceiling from frame-level
leakage is removed? See `docs/validation-plan.md` C0 for the full argument.

~3.5 h if the c40 crops are reused (below), ~4.5 h if they must be re-extracted.
Accelerator **GPU T4 x2**, Internet **ON**, fresh single-purpose notebook.

---

## Step 0 — reuse the existing c40 crops (saves ~1 h)

Kaggle saves the whole of `/kaggle/working` as a version's output, so the
`ffpp_c40_crops/` built by the C7 run should still exist. Attach that notebook's
output as an input to the new notebook (+ Add Input → Notebook Output → the c40
run), then locate the crops:

```bash
!find /kaggle/input -maxdepth 5 -type d -name "ffpp_c40_crops" 2>/dev/null; echo "---"; find /kaggle/input -path "*ffpp_c40_crops*" -name "*.jpg" | wc -l
```

Expect ~30,000. **If the crops are not there**, fall back to re-running cells 3–4
of `docs/c40-runbook.md` to rebuild them before continuing.

## Step 1 — clone

```bash
!cd /kaggle/working && rm -rf ./-DeepTrace && git clone -q https://github.com/Charles-AM/-DeepTrace.git ./-DeepTrace && cd ./-DeepTrace && git log --oneline -1
```

## Step 2 — video-level run

⚠️ Two things that will silently ruin this if you deviate:
1. **`--dataset-name` must be new** (`ffpp_c40_vid`). Manifests are cached by that
   name — reusing `ffpp_c40` would silently load the old *frame-level* split and
   you'd learn nothing.
2. **`--extra` must be last** on the command line (it is `argparse.REMAINDER`, so
   it swallows everything after it).

Substitute `<CROPS>` with the path found in Step 0.

```bash
!cd /kaggle/working/-DeepTrace && python -m src.run_ablation --data-root <CROPS> --dataset-name ffpp_c40_vid --configs baseline_spatial xception f3net --seeds 0 1 2 --epochs 15 --image-size 128 --batch-size 64 --reference xception --extra --group-by 'videos-([0-9]+)'
```

## Step 3 — verify the split actually grouped (do NOT skip)

The whole point is the split changed. Confirm no video id appears in more than one
split before trusting any number:

```bash
%%writefile /kaggle/working/-DeepTrace/check_split.py
import csv, re, collections
m = "results/manifests/ffpp_c40_vid_seed0_sz128.csv"
byvid = collections.defaultdict(set)
for r in csv.DictReader(open(m)):
    v = re.search(r"videos-([0-9]+)", r["path"])
    if v: byvid[v.group(1)].add(r["split"])
bad = {k: s for k, s in byvid.items() if len(s) > 1}
print(f"videos total: {len(byvid)}")
print(f"videos spanning >1 split: {len(bad)}  <-- MUST be 0")
if bad: print(dict(list(bad.items())[:5]))
```
```bash
!cd /kaggle/working/-DeepTrace && python check_split.py
```

`videos spanning >1 split: 0` is the pass condition. Anything else means the regex
didn't match the filenames and the run is invalid.

## Step 4 — bundle

```bash
!cd /kaggle/working && cp -r ./-DeepTrace/results ./c40vid_results && mv c40vid_results/ablation_table.md c40vid_results/ablation_c40_vid.md && tar czf c40vid_metrics.tar.gz --exclude="*.pt" --exclude="*/tb/*" c40vid_results && du -h c40vid_metrics.tar.gz
```

Metrics-only bundle stays small enough to download quickly; the checkpoints remain
in the version output if needed later.

---

## What the result means

Compare `f3net − xception` at L2 against the L1 (frame-level) numbers in
`results/in_domain_c40/`:

| outcome | interpretation |
|---|---|
| Absolute AUCs drop toward the published range (~0.90–0.96) **and** `f3net − xception` stays non-significant | **The paper's claim holds.** The null was not a ceiling artifact. Strongest outcome. |
| AUCs drop **and** the gap opens significantly in F3-Net's favour | The L1 null *was* a ceiling artifact. Claim reframes to "frequency helps once the task is hard enough" — outcome 2 of the c40 tree, still publishable. |
| AUCs barely move | Frame-level leakage was not the ceiling driver; look elsewhere (task scope, epochs, subset size). |

Either way, the **L1 vs L2 delta is itself reportable** — "frame-level splits inflate
FF++ AUC by X points" (C0.3), useful to anyone benchmarking on this dataset.

## Still open after this

L2 groups by the *target* id only. FF++ manipulated videos are `<target>_<source>`,
so identity `982` can be the target of one video and the source of another and still
cross the split boundary — CADDM's identity leakage. **L3 (identity-level, via
connected components over the pair graph, or the official FF++ split JSONs)** remains
the fully rigorous fix. See `docs/validation-plan.md` C0.4.
