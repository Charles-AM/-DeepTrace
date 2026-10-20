# Review comments, round 1 — and our responses

**Received 2026-10-20**, on the proposal presentation. Five comments. Two were
already satisfied, two are accepted, one is declined with reasons.

Recorded so the responses are consistent across the proposal, the slides and any
later conversation, and so a second round does not re-litigate settled ground.

---

## 1. Dataset access ✅ already satisfied

> *FaceForensics++ requires a request form before you get the download script.
> Confirm you have it now.*

We have it. The request form was completed, the download ran through
`src/ffpp_fastdl.py`, and 150 source-target pairs are extracted at both compression
levels, giving 30,000 crops each (`docs/REPRODUCIBILITY.md` §datasets).

**Action:** none. State it on the dataset slide if asked again.

---

## 2. Add a recent target ⛔ declined, with reasons

> *F3-Net (ECCV 2020) is fine as the source of the effect size, but add a recent
> target from the DeepfakeBench leaderboard, e.g. UCF (Yan et al., ICCV 2023). Even
> one run of it under your video-disjoint protocol would strengthen the story.*

The concern about currency is fair. The prescription does not work, for three
reasons, and the third is the decisive one.

**No matched pair.** Our design compares Xception against Xception with one
component added, everything else held identical. UCF differs end to end, so adding
it yields a benchmark comparison rather than an ablation, and the interpretability
the whole design buys is gone.

**No reference effect.** The +0.014 comes from F3-Net's own ablation table
(`docs/f3net-ablation-verified.md`). UCF has no published component-level increment
to anchor against, so the comparison would have nothing to be measured against.

**🚨 "Even one run" contradicts our own thesis.** Contribution 3 is that a single
training run cannot support a conclusion — both conditional analyses exclude
+0.014 and the crossed analysis does not. Offering a single run of a new
architecture as evidence would be doing on our slides exactly what we criticise in
other papers. The first question in the room becomes why that run should be trusted.

### The response: currency lives in the evidence, not the target

| paper | venue | role in our argument |
|---|---|---|
| **SBI** — Shiohara, Yamasaki | CVPR 2022 | reports **video-level** AUC, calls frame averaging "typical" |
| **DeepfakeBench** — Yan et al. | NeurIPS 2023 | frame vs video evaluation as an obstacle to fair comparison |
| **Kapoor & Narayanan** | *Patterns* 2023 | nonindependence between train and test as leakage |
| **UCF** — Yan et al. | ICCV 2023 | in our split audit: states **no partition at all** |
| **DF40** — Yan et al. | NeurIPS 2024 | checklist answer **"[No]"** to reporting error bars |
| **FreqDebias** — Kashiani et al. | CVPR 2025 | adopts **frame-level** AUC "following" four studies |

SBI and FreqDebias call **opposite** estimands the convention, three years apart,
at the same venue. That contradiction is stronger evidence that evaluation practice
is unsettled than any additional training run would be.

**The line to use:** the older papers are what we run; the recent ones are what we
read, and they are doing the arguing.

---

## 3. Name the roles ✅ accepted, with one correction

> *Xception is your Baseline (good). DeepfakeBench and Kapoor & Narayanan are
> effectively your Support papers. Say so explicitly.*

Accepted. One adjustment: **F3-Net is not a Target.** Calling it that implies we
are trying to beat it, and we are not. It is the source of the effect size.

| paper | role |
|---|---|
| Xception [CVPR 2017] | baseline architecture |
| F3-Net [ECCV 2020] | **source of the reference effect**, not a competitor |
| FaceForensics++ [ICCV 2019] | dataset |
| DeepfakeBench, Kapoor & Narayanan | supporting evidence |

---

## 4. Image-processing depth ✅ accepted — and we are better positioned than stated

> *Most of the work is statistics. Show the FAD frequency decomposition visually
> (DCT bands, what each band looks like on real vs fake crops).*

The perception is accurate: across the deck there is currently no image in which
anything is done to an image. One figure closes it.

Material already exists — `results/analysis/spectra/` holds DCT maps, a radial
power spectrum and a figure, none of which appears in the proposal.

**What to show:** one crop, an arrow, three images labelled low / middle / high
frequency, and a line saying the three are stacked into a nine-channel input. The
FAD slide currently *describes* this operation; the figure *performs* it.

⚠️ **Do NOT show real against fake side by side.** Our own spectral analysis
records the real-vs-fake DCT gap as **small, d ≈ 0.15, descriptive only**. Two
columns on a slide invite every viewer to hunt for a visible difference. They
either fail to find one, which undercuts us, or convince themselves they see one,
which is worse. Showing the decomposition is a method illustration and is safe;
showing a comparison implies a finding the data does not support.

**Action:** script to run on Kaggle (crops are not local), producing a four-panel
figure from one crop path.

---

## 5. Fix the number of seeds ✅ already satisfied

> *Fix the number of training seeds now (e.g. 5 per condition) so the GPU budget
> is concrete.*

Already fixed. **Five** per condition for the main c40 comparison, **three** at
c23, both architectures at every seed, recorded in `docs/REPRODUCIBILITY.md`.

**Action:** put the number on the compute slide so the question does not recur.

---

## Summary

| # | comment | status |
|---|---|---|
| 1 | dataset access | already satisfied |
| 2 | recent target | **declined** — contradicts contribution 3 |
| 3 | name the roles | accepted, F3-Net relabelled |
| 4 | image-processing depth | accepted, figure to build |
| 5 | fix seeds | already satisfied |

Two actions outstanding: the FAD decomposition figure, and putting the seed count
on the compute slide.
