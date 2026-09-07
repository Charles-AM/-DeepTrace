# Spectral analysis (DCT)

⚠️ **Descriptive only.** Real-vs-fake DCT gap is small (d ≈ 0.15) and survives
JPEG-q30. Frame-pseudoreplicated: a video-averaged effect size is owed, and
c40/H.264 validation has not been done.

✅ **NOT affected by the L1-checkpoint problem (G7).** `src/spectra.py` takes a
`--manifest` and never calls `build_model` or loads `best.pt` — it characterises
the **data**, not a trained model, so which checkpoints existed is irrelevant to
it. Only checkpoint-reloading analyses (`../late_fusion/`, `../cka/`) are.

```
python -m src.spectra --manifest <manifest.csv> --out-dir results/analysis/spectra
```
