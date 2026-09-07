# Efficiency profiling

Closes **G1**: this was the one hand-transcribed artifact in the repo; it is now
generated to file.

`params_flops_latency.csv` — params (M), GFLOPs, batch-32 latency (ms) for five
configurations. Separate frequency branch costs +31% FLOPs / +44% latency; FAD is
≈ +3%.

Produced by `eff_table.py` (fvcore `FlopCountAnalysis`, 50 timed iterations after
5 warmup, CUDA-synchronised) in the V1 Kaggle notebook — it needs a GPU, so it is
not a `src/` module.
