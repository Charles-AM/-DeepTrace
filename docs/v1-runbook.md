# V1 runbook — prediction dumps and cluster-aware intervals

Produces the paper's first valid inferential interval. Everything reported so far
uses **seed-level** intervals, which conflate optimisation noise with split
composition and ignore test-content clustering entirely — they are diagnostics,
not inference.

Also closes **G3** (`src/predict.py` has never been executed) and **G1** (the
efficiency CSV is the one hand-transcribed artifact).

~1 h. GPU T4 x2. Fresh single-purpose notebook.

---

## Inputs to attach

- the **c40 video-level** notebook output (checkpoints + `results/manifests/`)
- the **c23 video-level** notebook output
- `ffpp-crops-160` (c23 crops)
- the **c40-run** notebook output (c40 crops)

Crop paths must match what the manifests recorded, or `predict.py` will fail to
open images — the manifests store absolute paths.

## Cell 1 — clone

```bash
!cd /kaggle/working && rm -rf ./-DeepTrace && git clone -q https://github.com/Charles-AM/-DeepTrace.git ./-DeepTrace && cd ./-DeepTrace && git log --oneline -1
```

## Cell 2 — G3 smoke test (do this before the bulk dump)

`predict.py` has never run. Verify on one checkpoint before spending 30 minutes.

```bash
!cd /kaggle/working/-DeepTrace && python -m src.predict --run ffpp_c40_vid_xception_seed0 --results-root <C40VID_RESULTS> --dataset-name ffpp_c40_vid --seed 0 --image-size 128 --out-dir /kaggle/working/preds
```

Expect: `wrote .../ffpp_c40_vid_xception_seed0_test.csv (3000 predictions, 30 video groups)`.
**30 video groups is the number to check** — if it says 300, the manifest is
frame-level and the wrong results directory was passed.

Then sanity-check the file itself:
```bash
!head -3 /kaggle/working/preds/ffpp_c40_vid_xception_seed0_test.csv && python -c "
import csv,collections
r=list(csv.DictReader(open('/kaggle/working/preds/ffpp_c40_vid_xception_seed0_test.csv')))
print('rows', len(r))
print('labels', collections.Counter(x['label'] for x in r))
print('manipulations', collections.Counter(x['manipulation'] for x in r))
print('unparsed video_id', sum(1 for x in r if not x['video_id']))
"
```
`unparsed video_id` must be **0**. Anything else means the filename regex missed a
pattern and every downstream cluster is wrong.

## Cell 3 — dump predictions for the headline comparison

```bash
!cd /kaggle/working/-DeepTrace && for s in 0 1 2 3 4; do for c in xception f3net; do python -m src.predict --run ffpp_c40_vid_${c}_seed$s --results-root <C40VID_RESULTS> --dataset-name ffpp_c40_vid --seed $s --image-size 128 --out-dir /kaggle/working/preds; done; done
```
```bash
!cd /kaggle/working/-DeepTrace && for s in 0 1 2; do for c in xception f3net baseline_spatial frequency_only; do python -m src.predict --run ffpp_c23_vid_${c}_seed$s --results-root <C23VID_RESULTS> --dataset-name ffpp_c23_vid --seed $s --image-size 128 --out-dir /kaggle/working/preds; done; done
```

## Cell 4 — V1: cluster-aware paired intervals

Per seed, comparing the same test videos. `--margin 0.014` is the **verified**
published FAD gain (`docs/f3net-ablation-verified.md`).

```bash
!cd /kaggle/working/-DeepTrace && for s in 0 1 2 3 4; do echo "=== c40 seed $s ==="; python -m src.cluster_boot --a /kaggle/working/preds/ffpp_c40_vid_xception_seed${s}_test.csv --b /kaggle/working/preds/ffpp_c40_vid_f3net_seed${s}_test.csv --margin 0.014 --out-dir /kaggle/working/boot; done
```
```bash
!cd /kaggle/working/-DeepTrace && for s in 0 1 2; do echo "=== c23 seed $s ==="; python -m src.cluster_boot --a /kaggle/working/preds/ffpp_c23_vid_xception_seed${s}_test.csv --b /kaggle/working/preds/ffpp_c23_vid_f3net_seed${s}_test.csv --margin 0.014 --out-dir /kaggle/working/boot; done
```

Also run **frame-level** (no aggregation) on one seed — the naive-vs-clustered gap
*is* contribution 2:

```bash
!cd /kaggle/working/-DeepTrace && python -m src.cluster_boot --a /kaggle/working/preds/ffpp_c40_vid_xception_seed0_test.csv --b /kaggle/working/preds/ffpp_c40_vid_f3net_seed0_test.csv --frame-level --margin 0.014 --out-dir /kaggle/working/boot
```

**What to read:** the CI width at `unit=frame` vs `unit=video` vs `unit=identity`.
Expect frame to be narrowest (over-confident) and identity widest (~15 clusters —
the module warns below 10). That progression is the finding.

## Cell 5 — G1: regenerate the efficiency table to a file

```python
%%writefile /kaggle/working/-DeepTrace/eff_table.py
import torch, time, csv
from src.config import build_model
from fvcore.nn import FlopCountAnalysis
dev="cuda"; x=torch.randn(1,3,128,128,device=dev); rows=[]
for name in ["baseline_spatial","xception","f3net","full","frequency_only"]:
    m=build_model(name,image_size=128,pretrained=False).to(dev).eval()
    p=sum(t.numel() for t in m.parameters())/1e6
    with torch.no_grad(): f=FlopCountAnalysis(m,x).total()/1e9
    xb=torch.randn(32,3,128,128,device=dev)
    with torch.no_grad():
        for _ in range(5): m(xb)
        torch.cuda.synchronize(); t0=time.time()
        for _ in range(50): m(xb)
        torch.cuda.synchronize(); lat=(time.time()-t0)/50*1000
    rows.append({"config":name,"params_M":round(p,2),"gflops":round(f,2),
                 "batch32_latency_ms":round(lat,1)})
    print(rows[-1])
with open("/kaggle/working/efficiency.csv","w",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("wrote /kaggle/working/efficiency.csv")
```
```bash
!pip -q install --no-deps fvcore && pip -q install iopath && cd /kaggle/working/-DeepTrace && python eff_table.py
```

## Cell 6 — bundle

```bash
!cd /kaggle/working && tar czf v1_results.tar.gz preds boot efficiency.csv && du -h v1_results.tar.gz
```

---

## After

Send the bundle. Closes **G1**, **G3**, **G6**. Remaining after that: **G5**
(V8 fixed-split runs, ~2–3 h) and the open validation items (L3 identity splits,
video-level metric aggregation C0b, precision curve V7).
