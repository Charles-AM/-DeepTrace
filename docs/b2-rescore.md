# B2 recovery — re-score only, no retraining

The 2026-09-18 B2 run trained correctly (`train=23250 test=750`) but both
`src.predict` calls wrote to the same path: `predict.py` names its output
`<run>_<split>.csv`, identical for both since only the manifest differed. The
unseen dump overwrote the seen one.

**The checkpoint and both manifests are in that version's output.** Recovery is
inference only — roughly two minutes, no GPU training.

## Cells

New notebook. **Add Input → Notebook Output → the night-1 version** (the one
containing `night1/`). CPU is sufficient; GPU is faster.

```
!git clone https://github.com/Charles-AM/-DeepTrace.git /kaggle/working/-DeepTrace
```

```
!find /kaggle/input -name "best.pt" -path "*v2seen*"
!find /kaggle/input -name "ffpp_c40_v2*seed0_sz128.csv"
```

Note the directory holding `night1/` — call it `$NIGHT1`. Then, for each manifest:

```
!cd /kaggle/working/-DeepTrace && python -m src.predict \
    --run ffpp_c40_v2seen_xception_seed0 \
    --results-root $NIGHT1 \
    --dataset-name ffpp_c40_v2seen \
    --seed 0 --split-seed 0 --split test \
    --out-dir /kaggle/working/v2_predictions/seen
```

```
!cd /kaggle/working/-DeepTrace && python -m src.predict \
    --run ffpp_c40_v2seen_xception_seed0 \
    --results-root $NIGHT1 \
    --dataset-name ffpp_c40_v2unseen \
    --seed 0 --split-seed 0 --split test \
    --out-dir /kaggle/working/v2_predictions/unseen
```

⚠️ `--results-root` must be the directory that contains **both** `manifests/` and
`ffpp_c40_v2seen_xception_seed0/best.pt`. `predict.py` resolves the manifest from
`--dataset-name` and the checkpoint from `--run`, both relative to that root.

## Expected

Each call prints `wrote ... (750 predictions, ...)`. Two files, 750 rows each,
same model, matched test sets.

## Then

The AUC difference between them is the **seen-video advantage**, with
architecture, weights, training data, checkpoint-selection rule and test
composition all held constant. That is what converts "protocol gap" into a
measured leakage effect.
