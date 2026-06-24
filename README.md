# Phase-aligned IdeaBlock Retrieval

This repository keeps a single forecasting idea:

**Phase-aligned IdeaBlock Retrieval** uses phase-aligned block-level Key-Value
retrieval for long-term forecasting.

For a period length `P`, each query uses the last observed phase as the center
phase `p` and a phase neighborhood radius `r`. The key is built by collecting
historical observations whose phases lie in `[p-r, p+r]` across previous cycles
and flattening that phase-aligned local block:

```
Key   = Phase-aligned IdeaBlock(p, r)
Value = true future residual after the input window
```

In code, the memory stores the offset-normalized future trend:

```
Value = future - last_observed
```

At prediction time, the current input is converted into the same IdeaBlock
query, the model retrieves similar historical keys, aggregates their future
trends, and fuses the retrieved trend with the lookback prediction head.

## Components

1. **Phase Alignment**: regroup observed time steps by absolute phase under
   period `P`.
2. **IdeaBlock Construction**: build `Phase-aligned IdeaBlock(p, r)` from
   phases around `p` over previous cycles.
3. **Key-Value Memory**: store IdeaBlock keys from the training set and their
   future residual values.
4. **Future Retrieval Fusion**: retrieve and aggregate future residuals, then
   fuse them with the backbone forecast.

## Usage

Install dependencies:

```
pip install -r requirements.txt
```

Run a dataset script:

```
sh run_main.sh
```

Run directly:

```
python3 run.py \
  --data ETTh1 \
  --root_path ./dataset/ETT-small \
  --data_path ETTh1.csv \
  --model_id ETTh1_336_96 \
  --model PIBR \
  --features M \
  --seq_len 336 \
  --pred_len 96 \
  --enc_in 7
```

PIBR baseline settings are fixed internally:

```
phase_radius = 1
idea_block_cycles = 4
topm = 20
temperature = 0.1
```

The period length `P` is inferred from the dataset or data file name.

`-Phase` is accepted as a compatibility flag, but `PIBR` always uses
Phase-aligned IdeaBlock Retrieval.
