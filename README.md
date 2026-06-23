# Phase-aligned IdeaBlock Retrieval

This repository keeps a single forecasting idea:

**Phase-aligned IdeaBlock Retrieval** replaces RAFT's continuous patch-level
Key-Value retrieval with phase-aligned block-level Key-Value retrieval.

For a period length `P`, each query uses the last observed phase as the center
phase `p` and a phase neighborhood radius `r`. The key is built by collecting
historical observations whose phases lie in `[p-r, p+r]` across previous cycles
and flattening that phase-aligned local block:

```
Key   = Phase-aligned IdeaBlock(p, r)
Value = the true future after the input window
```

At prediction time, the current input is converted into the same IdeaBlock
query, the model retrieves similar historical keys, aggregates their future
values, and fuses the retrieved future with the backbone prediction.

## Components

1. **Phase Alignment**: regroup observed time steps by absolute phase under
   period `P`.
2. **IdeaBlock Construction**: build `Phase-aligned IdeaBlock(p, r)` from
   phases around `p` over previous cycles.
3. **Key-Value Memory**: store IdeaBlock keys from the training set and their
   true future values.
4. **Future Retrieval Fusion**: retrieve and aggregate futures, then fuse them
   with the backbone forecast.

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
  --model RAFT \
  --features M \
  --seq_len 336 \
  --pred_len 96 \
  --enc_in 7 \
  --period_len 24 \
  --idea_block_radius 1 \
  --idea_block_cycles 4 \
  --topm 20
```

Core retrieval parameters:

```
--period_len P          # cycle length used for phase alignment
--idea_block_radius r   # local phase neighborhood radius
--idea_block_cycles N   # number of previous cycles in each key
--topm K                # retrieved neighbors
--temperature T         # softmax temperature for retrieved future aggregation
```

`-Phase` is accepted as a compatibility flag, but the RAFT model now always
uses Phase-aligned IdeaBlock Retrieval.
