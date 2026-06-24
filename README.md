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
Value = backbone forecast error on a similar historical block
```

The default training flow first warms up the backbone, then refreshes the
retrieval memory with the backbone's training-set residuals:

```
Value[h] = true_future[h] - backbone(input)[h]
```

At prediction time, the current input is converted into the same IdeaBlock
query, the model retrieves similar historical errors, aggregates them, and
adds the retrieved correction to the backbone forecast. The older direct-future
target is kept as an ablation through `--retrieval-target future`.

## Components

1. **Phase Alignment**: regroup observed time steps by absolute phase under
   period `P`.
2. **IdeaBlock Construction**: build `Phase-aligned IdeaBlock(p, r)` from
   phases around `p` over previous cycles.
3. **Key-Value Memory**: store IdeaBlock keys from the training set and their
   backbone error values.
4. **Error Retrieval Correction**: retrieve historical backbone errors and add
   them to the current backbone forecast.

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
  --enc_in 7 \
  --period_len 24 \
  --idea_block_radius 1 \
  --idea_block_cycles 4 \
  --topm 20 \
  --retrieval-target error \
  --backbone-warmup-epochs 5 \
  --refresh-memory-every 1
```

Core retrieval parameters:

```
--period_len P          # cycle length used for phase alignment
--idea_block_radius r   # local phase neighborhood radius
--idea_block_cycles N   # number of previous cycles in each key
--topm K                # retrieved neighbors
--temperature T         # softmax temperature for retrieved future aggregation
--retrieval-target error  # default: retrieve backbone residual errors
--backbone-warmup-epochs N # backbone-only epochs before building error memory
--refresh-memory-every N   # refresh error memory after warmup; 0 builds once only
--retrieval-target future  # ablation: retrieve direct future residuals
--value-anchor phase    # store Value as future minus same-phase historical anchor
--value-anchor last     # ablation: store Value as future minus last observed point
--fusion-mode linear    # baseline-initialized linear fusion over backbone/retrieval
--fusion-mode gate      # sigmoid-gated residual retrieval fusion
--fusion-mode none      # backbone-only ablation
--retrieval-gate-init X # initial gate logit when --fusion-mode gate
--horizon-wise-phase    # retrieve each horizon step from its own future phase
```

`-Phase` is accepted as a compatibility flag, but `PIBR` always uses
Phase-aligned IdeaBlock Retrieval.
