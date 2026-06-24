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
Value = future residual after phase-aligned anchoring
```

In code, the memory stores the phase-anchored future residual. For each future
horizon `h`, the anchor is the latest observed point whose phase matches that
future timestamp:

```
Value[h] = future[h] - observed_same_phase_anchor[h]
```

At prediction time, the current input is converted into the same IdeaBlock
query, the model retrieves similar historical keys, aggregates their anchored
future residuals, adds the current sample's same-phase anchor back, and fuses
the retrieved future with the lookback prediction through a residual gate.

## Components

1. **Phase Alignment**: regroup observed time steps by absolute phase under
   period `P`.
2. **IdeaBlock Construction**: build `Phase-aligned IdeaBlock(p, r)` from
   phases around `p` over previous cycles.
3. **Key-Value Memory**: store IdeaBlock keys from the training set and their
   phase-anchored future residual values.
4. **Future Retrieval Fusion**: retrieve anchored futures, then blend them with
   the backbone forecast through a trainable residual gate.

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
  --value-anchor phase \
  --fusion-mode linear
```

Core retrieval parameters:

```
--period_len P          # cycle length used for phase alignment
--idea_block_radius r   # local phase neighborhood radius
--idea_block_cycles N   # number of previous cycles in each key
--topm K                # retrieved neighbors
--temperature T         # softmax temperature for retrieved future aggregation
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
