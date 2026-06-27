# Phase-aligned IdeaBlock Retrieval

This repository studies one forecasting idea:

**PIBR wraps a host forecasting model with phase-aligned IdeaBlock retrieval.**
The default host is `Linear`, and stronger hosts such as `iTransformer` can be
selected from `run.py` with `--pibr_host`.

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
query. The model retrieves similar historical keys, aggregates their future
trends, and evaluates them with fixed hypothesis-test fusion modes rather than
tuned gates.

## Components

1. **Phase Alignment**: regroup observed time steps by absolute phase under
   period `P`.
2. **IdeaBlock Construction**: build `Phase-aligned IdeaBlock(p, r)` from
   phases around `p` over previous cycles.
3. **Key-Value Memory**: store IdeaBlock keys from the training set and their
   future residual values.
4. **Phase Residual Adapter**: retrieve and aggregate future residuals, then
   test whether the resulting phase forecast helps a host model.

## Models

- `Linear`: default host baseline without phase retrieval.
- `iTransformer`: stronger host baseline without phase retrieval.
- `PIBR`: phase-aligned IdeaBlock retrieval wrapper around `--pibr_host`.

## Usage

Install dependencies:

```
pip install -r requirements.txt
```

Run a dataset script:

```
sh run_main.sh
```

Run the first plugin comparison:

```
sh scripts/etth1_plugin.sh
```

Run the stronger periodicity hypothesis check:

```
sh scripts/pems03_pibr_hypothesis.sh
```

Run directly:

```
python3 run.py \
  --data ETTh1 \
  --root_path ./dataset/ETT-small \
  --data_path ETTh1.csv \
  --model_id ETTh1_336_96 \
  --model PIBR \
  --pibr_host iTransformer \
  --pibr_fusion phase_only \
  --pibr_projector identity \
  --features M \
  --seq_len 336 \
  --pred_len 96 \
  --enc_in 7
```

Compare the plugin against the same backbone without phase retrieval:

```
python3 run.py \
  --data ETTh1 \
  --root_path ./dataset/ETT-small \
  --data_path ETTh1.csv \
  --model_id ETTh1_336_96 \
  --model iTransformer \
  --features M \
  --seq_len 336 \
  --pred_len 96 \
  --enc_in 7
```

Default phase retrieval settings:

```
phase_radius = 1
idea_block_cycles = 4
topm = 20
temperature = 0.1
pibr_fusion = phase_only
pibr_projector = identity
```

The main hypothesis-test modes are:

- `phase_only`: prediction is only `last_observed + retrieved_future_residual`.
- `fixed_avg`: prediction is a fixed 50/50 average of host forecast and phase forecast.

`learned_gate` remains available for follow-up engineering, but it is not the
primary evidence for the PIBR hypothesis.

The period length `P` is inferred from the dataset or data file name unless
`--period_len` is passed explicitly.

`-Phase` is accepted as a compatibility flag and maps to the standalone `PIBR`
model for older scripts.
