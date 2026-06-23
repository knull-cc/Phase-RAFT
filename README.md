# Retrieval-Augmented Forecasting of Time-series (RAFT)
This is the official PyTorch implementation of our paper ([Link](https://arxiv.org/abs/2505.04163)), which is accepted to ICML 2025. \
The code is build on the base of [Time-Series-Library](https://github.com/thuml/Time-Series-Library).


### Required Packages
* python == 3.9.13

Install dependencies with:
```
pip install -r requirements.txt
```

### Usage
1. Create ./data directory and place dataset files in ./data directory.
2. Run RAFT baseline.
```
python3 run.py --data [DATASET]
```
3. Run phase-aware RAFT by adding `-Phase`.
```
python3 run.py --data [DATASET] -Phase
```
This is shorthand for retrieval variant C plus conservative phase-domain residual fusion:
```
python3 run.py --data [DATASET] --retrieval_variant C --phase_fusion --phase_fusion_mode residual
```
4. Run the provided experiment scripts.
```
sh run_main.sh
sh run_main.sh -Phase
```
5. Run Phase-conditioned Residual Corrector (PRC) as a post-processing module.
```
sh run_main.sh -phase_block
```

Retrieval variants:
* `--retrieval_variant A`: RAFT original shape top-k.
* `--retrieval_variant B`: shape top-k -> phase hard top-m.
* `--retrieval_variant C`: shape top-k -> shape+phase soft re-rank top-m.

Phase-aware retrieval parameters:
```
--topm 20 --phase_top_m 5 --phase_lambda 0.1 --phase_tau 2 --phase_period 24
```

Phase-domain fusion can be enabled independently. `residual` is the conservative
add-on mode; `backbone` uses the phase-domain predictor as the main output head.
```
--phase_fusion --phase_fusion_mode residual --period_list 24 --phase_fusion_scale 0.1
--phase_fusion --phase_fusion_mode backbone --period_list 24
```

Recommended ablations:
```
--retrieval_variant A --topm 20
--retrieval_variant B --topm 100 --phase_top_m 20 --phase_tau 2
--retrieval_variant B --topm 100 --phase_top_m 20 --phase_tau 2 -phase_block
--retrieval_variant B --topm 100 --phase_top_m 20 --phase_tau 2 --phase_fusion --phase_fusion_scale 0.05
--retrieval_variant C --topm 100 --phase_top_m 20 --phase_lambda 1.0 --phase_tau 2 --phase_fusion --phase_fusion_scale 0.05
--retrieval_variant B --topm 100 --phase_top_m 20 --phase_tau 2 --phase_fusion --phase_fusion_mode backbone
--retrieval_variant C --topm 100 --phase_top_m 20 --phase_lambda 1.0 --phase_tau 2 --phase_fusion --phase_fusion_mode backbone
```

PhaseBlock/PRC builds a full residual datastore before test. In the default
`full` bank mode, every memory window contributes one phase-aligned local
TimeBlock key per forecast step, and the value is that step's residual. By
default it uses validation residuals only, so test never retrieves from test
and the residual bank is not produced by sweeping the full training set. Use
`--phase_block_periods 24` to force a period, or leave it unset to estimate
periods by FFT:
```
-phase_block --phase_block_bank_mode full --phase_block_topk 5 --phase_block_alpha 0.2
```
