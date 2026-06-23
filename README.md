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
This is shorthand for retrieval variant C plus phase-domain fusion:
```
python3 run.py --data [DATASET] --retrieval_variant C --phase_fusion
```
4. Run the provided experiment scripts.
```
sh run_main.sh
sh run_main.sh -Phase
```

Retrieval variants:
* `--retrieval_variant A`: RAFT original shape top-k.
* `--retrieval_variant B`: shape top-k -> phase hard top-m.
* `--retrieval_variant C`: shape top-k -> shape+phase soft re-rank top-m.

Phase-aware retrieval parameters:
```
--topm 20 --phase_top_m 5 --phase_lambda 0.1 --phase_tau 2 --phase_period 24
```

Phase-domain fusion can be enabled independently:
```
--phase_fusion --period_list 24
```

Recommended ablations:
```
--retrieval_variant A --topm 20
--retrieval_variant B --topm 100 --phase_top_m 20 --phase_tau 2
--retrieval_variant B --topm 100 --phase_top_m 20 --phase_tau 2 --phase_fusion
--retrieval_variant C --topm 100 --phase_top_m 20 --phase_lambda 1.0 --phase_tau 2 --phase_fusion
```
