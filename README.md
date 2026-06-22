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
3. Run Phase-RAFT by adding `-Phase`.
```
python3 run.py --data [DATASET] -Phase
```
4. Run the provided experiment scripts.
```
sh run_main.sh
sh run_main.sh -Phase
```

`--no-retrieval` can be combined with `-Phase` for the Phase-RAFT no-retrieval ablation.
