# From Linear Models to ResNets in JAX

A compact JAX-based vision benchmarking project that compares model families and optimizers on **grayscale CIFAR-10**.

## What this repository contains

- End-to-end experiment notebook for training multiple image classifiers in JAX
- Script version of the notebook for easier code browsing
- Saved checkpoints for representative Adam runs
- Precomputed training and test curves
- Summary plots and a metrics table for quick inspection

## Models covered

- Linear classifier
- Shallow MLP
- Deep MLP
- Deep MLP with ReLU
- CNN
- CNN with dropout
- VGG-style CNN
- ResNet-style CNN

## Optimizers covered

- SGD
- SGD with momentum
- Adam

## Snapshot of results

The strongest run in this repo is **ResNet-style CNN with Adam**, reaching **71.21%** best test accuracy over **20 epochs** on grayscale CIFAR-10.

See `results/metrics_summary.csv` for the full comparison.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── vision_benchmark_jax.ipynb
├── src/
│   └── vision_benchmark_jax.py
├── results/
│   ├── metrics_summary.csv
│   ├── curves/
│   └── plots/
└── checkpoints/
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/vision_benchmark_jax.ipynb
```

For CPU-only setups, the JAX docs note that a typical installation is `pip install -U jax`; GPU installs use different extras depending on the platform.

## Notes

- The project intentionally keeps the original experiment flow lightweight and easy to inspect.
- This repo is positioned as an **empirical benchmark project**, not as a coursework dump.
- The included artifacts are limited to essential experiments and representative checkpoints so the repo stays clean.

## Possible next upgrades

- Add data augmentation and learning-rate schedules
- Track runs with Weights & Biases
- Add confusion matrices and per-class accuracy
- Refactor architectures into reusable modules
