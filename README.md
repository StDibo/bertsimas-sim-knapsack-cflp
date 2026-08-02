# Robust Knapsack — Bertsimas and Sim

This repository contains the software developed for a Bachelor's thesis
in Computer Science at the University of Florence.

The project provides reproducible generation of 0-1 Knapsack instances
and the construction of nominal and Bertsimas-Sim robust models using
the Gurobi Python API.

## Main modules

- `lpKnapsackGenerator.py`: instance generation, nominal model,
  robust model and LP export.
- `lpKnapsackExperiment.py`: systematic execution of nominal and
  robust experiments.
- `lpKnapsackAnalysis.py`: validation and analysis of experimental results.

## Requirements

- Python 3
- Gurobi Optimizer
- A valid Gurobi license
- pandas
- matplotlib

## Reproducibility

Each instance is identified by:

- number of items `n`;
- maximum relative deviation `delta`;
- capacity ratio `rcp`;
- pseudorandom seed.

The uncertainty budget `gamma_rel` does not affect instance generation.
The same instance can therefore be reused for different robustness levels.
