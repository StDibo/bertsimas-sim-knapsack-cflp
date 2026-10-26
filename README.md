# Reproducible Bertsimas-Sim Models for Knapsack and CFLP

This repository contains the software developed for a Bachelor's thesis in
Computer Science at the University of Florence.

The project applies the Bertsimas-Sim budgeted uncertainty framework to two
classical combinatorial optimization problems:

- the 0-1 Knapsack Problem;
- the single-source Capacitated Facility Location Problem (CFLP).

It provides reproducible instance generation, construction of nominal and
robust models using the Gurobi Python API, systematic experimental campaigns,
and analysis of the resulting data.

## Repository Structure

### Knapsack

- `knapsack/knapsack_generator.py`: reproducible instance generation, construction of nominal and Bertsimas–Sim robust models, and LP export.
- `knapsack/knapsack_experiment.py`: execution of the nominal and robust experimental grid and collection of solver results in CSV format.
- `knapsack/knapsack_analysis.py`: computation of the Price of Robustness, solution-similarity metrics, runtimes, time-limit statistics, and plots.

### Capacitated Facility Location

- `capacitated_facility_location/cfl_generator.py`: validation of explicit CFLP instances, loading of OR-Library benchmarks, extraction of subinstances, construction of nominal and Bertsimas–Sim robust models, and LP export.
- `capacitated_facility_location/cfl_experiment.py`: validation against published OR-Library optima and execution of the benchmark and scalability campaigns.
- `capacitated_facility_location/cfl_analysis.py`: analysis of the Price of Robustness, facility-opening decisions, capacity and opening-cost effects, runtimes, time limits, and scalability results.

## Requirements

- Python 3.10 or later;
- Gurobi Optimizer;
- a valid Gurobi license.

The required Python packages are listed in `requirements.txt` and can be
installed with:

```bash
python -m pip install -r requirements.txt
