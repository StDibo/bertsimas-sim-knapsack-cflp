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

- `knapsack/knapsack_generator.py`: reproducible instance generation,
  construction of nominal and robust models, and LP export.
- `knapsack/knapsack_experiment.py`: execution of the nominal and robust
  experimental campaign.
- `knapsack/knapsack_analysis.py`: validation, aggregation, and visualization
  of the experimental results.

### Capacitated Facility Location

- `capacitated_facility_location/cfl_generator.py`: reproducible T4-type
  instance generation, construction of nominal and robust CFLP models, and LP
  export.
- `capacitated_facility_location/cfl_experiment.py`: execution of the
  experimental campaign and Monte Carlo evaluation of static solutions.
- `capacitated_facility_location/cfl_analysis.py`: validation, aggregation,
  and visualization of the CFLP results.

## Requirements

- Python 3.10 or later;
- Gurobi Optimizer;
- a valid Gurobi license.

The required Python packages are listed in `requirements.txt` and can be
installed with:

```bash
python -m pip install -r requirements.txt
