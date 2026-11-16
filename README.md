# Reproducible Bertsimas–Sim Models for Knapsack and CFLP

This repository contains the software developed for a Bachelor's thesis in Computer Science at the **University of Florence**.

The project applies the **Bertsimas–Sim budgeted uncertainty framework** to two classical combinatorial optimization problems:
* The **0-1 Knapsack Problem**
* The **single-source Capacitated Facility Location Problem (CFLP)**

It provides reproducible instance generation, construction of nominal and robust models using the Gurobi Python API, systematic experimental campaigns, and analysis of the resulting data.

---

## Repository Structure

### Knapsack
* `knapsack/knapsack_generator.py`: Reproducible instance generation, construction of nominal and Bertsimas–Sim robust models, and LP export.
* `knapsack/knapsack_experiment.py`: Execution of the nominal and robust experimental grid and collection of solver results in CSV format.
* `knapsack/knapsack_analysis.py`: Computation of the Price of Robustness, solution-similarity metrics, runtimes, time-limit statistics, and plots.

### Capacitated Facility Location
* `capacitated_facility_location/cfl_generator.py`: Validation of explicit CFLP instances, loading of OR-Library benchmarks, extraction of subinstances, construction of nominal and Bertsimas–Sim robust models, and LP export.
* `capacitated_facility_location/cfl_experiment.py`: Validation against published OR-Library optima and execution of the benchmark and scalability campaigns.
* `capacitated_facility_location/cfl_analysis.py`: Analysis of the Price of Robustness, facility-opening decisions, capacity and opening-cost effects, runtimes, time limits, and scalability results.

---

## Requirements & Installation

* **Python** 3.10 or later
* **Gurobi Optimizer** with a valid Gurobi license (academic or commercial)

1. Clone the repository:
   ```bash
   git clone https://github.com<your-username>/<your-repo-name>.git
   cd <your-repo-name>
   ```

2. Install the required Python packages:
   ```bash
   python -m pip install -r requirements.txt
   ```

---

## Quick Start

To run the experimental campaigns and generate the analysis plots, execute the scripts from the root directory:

```bash
# Run Knapsack experiments and analysis
python knapsack/knapsack_experiment.py
python knapsack/knapsack_analysis.py

# Run CFLP experiments and analysis
python capacitated_facility_location/cfl_experiment.py
python capacitated_facility_location/cfl_analysis.py
```

---

## Thesis & License

* **Thesis Document:** *Link to the official PDF on the University of Florence archive will be added here upon publication.*
* **License:** The source code in this repository is licensed under the [MIT License](LICENSE).

