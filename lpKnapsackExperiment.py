"""Reproducible execution of nominal and robust Knapsack experiments."""

import argparse
import csv
from itertools import product
from pathlib import Path
from typing import Iterable

import gurobipy as gp
from gurobipy import GRB

from lpKnapsackGenerator import (
    KnapsackInstance,
    build_nominal_model,
    build_robust_model,
    generate_instance,
)


NS = [200, 500, 1000, 10000]
DELTAS = [0.10, 0.25, 0.50]
CAPACITY_RATIOS = [0.20, 0.50, 0.80]
GAMMA_RELS = [0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00]
SEEDS = [1, 2, 3, 4, 5]

TIME_LIMIT_SEC = 1800
SOLVER_THREADS = 4
SOLVER_SEED = 0
MIP_GAP = 1e-4

FIELDNAMES = [
    "instance_id",
    "model_type",
    "n",
    "delta",
    "capacity_ratio",
    "seed",
    "gamma_rel",
    "gamma",
    "objective_value",
    "runtime_sec",
    "status",
    "selected_items",
    "selected_count",
    "best_bound",
    "mip_gap",
    "node_count",
    "num_variables",
    "num_constraints",
    "solution_count",
]


def make_instance_id(
    n: int, delta: float, capacity_ratio: float, seed: int
) -> str:
    """Build a readable identifier from the reproducible parameters."""

    return f"n{n}_d{delta:.2f}_cr{capacity_ratio:.2f}_s{seed}"


def status_name(status: int) -> str:
    """Return a stable name for the relevant Gurobi statuses."""

    names = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.INTERRUPTED: "INTERRUPTED",
    }
    return names.get(status, f"STATUS_{status}")


def solve_model(
    model: gp.Model,
    instance: KnapsackInstance,
    model_type: str,
    gamma_rel: float,
    instance_id: str,
) -> dict[str, object]:
    """Optimize a model and collect its incumbent and solver statistics."""
    
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = TIME_LIMIT_SEC
    model.Params.Threads = SOLVER_THREADS
    model.Params.Seed = SOLVER_SEED
    model.Params.MIPGap = MIP_GAP
    model.optimize()

    solution_count = int(model.SolCount)
    has_solution = solution_count > 0
    selected = []
    if has_solution:
        selected = [
            j for j in range(instance.n) if model.getVarByName(f"x[{j}]").X > 0.5
        ]

    return {
        "instance_id": instance_id,
        "model_type": model_type,
        "n": instance.n,
        "delta": instance.delta,
        "capacity_ratio": instance.capacity_ratio,
        "seed": instance.seed,
        "gamma_rel": gamma_rel,
        "gamma": gamma_rel * instance.n,
        "objective_value": model.ObjVal if has_solution else "",
        "runtime_sec": model.Runtime,
        "status": status_name(model.Status),
        "selected_items": ";".join(str(j) for j in selected),
        "selected_count": len(selected) if has_solution else "",
        "best_bound": model.ObjBound,
        "mip_gap": model.MIPGap if has_solution else "",
        "node_count": model.NodeCount,
        "num_variables": model.NumVars,
        "num_constraints": model.NumConstrs,
        "solution_count": solution_count,
    }


def run_experiment(
    ns: Iterable[int],
    deltas: Iterable[float],
    capacity_ratios: Iterable[float],
    gamma_rels: Iterable[float],
    seeds: Iterable[int],
    output_path: Path,
) -> None:
    """Generate each instance once, then solve its nominal and robust models."""

    ns, deltas = list(ns), list(deltas)
    capacity_ratios = list(capacity_ratios)
    gamma_rels, seeds = list(gamma_rels), list(seeds)
    instance_count = len(ns) * len(deltas) * len(capacity_ratios) * len(seeds)
    model_count = instance_count * (1 + len(gamma_rels))

    print(
        "Grid: "
        f"n={ns}, delta={deltas}, capacity_ratio={capacity_ratios}, "
        f"gamma_rel={gamma_rels}, seed={seeds}"
    )
    print(f"Distinct instances: {instance_count}; total models: {model_count}")
    print("The complete experiment may take a long time.")

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()

        for n, delta, capacity_ratio, seed in product(
            ns, deltas, capacity_ratios, seeds
        ):
            instance = generate_instance(n, delta, capacity_ratio, seed)
            instance_id = make_instance_id(n, delta, capacity_ratio, seed)

            nominal = build_nominal_model(instance)
            writer.writerow(solve_model(nominal, instance, "nominal", 0.0, instance_id))
            csv_file.flush()
            nominal.dispose()

            for gamma_rel in gamma_rels:
                robust = build_robust_model(instance, gamma_rel)
                writer.writerow(
                    solve_model(robust, instance, "robust", gamma_rel, instance_id)
                )
                csv_file.flush()
                robust.dispose()

    print(f"Results saved to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="path of the new CSV file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or Path(__file__).with_name("knapsack_results.csv")
    run_experiment(
        NS,
        DELTAS,
        CAPACITY_RATIOS,
        GAMMA_RELS,
        SEEDS,
        output,
    )


if __name__ == "__main__":
    main()
