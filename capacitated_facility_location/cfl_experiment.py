"""Reproducible execution of nominal and robust CFLP experiments."""

import argparse
import csv
import random
from itertools import product
from pathlib import Path
from typing import Iterable, Sequence

import gurobipy as gp
from gurobipy import GRB

from cfl_generator import (
    CFLInstance,
    build_nominal_model,
    build_robust_model,
    generate_instance,
)


INSTANCE_SIZES = [(30, 50), (30, 70), (50, 70), (50, 100)]
GAMMAS = [1, 2, 3, 5, 7]
SEEDS = list(range(1, 11))

MONTE_CARLO_SCENARIOS = 5000

TIME_LIMIT_SEC = 1800
SOLVER_THREADS = 4
SOLVER_SEED = 0
MIP_GAP = 1e-4

FIELDNAMES = [
    "instance_id",
    "model_type",
    "facility_count",
    "customer_count",
    "seed",
    "gamma",
    "gamma_rel",
    "objective_value",
    "runtime_sec",
    "status",
    "open_facilities",
    "assignments",
    "empirical_infeasibility_rate",
    "simulation_scenarios",
    "simulation_seed",
    "best_bound",
    "mip_gap",
]


def make_instance_id(
    facility_count: int, customer_count: int, seed: int
) -> str:
    """Build a readable identifier from the reproducible parameters."""

    return f"m{facility_count}_n{customer_count}_s{seed}"


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


def simulation_seed(
    facility_count: int, customer_count: int, seed: int
) -> int:
    """Derive a separate reproducible seed for demand scenarios."""

    return 1_000_000 + facility_count * 10_000 + customer_count * 100 + seed


def generate_demand_scenarios(
    instance: CFLInstance,
    scenario_count: int,
    seed: int,
) -> tuple[tuple[float, ...], ...]:
    """Generate one-sided interval-bounded adverse-demand scenarios."""

    if scenario_count <= 0:
        raise ValueError("scenario_count must be greater than zero")

    rng = random.Random(seed)
    return tuple(
        tuple(
            instance.nominal_demands[j]
            + instance.maximum_deviations[j] * rng.random()
            for j in range(instance.customer_count)
        )
        for _ in range(scenario_count)
    )


def extract_solution(
    model: gp.Model, instance: CFLInstance
) -> tuple[list[int], list[int]]:
    """Extract open facilities and the assigned facility of each customer."""

    open_facilities = [
        i
        for i in range(instance.facility_count)
        if model.getVarByName(f"y[{i}]").X > 0.5
    ]
    assignments: list[int] = []
    for j in range(instance.customer_count):
        assigned = next(
            (
                i
                for i in range(instance.facility_count)
                if model.getVarByName(f"x[{i},{j}]").X > 0.5
            ),
            None,
        )
        if assigned is None:
            raise ValueError(f"customer {j} has no facility in the incumbent")
        assignments.append(assigned)
    return open_facilities, assignments


def empirical_infeasibility_rate(
    instance: CFLInstance,
    assignments: Sequence[int],
    scenarios: Sequence[Sequence[float]],
) -> float:
    """Return the fraction of scenarios violating at least one capacity."""

    if len(assignments) != instance.customer_count:
        raise ValueError("assignments must contain one facility per customer")
    if not scenarios:
        raise ValueError("at least one demand scenario is required")

    infeasible_scenarios = 0
    for demands in scenarios:
        if len(demands) != instance.customer_count:
            raise ValueError("each scenario must contain one demand per customer")
        loads = [0.0] * instance.facility_count
        for j, facility in enumerate(assignments):
            if not 0 <= facility < instance.facility_count:
                raise ValueError("assignment contains an invalid facility index")
            loads[facility] += demands[j]
        if any(
            load > instance.capacities[i] + 1e-9
            for i, load in enumerate(loads)
        ):
            infeasible_scenarios += 1

    return infeasible_scenarios / len(scenarios)


def solve_model(
    model: gp.Model,
    instance: CFLInstance,
    model_type: str,
    gamma: float,
    instance_id: str,
    scenarios: Sequence[Sequence[float]],
    scenario_seed: int,
) -> dict[str, object]:
    """Optimize a model and collect its solution and solver statistics."""

    model.Params.OutputFlag = 0
    model.Params.TimeLimit = TIME_LIMIT_SEC
    model.Params.Threads = SOLVER_THREADS
    model.Params.Seed = SOLVER_SEED
    model.Params.MIPGap = MIP_GAP
    model.optimize()

    has_solution = model.SolCount > 0
    open_facilities: list[int] = []
    assignments: list[int] = []
    infeasibility_rate: float | str = ""
    if has_solution:
        open_facilities, assignments = extract_solution(model, instance)
        infeasibility_rate = empirical_infeasibility_rate(
            instance, assignments, scenarios
        )

    return {
        "instance_id": instance_id,
        "model_type": model_type,
        "facility_count": instance.facility_count,
        "customer_count": instance.customer_count,
        "seed": instance.seed,
        "gamma": gamma,
        "gamma_rel": gamma / instance.customer_count,
        "objective_value": model.ObjVal if has_solution else "",
        "runtime_sec": model.Runtime,
        "status": status_name(model.Status),
        "open_facilities": ";".join(str(i) for i in open_facilities),
        "assignments": ";".join(
            f"{j}:{facility}" for j, facility in enumerate(assignments)
        ),
        "empirical_infeasibility_rate": infeasibility_rate,
        "simulation_scenarios": len(scenarios),
        "simulation_seed": scenario_seed,
        "best_bound": model.ObjBound,
        "mip_gap": model.MIPGap if has_solution else "",
    }


def run_experiment(
    instance_sizes: Iterable[tuple[int, int]],
    gammas: Iterable[int],
    seeds: Iterable[int],
    scenario_count: int,
    output_path: Path,
) -> None:
    """Generate each instance once and solve all its model configurations."""

    instance_sizes = list(instance_sizes)
    gammas = list(gammas)
    seeds = list(seeds)
    if any(gamma <= 0 for gamma in gammas):
        raise ValueError("robust gamma values must be greater than zero")

    instance_count = len(instance_sizes) * len(seeds)
    model_count = instance_count * (1 + len(gammas))
    print(
        "Grid: "
        f"sizes={instance_sizes}, gamma={gammas}, seed={seeds}, "
        f"Monte Carlo scenarios={scenario_count}"
    )
    print(f"Distinct instances: {instance_count}; total models: {model_count}")
    print("The complete experiment may take a long time.")

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()

        for (facility_count, customer_count), seed in product(
            instance_sizes, seeds
        ):
            if any(gamma > customer_count for gamma in gammas):
                raise ValueError("gamma cannot exceed the customer count")

            instance = generate_instance(facility_count, customer_count, seed)
            instance_id = make_instance_id(
                facility_count, customer_count, seed
            )
            scenario_seed = simulation_seed(
                facility_count, customer_count, seed
            )
            scenarios = generate_demand_scenarios(
                instance, scenario_count, scenario_seed
            )

            nominal = build_nominal_model(instance)
            writer.writerow(
                solve_model(
                    nominal,
                    instance,
                    "nominal",
                    0.0,
                    instance_id,
                    scenarios,
                    scenario_seed,
                )
            )
            csv_file.flush()
            nominal.dispose()

            for gamma in gammas:
                robust = build_robust_model(instance, float(gamma))
                writer.writerow(
                    solve_model(
                        robust,
                        instance,
                        "robust",
                        float(gamma),
                        instance_id,
                        scenarios,
                        scenario_seed,
                    )
                )
                csv_file.flush()
                robust.dispose()

    print(f"Results saved to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="path of the new CSV file")
    parser.add_argument(
        "--scenario-count",
        type=int,
        default=MONTE_CARLO_SCENARIOS,
        help="number of demand scenarios evaluated for each solution",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or Path(__file__).with_name("cfl_results.csv")
    run_experiment(
        INSTANCE_SIZES,
        GAMMAS,
        SEEDS,
        args.scenario_count,
        output,
    )


if __name__ == "__main__":
    main()
