"""Reproducible execution of nominal and robust CFLP experiments."""

import csv
import math
import random
import re
from pathlib import Path
from typing import Iterable

import gurobipy as gp
from gurobipy import GRB

from cfl_generator import (
    CFLInstance,
    build_nominal_model,
    build_robust_model,
    load_orlib_instance,
    select_subinstance,
)

# Download the required OR-Library files into this directory before execution.
ORLIB_DIRECTORY = Path(__file__).parent / "or_library_files"
INSTANCE_NAMES = (
    "cap41",
    "cap42",
    "cap43",
    "cap44",
    "cap61",
    "cap62",
    "cap63",
    "cap64",
    "cap71",
    "cap72",
    "cap73",
    "cap74",
    "cap81",
    "cap82",
    "cap83",
    "cap84",
    "cap91",
    "cap92",
    "cap93",
    "cap94",
    "cap101",
    "cap102",
    "cap103",
    "cap104",
    "cap111",
    "cap112",
    "cap113",
    "cap114",
    "cap121",
    "cap122",
    "cap123",
    "cap124",
    "cap131",
    "cap132",
    "cap133",
    "cap134",
)
CAPOPT_PATH = ORLIB_DIRECTORY / "capopt.txt"

# parameters for orlib instances campaign (benchmark campaign)
DELTAS = (0.10, 0.25, 0.50)
GAMMA_RELS = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 1.00)

# parameters for scalability campaign
SCALABILITY_CAPACITIES = {
    "capa": 8000.0,
    "capb": 5000.0,
    "capc": 5000.0,
}
SCALABILITY_SOURCES = tuple(SCALABILITY_CAPACITIES)
SCALABILITY_SIZES = (
    (10, 100),
    (20, 200),
    (40, 400),
    (80, 800),
    (100, 1000),
)
SCALABILITY_DELTAS = (0.10, 0.25, 0.50)
SCALABILITY_GAMMA_RELS = (0.05, 0.20, 0.40)
SCALABILITY_SELECTION_SEEDS = (0, 1, 2, 3)

# "benchmark" or "scalability"
CAMPAIGN_TO_RUN = "scalability"

TIME_LIMIT_SEC = 1800
SOLVER_THREADS = 4
MIP_GAP = 1e-4
VALIDATION_MIP_GAP = 1e-9
# Positive values below this threshold are treated as numerical noise.
ALLOCATION_TOLERANCE = 1e-8
OPTIMUM_ABS_TOLERANCE = 1e-3

FIELDNAMES = [
    "instance_id",
    "model_type",
    "facility_count",
    "customer_count",
    "capacity_to_demand_ratio",
    "delta",
    "gamma_rel",
    "objective_value",
    "runtime_sec",
    "status",
    "open_facilities",
    "allocation_fractions",
    "best_bound",
    "mip_gap",
]


def load_orlib_optimal_objective(
    optimal_values_path: str | Path,
    instance_name: str,
    facility_capacity: float | None = None,
) -> float:
    """Return the published optimum for a named OR-Library CFLP instance."""

    path = Path(optimal_values_path)
    if not path.is_file():
        raise FileNotFoundError(f"OR-Library optimum file not found: {path}")

    instance_id = Path(instance_name).stem
    capacity_pattern = re.compile(
        r"^\s*(?:(cap[a-z])\s+)?([0-9.]+)\s+"
        r"\(capacity\s+([0-9.]+)\)\s*$"
    )
    current_large_set: str | None = None
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        capacity_match = capacity_pattern.match(line)
        if capacity_match:
            named_set, objective, capacity = capacity_match.groups()
            if named_set is not None:
                current_large_set = named_set
            if (
                current_large_set == instance_id
                and facility_capacity is not None
                and float(capacity) == facility_capacity
            ):
                return float(objective)
            continue

        fields = line.split()
        if not fields or fields[0] != instance_id:
            continue
        if len(fields) < 2:
            raise ValueError(
                f"missing optimum for {instance_id} at line {line_number}"
            )
        try:
            return float(fields[1])
        except ValueError as error:
            raise ValueError(
                f"invalid optimum for {instance_id} at line {line_number}"
            ) from error

    raise ValueError(f"no optimum found for {instance_id} in {path}")


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
    instance: CFLInstance,
    mip_gap: float = MIP_GAP,
) -> dict[str, object]:
    """Optimize one model and return solution and solver statistics."""

    model.Params.OutputFlag = 0
    model.Params.TimeLimit = TIME_LIMIT_SEC
    model.Params.Threads = SOLVER_THREADS
    model.Params.Seed = 0
    model.Params.MIPGap = mip_gap
    model.optimize()

    has_solution = model.SolCount > 0
    open_facilities: list[int] = []
    allocations: list[str] = []
    if has_solution:
        open_facilities = [
            i
            for i in range(instance.facility_count)
            if model.getVarByName(f"y[{i}]").X > 0.5
        ]
        for i in range(instance.facility_count):
            for j in range(instance.customer_count):
                value = model.getVarByName(f"x[{i},{j}]").X
                if value > ALLOCATION_TOLERANCE:
                    allocations.append(f"{i}:{j}:{value:.12g}")

    return {
        "objective_value": model.ObjVal if has_solution else "",
        "runtime_sec": model.Runtime,
        "status": status_name(model.Status),
        "open_facilities": ";".join(str(i) for i in open_facilities),
        "allocation_fractions": ";".join(allocations),
        "best_bound": model.ObjBound,
        "mip_gap": model.MIPGap if has_solution else "",
    }


def validate_nominal_instances(
    instance_names: Iterable[str],
    optimal_values_path: str | Path,
) -> list[dict[str, object]]:
    """Solve and return nominal records after OR-Library validation."""

    records: list[dict[str, object]] = []
    for instance_id in instance_names:
        instance_path = ORLIB_DIRECTORY / f"{instance_id}.txt"
        instance = load_orlib_instance(instance_path, deviation_rates=0.0)
        expected_objective = load_orlib_optimal_objective(
            optimal_values_path,
            instance_id,
        )

        model = build_nominal_model(instance)
        try:
            solution = solve_model(
                model,
                instance,
                mip_gap=VALIDATION_MIP_GAP,
            )
        finally:
            model.dispose()

        record = {
            "instance_id": instance_id,
            "model_type": "nominal",
            "facility_count": instance.facility_count,
            "customer_count": instance.customer_count,
            "capacity_to_demand_ratio": (
                sum(instance.capacities) / sum(instance.nominal_demands)
            ),
            "delta": "",
            "gamma_rel": "",
            **solution,
        }

        if record["status"] != "OPTIMAL":
            raise RuntimeError(
                f"nominal {instance_id} status is {record['status']}, "
                "expected OPTIMAL"
            )

        objective = float(record["objective_value"])
        if not math.isclose(
            objective,
            expected_objective,
            rel_tol=0.0,
            abs_tol=OPTIMUM_ABS_TOLERANCE,
        ):
            raise RuntimeError(
                f"nominal {instance_id} objective does not match the "
                f"OR-Library value: obtained {objective}, "
                f"expected {expected_objective}"
            )

        records.append(record)

    return records


def run_experiment(
    instance_names: Iterable[str],
    deltas: Iterable[float],
    gamma_rels: Iterable[float],
    output_path: Path,
) -> None:
    """Validate nominal models, then solve the configured robust grid."""

    instance_names = tuple(instance_names)
    deltas = tuple(float(delta) for delta in deltas)
    gamma_rels = tuple(float(gamma_rel) for gamma_rel in gamma_rels)
    if any(not math.isfinite(delta) or delta < 0 for delta in deltas):
        raise ValueError("deltas must be finite and non-negative")
    if any(
        not math.isfinite(gamma_rel) or not 0 <= gamma_rel <= 1
        for gamma_rel in gamma_rels
    ):
        raise ValueError("gamma_rels must be finite values between zero and one")
    nominal_model_count = len(instance_names)
    robust_model_count = len(instance_names) * len(deltas) * len(gamma_rels)
    total_model_count = nominal_model_count + robust_model_count
    print(
        "Grid: "
        f"instances={list(instance_names)}, delta={list(deltas)}, "
        f"gamma_rel={list(gamma_rels)}; "
        f"nominal models: {nominal_model_count}; "
        f"robust models: {robust_model_count}; "
        f"total models: {total_model_count}; "
        f"time limit per model: {TIME_LIMIT_SEC} seconds"
    )
    if output_path.exists():
        print(f"Existing results will be overwritten: {output_path}")

    nominal_records = validate_nominal_instances(
        instance_names,
        CAPOPT_PATH,
    )
    print(f"Validated {nominal_model_count} nominal OR-Library instances")

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(nominal_records)
        csv_file.flush()

        for instance_id in instance_names:
            instance_path = ORLIB_DIRECTORY / f"{instance_id}.txt"
            for delta in deltas:
                instance = load_orlib_instance(
                    instance_path,
                    deviation_rates=delta,
                )
                for gamma_rel in gamma_rels:
                    gamma = gamma_rel * instance.customer_count
                    model = build_robust_model(instance, gamma)
                    try:
                        solution = solve_model(
                            model,
                            instance,
                        )
                        record = {
                            "instance_id": instance_id,
                            "model_type": "robust",
                            "facility_count": instance.facility_count,
                            "customer_count": instance.customer_count,
                            "capacity_to_demand_ratio": (
                                sum(instance.capacities)
                                / sum(instance.nominal_demands)
                            ),
                            "delta": delta,
                            "gamma_rel": gamma_rel,
                            **solution,
                        }
                        writer.writerow(record)
                        csv_file.flush()
                    finally:
                        model.dispose()

    print(f"Results saved to {output_path}")


def run_scalability_experiment(output_path: Path) -> None:
    """Compare nominal and robust models on nested A--C subinstances."""

    seed_size_pairs = [
        (selection_seed, size)
        for selection_seed in SCALABILITY_SELECTION_SEEDS
        for size in SCALABILITY_SIZES
        if selection_seed == 0 or size != SCALABILITY_SIZES[-1]
    ]
    nominal_model_count = len(SCALABILITY_SOURCES) * len(seed_size_pairs)
    robust_model_count = (
        nominal_model_count
        * len(SCALABILITY_DELTAS)
        * len(SCALABILITY_GAMMA_RELS)
    )
    scalability_fieldnames = [
        FIELDNAMES[0],
        "selection_seed",
        *FIELDNAMES[1:],
    ]
    total_model_count = nominal_model_count + robust_model_count
    print(
        "Scalability campaign: "
        f"sources={list(SCALABILITY_SOURCES)}, "
        f"sizes={list(SCALABILITY_SIZES)}, "
        f"selection seeds={list(SCALABILITY_SELECTION_SEEDS)}, "
        f"delta={list(SCALABILITY_DELTAS)}, "
        f"gamma_rel={list(SCALABILITY_GAMMA_RELS)}; "
        f"nominal models={nominal_model_count}; "
        f"robust models={robust_model_count}; "
        f"total models={total_model_count}; "
        f"time limit per model={TIME_LIMIT_SEC} seconds"
    )
    if output_path.exists():
        print(f"Existing results will be overwritten: {output_path}")

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=scalability_fieldnames)
        writer.writeheader()

        for source_name in SCALABILITY_SOURCES:
            capacity = SCALABILITY_CAPACITIES[source_name]
            reference_instance = load_orlib_instance(
                ORLIB_DIRECTORY / f"{source_name}.txt",
                deviation_rates=0.0,
                facility_capacity=capacity,
            )
            full_instances = {
                delta: load_orlib_instance(
                    ORLIB_DIRECTORY / f"{source_name}.txt",
                    deviation_rates=delta,
                    facility_capacity=capacity,
                )
                for delta in SCALABILITY_DELTAS
            }
            for selection_seed, (facility_count, customer_count) in (
                seed_size_pairs
            ):
                rng = random.Random(selection_seed)
                facility_indices = list(
                    range(reference_instance.facility_count)
                )
                customer_indices = list(
                    range(reference_instance.customer_count)
                )
                rng.shuffle(facility_indices)
                rng.shuffle(customer_indices)
                instance = select_subinstance(
                    reference_instance,
                    facility_indices[:facility_count],
                    customer_indices[:customer_count],
                )
                instance_id = (
                    f"{source_name}_{capacity:g}_"
                    f"{facility_count}x{customer_count}"
                )
                shared_fields = {
                    "instance_id": instance_id,
                    "selection_seed": selection_seed,
                    "facility_count": facility_count,
                    "customer_count": customer_count,
                    "capacity_to_demand_ratio": (
                        sum(instance.capacities)
                        / sum(instance.nominal_demands)
                    ),
                }

                nominal_model = build_nominal_model(instance)
                try:
                    nominal_solution = solve_model(nominal_model, instance)
                finally:
                    nominal_model.dispose()
                if nominal_solution["status"] == "INTERRUPTED":
                    print("Scalability campaign interrupted; result not saved")
                    return
                writer.writerow(
                    {
                        **shared_fields,
                        "model_type": "nominal",
                        "delta": "",
                        "gamma_rel": "",
                        **nominal_solution,
                    }
                )
                csv_file.flush()

                for delta, full_instance in full_instances.items():
                    robust_instance = select_subinstance(
                        full_instance,
                        facility_indices[:facility_count],
                        customer_indices[:customer_count],
                    )
                    for gamma_rel in SCALABILITY_GAMMA_RELS:
                        gamma = gamma_rel * customer_count
                        robust_model = build_robust_model(
                            robust_instance,
                            gamma,
                        )
                        try:
                            robust_solution = solve_model(
                                robust_model,
                                robust_instance,
                            )
                        finally:
                            robust_model.dispose()
                        if robust_solution["status"] == "INTERRUPTED":
                            print(
                                "Scalability campaign interrupted; "
                                "result not saved"
                            )
                            return
                        writer.writerow(
                            {
                                **shared_fields,
                                "model_type": "robust",
                                "delta": delta,
                                "gamma_rel": gamma_rel,
                                **robust_solution,
                            }
                        )
                        csv_file.flush()

    print(f"Scalability results saved to {output_path}")


def main() -> None:
    if CAMPAIGN_TO_RUN == "benchmark":
        run_experiment(
            INSTANCE_NAMES,
            DELTAS,
            GAMMA_RELS,
            Path(__file__).with_name("cfl_results.csv"),
        )
    elif CAMPAIGN_TO_RUN == "scalability":
        run_scalability_experiment(
            Path(__file__).with_name("cfl_scalability_results.csv")
        )
    else:
        raise ValueError(
            "CAMPAIGN_TO_RUN must be 'benchmark' or 'scalability'"
        )


if __name__ == "__main__":
    main()
