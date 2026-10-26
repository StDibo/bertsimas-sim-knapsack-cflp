"""Instances and models for the capacitated facility location problem."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import gurobipy as gp
from gurobipy import GRB


@dataclass(frozen=True)
class CFLInstance:
    """Immutable data describing one capacitated facility location instance."""

    facility_count: int
    customer_count: int
    opening_costs: tuple[float, ...]
    assignment_costs: tuple[tuple[float, ...], ...]
    capacities: tuple[float, ...]
    nominal_demands: tuple[float, ...]
    maximum_deviations: tuple[float, ...]


def generate_instance(
    opening_costs: Sequence[float],
    assignment_costs: Sequence[Sequence[float]],
    capacities: Sequence[float],
    nominal_demands: Sequence[float],
    maximum_deviations: Sequence[float],
) -> CFLInstance:
    """Validate explicit data and generate an immutable CFL instance."""

    opening_costs_tuple = tuple(float(value) for value in opening_costs)
    capacities_tuple = tuple(float(value) for value in capacities)
    nominal_demands_tuple = tuple(float(value) for value in nominal_demands)
    maximum_deviations_tuple = tuple(
        float(value) for value in maximum_deviations
    )
    assignment_costs_tuple = tuple(
        tuple(float(value) for value in row) for row in assignment_costs
    )

    facility_count = len(opening_costs_tuple)
    customer_count = len(nominal_demands_tuple)

    if facility_count == 0:
        raise ValueError("at least one candidate facility is required")
    if customer_count == 0:
        raise ValueError("at least one customer is required")
    if len(capacities_tuple) != facility_count:
        raise ValueError("capacities must contain one value per facility")
    if len(assignment_costs_tuple) != facility_count:
        raise ValueError("assignment_costs must contain one row per facility")
    if any(len(row) != customer_count for row in assignment_costs_tuple):
        raise ValueError("each assignment-cost row must contain one value per customer")
    if len(maximum_deviations_tuple) != customer_count:
        raise ValueError("maximum_deviations must contain one value per customer")

    if any(
        not math.isfinite(value) or value < 0
        for value in opening_costs_tuple
    ):
        raise ValueError("opening costs must be finite and non-negative")
    if any(
        not math.isfinite(value) or value <= 0 for value in capacities_tuple
    ):
        raise ValueError("facility capacities must be finite and positive")
    if any(
        not math.isfinite(value) or value <= 0
        for value in nominal_demands_tuple
    ):
        raise ValueError("nominal customer demands must be finite and positive")
    if any(
        not math.isfinite(value) or value < 0
        for value in maximum_deviations_tuple
    ):
        raise ValueError(
            "maximum demand deviations must be finite and non-negative"
        )
    if any(
        not math.isfinite(value) or value < 0
        for row in assignment_costs_tuple
        for value in row
    ):
        raise ValueError("assignment costs must be finite and non-negative")

    return CFLInstance(
        facility_count=facility_count,
        customer_count=customer_count,
        opening_costs=opening_costs_tuple,
        assignment_costs=assignment_costs_tuple,
        capacities=capacities_tuple,
        nominal_demands=nominal_demands_tuple,
        maximum_deviations=maximum_deviations_tuple,
    )


def load_orlib_instance(
    file_path: str | Path,
    *,
    deviation_rates: float | Sequence[float],
    facility_capacity: float | None = None,
) -> CFLInstance:
    """Load OR-Library data and define maximum demand deviations.

    OR-Library files contain nominal demands only. Use a rate of zero to keep
    those demands nominal, one non-negative rate for a common relative
    deviation, or one rate per customer for heterogeneous deviations. Sets
    A--C contain a capacity placeholder, which is replaced by the supplied
    facility_capacity.
    """

    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"OR-Library instance not found: {path}")

    tokens = path.read_text(encoding="utf-8").split()
    if "capacity" in tokens:
        if facility_capacity is None:
            raise ValueError(
                "facility_capacity is required for OR-Library sets A, B, and C"
            )
        tokens = [
            str(facility_capacity) if token == "capacity" else token
            for token in tokens
        ]
    try:
        values = [float(token) for token in tokens]
    except ValueError as error:
        raise ValueError(
            "the loader accepts fully numeric OR-Library files only"
        ) from error

    if len(values) < 2:
        raise ValueError("the OR-Library file is missing its dimensions")
    if not values[0].is_integer() or not values[1].is_integer():
        raise ValueError("facility and customer counts must be integers")

    facility_count = int(values[0])
    customer_count = int(values[1])
    if facility_count <= 0 or customer_count <= 0:
        raise ValueError("facility and customer counts must be positive")

    expected_values = 2 + 2 * facility_count + customer_count * (
        1 + facility_count
    )
    if len(values) != expected_values:
        raise ValueError(
            f"expected {expected_values} numeric values, found {len(values)}"
        )

    cursor = 2
    capacities: list[float] = []
    opening_costs: list[float] = []
    for _ in range(facility_count):
        capacities.append(values[cursor])
        opening_costs.append(values[cursor + 1])
        cursor += 2

    nominal_demands: list[float] = []
    assignment_costs: list[list[float]] = [
        [] for _ in range(facility_count)
    ]
    for _ in range(customer_count):
        nominal_demands.append(values[cursor])
        cursor += 1
        for i in range(facility_count):
            assignment_costs[i].append(values[cursor])
            cursor += 1

    if isinstance(deviation_rates, (int, float)):
        rates = (float(deviation_rates),) * customer_count
    else:
        rates = tuple(float(value) for value in deviation_rates)
    if len(rates) != customer_count:
        raise ValueError("deviation_rates must contain one value per customer")
    if any(not math.isfinite(rate) or rate < 0 for rate in rates):
        raise ValueError("demand deviation rates must be finite and non-negative")

    maximum_deviations = tuple(
        nominal_demands[j] * rates[j] for j in range(customer_count)
    )
    return generate_instance(
        opening_costs=opening_costs,
        assignment_costs=assignment_costs,
        capacities=capacities,
        nominal_demands=nominal_demands,
        maximum_deviations=maximum_deviations,
    )


def select_subinstance(
    instance: CFLInstance,
    facility_indices: Sequence[int],
    customer_indices: Sequence[int],
) -> CFLInstance:
    """Return the CFL subinstance induced by selected facilities and customers."""

    facilities = tuple(facility_indices)
    customers = tuple(customer_indices)
    return generate_instance(
        opening_costs=[instance.opening_costs[i] for i in facilities],
        assignment_costs=[
            [instance.assignment_costs[i][j] for j in customers]
            for i in facilities
        ],
        capacities=[instance.capacities[i] for i in facilities],
        nominal_demands=[instance.nominal_demands[j] for j in customers],
        maximum_deviations=[instance.maximum_deviations[j] for j in customers],
    )


def build_nominal_model(instance: CFLInstance) -> gp.Model:
    """Build the nominal multi-source capacitated facility location model."""

    facilities = range(instance.facility_count)
    customers = range(instance.customer_count)

    model = gp.Model("cfl_nominal")
    y = model.addVars(facilities, vtype=GRB.BINARY, name="y")
    x = model.addVars(
        facilities,
        customers,
        lb=0.0,
        ub=1.0,
        vtype=GRB.CONTINUOUS,
        name="x",
    )

    model.setObjective(
        gp.quicksum(
            instance.opening_costs[i] * y[i] for i in facilities
        )
        + gp.quicksum(
            instance.assignment_costs[i][j] * x[i, j]
            for i in facilities
            for j in customers
        ),
        GRB.MINIMIZE,
    )
    model.addConstrs(
        (
            gp.quicksum(x[i, j] for i in facilities) == 1
            for j in customers
        ),
        name="assignment",
    )
    model.addConstrs(
        (x[i, j] <= y[i] for i in facilities for j in customers),
        name="open_facility",
    )
    model.addConstrs(
        (
            gp.quicksum(
                instance.nominal_demands[j] * x[i, j] for j in customers
            )
            <= instance.capacities[i] * y[i]
            for i in facilities
        ),
        name="capacity",
    )
    model.update()
    return model


def build_robust_model(
    instance: CFLInstance,
    gamma: float,
) -> gp.Model:
    """Build the Bertsimas-Sim robust CFL counterpart.

    One global budget controls demand uncertainty in every capacity constraint.
    """

    facilities = range(instance.facility_count)
    customers = range(instance.customer_count)
    absolute_gamma = float(gamma)
    if not 0 <= absolute_gamma <= instance.customer_count:
        raise ValueError("gamma must be between zero and the customer count")

    model = gp.Model("cfl_robust")
    y = model.addVars(facilities, vtype=GRB.BINARY, name="y")
    x = model.addVars(
        facilities,
        customers,
        lb=0.0,
        ub=1.0,
        vtype=GRB.CONTINUOUS,
        name="x",
    )
    theta = model.addVars(
        facilities, lb=0.0, vtype=GRB.CONTINUOUS, name="theta"
    )
    q = model.addVars(
        facilities,
        customers,
        lb=0.0,
        vtype=GRB.CONTINUOUS,
        name="q",
    )

    model.setObjective(
        gp.quicksum(
            instance.opening_costs[i] * y[i] for i in facilities
        )
        + gp.quicksum(
            instance.assignment_costs[i][j] * x[i, j]
            for i in facilities
            for j in customers
        ),
        GRB.MINIMIZE,
    )
    model.addConstrs(
        (
            gp.quicksum(x[i, j] for i in facilities) == 1
            for j in customers
        ),
        name="assignment",
    )
    model.addConstrs(
        (x[i, j] <= y[i] for i in facilities for j in customers),
        name="open_facility",
    )
    model.addConstrs(
        (
            gp.quicksum(
                instance.nominal_demands[j] * x[i, j] for j in customers
            )
            + absolute_gamma * theta[i]
            + gp.quicksum(q[i, j] for j in customers)
            <= instance.capacities[i] * y[i]
            for i in facilities
        ),
        name="robust_capacity",
    )
    model.addConstrs(
        (
            theta[i] + q[i, j]
            >= instance.maximum_deviations[j] * x[i, j]
            for i in facilities
            for j in customers
        ),
        name="protection",
    )
    model.update()
    return model


def write_nominal_lp(instance: CFLInstance, output_path: str | Path) -> Path:
    """Export one nominal CFL model in LP format."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = build_nominal_model(instance)
    model.write(str(path))
    model.dispose()
    return path


def write_robust_lp(
    instance: CFLInstance,
    gamma: float,
    output_path: str | Path,
) -> Path:
    """Export one Bertsimas-Sim robust CFL model in LP format."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = build_robust_model(instance, gamma)
    model.write(str(path))
    model.dispose()
    return path
