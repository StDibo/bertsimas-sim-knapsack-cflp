"""Reproducible instances and models for the robust single-source CFLP."""

import math
import random
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
    seed: int | None = None


def create_instance(
    opening_costs: Sequence[float],
    assignment_costs: Sequence[Sequence[float]],
    capacities: Sequence[float],
    nominal_demands: Sequence[float],
    maximum_deviations: Sequence[float],
    seed: int | None = None,
) -> CFLInstance:
    """Validate explicit data and create an immutable CFL instance."""

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

    if any(value < 0 for value in opening_costs_tuple):
        raise ValueError("opening costs must be non-negative")
    if any(value <= 0 for value in capacities_tuple):
        raise ValueError("facility capacities must be greater than zero")
    if any(value <= 0 for value in nominal_demands_tuple):
        raise ValueError("nominal customer demands must be greater than zero")
    if any(value < 0 for value in maximum_deviations_tuple):
        raise ValueError("maximum demand deviations must be non-negative")
    if any(value < 0 for row in assignment_costs_tuple for value in row):
        raise ValueError("assignment costs must be non-negative")

    return CFLInstance(
        facility_count=facility_count,
        customer_count=customer_count,
        opening_costs=opening_costs_tuple,
        assignment_costs=assignment_costs_tuple,
        capacities=capacities_tuple,
        nominal_demands=nominal_demands_tuple,
        maximum_deviations=maximum_deviations_tuple,
        seed=seed,
    )


def generate_instance(
    facility_count: int,
    customer_count: int,
    seed: int,
) -> CFLInstance:
    """Generate one reproducible instance using the Ryu--Park T4 scheme.

    Nominal demands, capacities, and opening costs are sampled from the
    integer intervals [10, 50], [100, 500], and [300, 700]. Facilities and
    customers are placed uniformly in a 190 by 190 square; assignment costs
    are the floors of their Euclidean distances. Each maximum demand
    deviation is the floor of the nominal demand multiplied by a rate sampled
    from 0.100 to 0.500 in increments of 0.001.
    """

    if facility_count <= 0:
        raise ValueError("facility_count must be greater than zero")
    if customer_count <= 0:
        raise ValueError("customer_count must be greater than zero")

    rng = random.Random(seed)
    nominal_demands = tuple(
        float(rng.randint(10, 50)) for _ in range(customer_count)
    )
    capacities = tuple(
        float(rng.randint(100, 500)) for _ in range(facility_count)
    )
    opening_costs = tuple(
        float(rng.randint(300, 700)) for _ in range(facility_count)
    )
    facility_positions = tuple(
        (rng.uniform(0.0, 190.0), rng.uniform(0.0, 190.0))
        for _ in range(facility_count)
    )
    customer_positions = tuple(
        (rng.uniform(0.0, 190.0), rng.uniform(0.0, 190.0))
        for _ in range(customer_count)
    )
    assignment_costs = tuple(
        tuple(
            float(
                math.floor(
                    math.dist(facility_positions[i], customer_positions[j])
                )
            )
            for j in range(customer_count)
        )
        for i in range(facility_count)
    )
    maximum_deviations = tuple(
        float(math.floor(demand * rng.randint(100, 500) / 1000.0))
        for demand in nominal_demands
    )

    return create_instance(
        opening_costs=opening_costs,
        assignment_costs=assignment_costs,
        capacities=capacities,
        nominal_demands=nominal_demands,
        maximum_deviations=maximum_deviations,
        seed=seed,
    )


def _absolute_budgets_by_facility(
    gammas: float | Sequence[float],
    facility_count: int,
    customer_count: int,
) -> tuple[float, ...]:
    """Return one validated absolute uncertainty budget per facility."""

    if isinstance(gammas, (int, float)):
        budgets = (float(gammas),) * facility_count
    else:
        budgets = tuple(float(value) for value in gammas)

    if len(budgets) != facility_count:
        raise ValueError("gammas must contain one value per facility")
    if any(not 0 <= value <= customer_count for value in budgets):
        raise ValueError(
            "each absolute budget must be between zero and the customer count"
        )
    return budgets


def build_nominal_model(instance: CFLInstance) -> gp.Model:
    """Build the nominal static single-source CFL model."""

    facilities = range(instance.facility_count)
    customers = range(instance.customer_count)

    model = gp.Model("cfl_nominal")
    y = model.addVars(facilities, vtype=GRB.BINARY, name="y")
    x = model.addVars(facilities, customers, vtype=GRB.BINARY, name="x")

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
    gammas: float | Sequence[float],
) -> gp.Model:
    """Build the compact robust CFL counterpart for uncertain demand.

    ``gammas[i]`` is the absolute budget associated with the capacity
    constraint of facility ``i``. Passing one scalar is a convenience for
    assigning the same absolute budget to every facility.
    """

    facilities = range(instance.facility_count)
    customers = range(instance.customer_count)
    absolute_budgets = _absolute_budgets_by_facility(
        gammas,
        instance.facility_count,
        instance.customer_count,
    )

    model = gp.Model("cfl_robust")
    y = model.addVars(facilities, vtype=GRB.BINARY, name="y")
    x = model.addVars(facilities, customers, vtype=GRB.BINARY, name="x")
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
            + absolute_budgets[i] * theta[i]
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
    gammas: float | Sequence[float],
    output_path: str | Path,
) -> Path:
    """Export one robust CFL model in LP format."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = build_robust_model(instance, gammas)
    model.write(str(path))
    model.dispose()
    return path
