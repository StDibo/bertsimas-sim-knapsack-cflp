"""Reproducible instance generation and models for the robust 0-1 Knapsack."""

from dataclasses import dataclass
from pathlib import Path
import random

import gurobipy as gp
from gurobipy import GRB


@dataclass(frozen=True)
class KnapsackInstance:
    """Immutable data describing one Knapsack instance."""

    n: int
    profits: tuple[int, ...]
    nominal_weights: tuple[int, ...]
    maximum_deviations: tuple[float, ...]
    capacity: float
    seed: int
    delta: float
    capacity_ratio: float


def generate_instance(
    n: int, delta: float, capacity_ratio: float, seed: int
) -> KnapsackInstance:
    """Generate a reproducible instance independently of the Gamma budget."""
    
    if n <= 0:
        raise ValueError("n must be greater than zero")
    if delta < 0:
        raise ValueError("delta must be non-negative")
    if not 0 < capacity_ratio <= 1:
        raise ValueError("capacity_ratio must be in the interval (0, 1]")

    rng = random.Random(seed)
    nominal_weights = tuple(rng.randint(20, 29) for _ in range(n)) # to modify later on
    profits = tuple(rng.randint(16, 77) for _ in range(n))
    maximum_deviations = tuple(
        round(weight * delta, 2) for weight in nominal_weights
    )
    capacity = round(capacity_ratio * sum(nominal_weights), 2)

    return KnapsackInstance(
        n=n,
        profits=profits,
        nominal_weights=nominal_weights,
        maximum_deviations=maximum_deviations,
        capacity=capacity,
        seed=seed,
        delta=delta,
        capacity_ratio=capacity_ratio,
    )


def build_nominal_model(instance: KnapsackInstance) -> gp.Model:
    """Build the nominal model without robust auxiliary variables."""

    model = gp.Model("knapsack_nominal")
    x = model.addVars(instance.n, vtype=GRB.BINARY, name="x")
    model.setObjective(
        gp.quicksum(instance.profits[j] * x[j] for j in range(instance.n)),
        GRB.MAXIMIZE,
    )
    model.addConstr(
        gp.quicksum(
            instance.nominal_weights[j] * x[j] for j in range(instance.n)
        )
        <= instance.capacity,
        name="capacity",
    )
    model.update()
    return model


def build_robust_model(
    instance: KnapsackInstance, gamma_rel: float
) -> gp.Model:
    """Build the Bertsimas-Sim robust counterpart."""

    if not 0 <= gamma_rel <= 1:
        raise ValueError("gamma_rel must be in the interval [0, 1]")

    gamma = gamma_rel * instance.n
    model = gp.Model("knapsack_robust")
    x = model.addVars(instance.n, vtype=GRB.BINARY, name="x")
    theta = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="theta")
    q = model.addVars(instance.n, lb=0.0, vtype=GRB.CONTINUOUS, name="q")

    model.setObjective(
        gp.quicksum(instance.profits[j] * x[j] for j in range(instance.n)),
        GRB.MAXIMIZE,
    )
    model.addConstr(
        gp.quicksum(
            instance.nominal_weights[j] * x[j] for j in range(instance.n)
        )
        + gamma * theta
        + gp.quicksum(q[j] for j in range(instance.n))
        <= instance.capacity,
        name="capacity",
    )
    model.addConstrs(
        (
            theta + q[j] >= instance.maximum_deviations[j] * x[j]
            for j in range(instance.n)
        ),
        name="protection",
    )
    model.update()
    return model


def write_nominal_lp(instance: KnapsackInstance, output_path: str | Path) -> Path:
    """Export one nominal model in LP format."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    build_nominal_model(instance).write(str(path))
    return path


def write_robust_lp(
    instance: KnapsackInstance, gamma_rel: float, output_path: str | Path
) -> Path:
    """Export one robust model in LP format."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    build_robust_model(instance, gamma_rel).write(str(path))
    return path
