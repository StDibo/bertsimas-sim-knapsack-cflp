"""Generazione riproducibile e modelli per il Knapsack 0-1 robusto."""

from dataclasses import dataclass
from pathlib import Path
import random

import gurobipy as gp
from gurobipy import GRB


@dataclass(frozen=True)
class KnapsackInstance:
    """Dati immutabili di una singola istanza Knapsack."""

    n: int
    profits: tuple[int, ...]
    nominal_weights: tuple[int, ...]
    maximum_deviations: tuple[float, ...]
    capacity: float
    seed: int
    delta: float
    rcp: float


def generate_instance(n: int, delta: float, rcp: float, seed: int) -> KnapsackInstance:
    """Genera un'istanza riproducibile, indipendente dal budget Gamma."""
    if n <= 0:
        raise ValueError("n deve essere maggiore di zero")
    if delta < 0:
        raise ValueError("delta deve essere non negativo")
    if not 0 < rcp <= 1:
        raise ValueError("rcp deve appartenere all'intervallo (0, 1]")

    rng = random.Random(seed)
    nominal_weights = tuple(rng.randint(20, 29) for _ in range(n))
    profits = tuple(rng.randint(16, 77) for _ in range(n))
    maximum_deviations = tuple(
        round(weight * delta, 2) for weight in nominal_weights
    )
    capacity = round(rcp * sum(nominal_weights), 2)

    return KnapsackInstance(
        n=n,
        profits=profits,
        nominal_weights=nominal_weights,
        maximum_deviations=maximum_deviations,
        capacity=capacity,
        seed=seed,
        delta=delta,
        rcp=rcp,
    )


def build_nominal_model(instance: KnapsackInstance) -> gp.Model:
    """Costruisce il modello nominale senza variabili robuste ausiliarie."""
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
    """Costruisce la controparte robusta di Bertsimas e Sim."""
    if not 0 <= gamma_rel <= 1:
        raise ValueError("gamma_rel deve appartenere all'intervallo [0, 1]")

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
    """Esporta un singolo modello nominale in formato LP."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    build_nominal_model(instance).write(str(path))
    return path


def write_robust_lp(
    instance: KnapsackInstance, gamma_rel: float, output_path: str | Path
) -> Path:
    """Esporta un singolo modello robusto in formato LP."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    build_robust_model(instance, gamma_rel).write(str(path))
    return path


def lpGenerator(
    n: int,
    gammaRel: float,
    delta: float,
    rcp: float,
    seed: int | None = None,
    output_path: str | Path = "modelloRobusto.lp",
) -> Path:
    """Wrapper compatibile che genera ed esporta direttamente un LP robusto."""
    if seed is None:
        seed = 0
    instance = generate_instance(n=n, delta=delta, rcp=rcp, seed=seed)
    return write_robust_lp(instance, gammaRel, output_path)
