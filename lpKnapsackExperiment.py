"""Esecuzione riproducibile degli esperimenti Knapsack nominali e robusti."""

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
RCPS = [0.20, 0.50, 0.80]
GAMMA_RELS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
SEEDS = [1, 2, 3, 4, 5]

TIME_LIMIT_SEC = 1800
SOLVER_THREADS = 1
SOLVER_SEED = 0
EXPORT_ALL_LP = False

FIELDNAMES = [
    "instance_id",
    "model_type",
    "n",
    "delta",
    "rcp",
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


def make_instance_id(n: int, delta: float, rcp: float, seed: int) -> str:
    """Costruisce un identificativo leggibile dai soli parametri rigenerabili."""
    return f"n{n}_d{delta:.2f}_r{rcp:.2f}_s{seed}"


def status_name(status: int) -> str:
    """Restituisce un nome stabile per gli stati Gurobi più rilevanti."""
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
    """Ottimizza un modello e raccoglie anche incumbent e statistiche."""
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = TIME_LIMIT_SEC
    model.Params.Threads = SOLVER_THREADS
    model.Params.Seed = SOLVER_SEED
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
        "rcp": instance.rcp,
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
    rcps: Iterable[float],
    gamma_rels: Iterable[float],
    seeds: Iterable[int],
    output_path: Path,
    export_all_lp: bool = EXPORT_ALL_LP,
) -> None:
    """Genera ogni istanza una volta, quindi risolve nominale e robusti."""
    ns, deltas, rcps = list(ns), list(deltas), list(rcps)
    gamma_rels, seeds = list(gamma_rels), list(seeds)
    instance_count = len(ns) * len(deltas) * len(rcps) * len(seeds)
    model_count = instance_count * (1 + len(gamma_rels))

    print(f"Griglia: n={ns}, delta={deltas}, rcp={rcps}, gamma_rel={gamma_rels}, seed={seeds}")
    print(f"Istanze distinte: {instance_count}; modelli totali: {model_count}")
    print("La campagna completa può richiedere molto tempo.")

    if output_path.exists():
        raise FileExistsError(
            f"Il file {output_path} esiste già: non verrà sovrascritto."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lp_directory = output_path.parent / "modelli_lp"
    if export_all_lp:
        lp_directory.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()

        for n, delta, rcp, seed in product(ns, deltas, rcps, seeds):
            instance = generate_instance(n, delta, rcp, seed)
            instance_id = make_instance_id(n, delta, rcp, seed)

            nominal = build_nominal_model(instance)
            if export_all_lp:
                nominal.write(str(lp_directory / f"{instance_id}_nominal.lp"))
            writer.writerow(solve_model(nominal, instance, "nominal", 0.0, instance_id))
            csv_file.flush()
            nominal.dispose()

            for gamma_rel in gamma_rels:
                robust = build_robust_model(instance, gamma_rel)
                if export_all_lp:
                    suffix = f"{gamma_rel:.2f}".replace(".", "p")
                    robust.write(
                        str(lp_directory / f"{instance_id}_robust_g{suffix}.lp")
                    )
                writer.writerow(
                    solve_model(robust, instance, "robust", gamma_rel, instance_id)
                )
                csv_file.flush()
                robust.dispose()

    print(f"Risultati salvati in {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="percorso del nuovo CSV")
    parser.add_argument("--export-all-lp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or Path(__file__).with_name("risultati_knapsack_nuovi.csv")
    run_experiment(NS, DELTAS, RCPS, GAMMA_RELS, SEEDS, output, args.export_all_lp)


if __name__ == "__main__":
    main()
