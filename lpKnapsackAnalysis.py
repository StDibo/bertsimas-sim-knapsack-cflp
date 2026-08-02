"""Validazione e analisi dei risultati sperimentali del Knapsack."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REQUIRED_COLUMNS = {
    "instance_id",
    "model_type",
    "n",
    "delta",
    "rcp",
    "seed",
    "gamma_rel",
    "objective_value",
    "runtime_sec",
    "status",
    "selected_items",
    "num_variables",
    "num_constraints",
}


def load_results(file_path: str | Path) -> pd.DataFrame:
    """Carica il CSV nuovo e ne valida lo schema minimo."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File dei risultati non trovato: {path}")

    data = pd.read_csv(path, keep_default_na=False)
    data.columns = data.columns.str.strip().str.lower()
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(
            "Schema CSV non aggiornato; colonne mancanti: " + ", ".join(sorted(missing))
        )

    numeric_columns = [
        "n",
        "delta",
        "rcp",
        "seed",
        "gamma_rel",
        "gamma",
        "objective_value",
        "runtime_sec",
        "selected_count",
        "best_bound",
        "mip_gap",
        "node_count",
        "num_variables",
        "num_constraints",
        "solution_count",
    ]
    for column in numeric_columns:
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    data["model_type"] = data["model_type"].str.strip().str.lower()
    data["status"] = data["status"].str.strip().str.upper()
    return data


def parse_selected_items(value: object) -> set[int]:
    """Converte la rappresentazione CSV separata da punto e virgola."""
    text = str(value).strip()
    if not text:
        return set()
    return {int(item) for item in text.split(";") if item != ""}


def optimal_complete_results(data: pd.DataFrame) -> pd.DataFrame:
    """Seleziona istanze ottime nel nominale e in tutti i budget presenti."""
    nominal = data[(data["model_type"] == "nominal") & (data["status"] == "OPTIMAL")]
    robust = data[(data["model_type"] == "robust") & (data["status"] == "OPTIMAL")]
    expected_gammas = set(data.loc[data["model_type"] == "robust", "gamma_rel"].dropna())

    nominal_ids = set(nominal["instance_id"])
    complete_ids = {
        instance_id
        for instance_id, group in robust.groupby("instance_id")
        if set(group["gamma_rel"]) == expected_gammas
        and not group["gamma_rel"].duplicated().any()
    }
    included_ids = nominal_ids.intersection(complete_ids)
    return data[data["instance_id"].isin(included_ids)].copy()


def add_comparison_metrics(data: pd.DataFrame) -> pd.DataFrame:
    """Associa il vero nominale e calcola PoR, Hamming e Jaccard per istanza."""
    complete = optimal_complete_results(data)
    nominal = complete[complete["model_type"] == "nominal"][
        ["instance_id", "objective_value", "selected_items"]
    ].rename(
        columns={
            "objective_value": "nominal_objective",
            "selected_items": "nominal_selected_items",
        }
    )
    robust = complete[complete["model_type"] == "robust"].merge(
        nominal, on="instance_id", how="inner", validate="many_to_one"
    )
    robust["price_of_robustness_pct"] = (
        (robust["nominal_objective"] - robust["objective_value"])
        / robust["nominal_objective"]
        * 100
    )

    def structural_metrics(row: pd.Series) -> pd.Series:
        nominal_set = parse_selected_items(row["nominal_selected_items"])
        robust_set = parse_selected_items(row["selected_items"])
        symmetric_difference = nominal_set.symmetric_difference(robust_set)
        union = nominal_set.union(robust_set)
        return pd.Series(
            {
                "hamming_distance": len(symmetric_difference),
                "hamming_distance_normalized": len(symmetric_difference) / row["n"],
                "jaccard_similarity": len(nominal_set.intersection(robust_set)) / len(union)
                if union
                else 1.0,
            }
        )

    return pd.concat([robust, robust.apply(structural_metrics, axis=1)], axis=1)


def monotonicity_violations(
    comparisons: pd.DataFrame, tolerance: float = 1e-6
) -> pd.DataFrame:
    """Restituisce gli aumenti dell'obiettivo al crescere di Gamma."""
    violations: list[dict[str, object]] = []
    for instance_id, group in comparisons.groupby("instance_id"):
        ordered = group.sort_values("gamma_rel")
        previous = None
        for row in ordered.itertuples(index=False):
            if previous is not None and row.objective_value > previous.objective_value + tolerance:
                violations.append(
                    {
                        "instance_id": instance_id,
                        "gamma_rel_previous": previous.gamma_rel,
                        "gamma_rel_current": row.gamma_rel,
                        "objective_previous": previous.objective_value,
                        "objective_current": row.objective_value,
                    }
                )
            previous = row
    return pd.DataFrame(
        violations,
        columns=[
            "instance_id",
            "gamma_rel_previous",
            "gamma_rel_current",
            "objective_previous",
            "objective_current",
        ],
    )


def save_line_plot(
    summary: pd.DataFrame,
    x: str,
    groups: str,
    y: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(9, 5))
    for group_value, group in summary.groupby(groups):
        ordered = group.sort_values(x)
        axis.plot(ordered[x], ordered[y], marker="o", label=f"{groups}={group_value}")
    axis.set(title=title, xlabel=r"Budget relativo $\Gamma_{rel}$", ylabel=ylabel)
    axis.grid(True, linestyle="--", alpha=0.5)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def create_analysis(data: pd.DataFrame, output_directory: str | Path) -> None:
    """Esegue controlli e produce tabelle e grafici sulle sole istanze complete."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    comparisons = add_comparison_metrics(data)
    included = comparisons["instance_id"].nunique()
    total = data["instance_id"].nunique()
    print(f"Istanze incluse nelle curve complete: {included} su {total}")
    if comparisons.empty:
        raise ValueError("Nessuna istanza ottima e completa disponibile per l'analisi")

    comparisons.to_csv(output / "confronti_per_istanza.csv", index=False)
    violations = monotonicity_violations(comparisons)
    violations.to_csv(output / "violazioni_monotonicita.csv", index=False)
    print(f"Violazioni di monotonicità: {len(violations)}")

    budget_summary = comparisons.groupby("gamma_rel", as_index=False).agg(
        price_of_robustness_mean=("price_of_robustness_pct", "mean"),
        price_of_robustness_std=("price_of_robustness_pct", "std"),
        hamming_normalized_mean=("hamming_distance_normalized", "mean"),
        jaccard_mean=("jaccard_similarity", "mean"),
        included_instances=("instance_id", "nunique"),
    )
    budget_summary.to_csv(output / "riepilogo_budget.csv", index=False)

    for parameter, filename, title in (
        ("rcp", "impatto_rcp.png", "Price of Robustness per rapporto di capacità"),
        ("delta", "impatto_delta.png", "Price of Robustness per deviazione massima"),
    ):
        summary = comparisons.groupby([parameter, "gamma_rel"], as_index=False)[
            "price_of_robustness_pct"
        ].mean()
        save_line_plot(
            summary,
            "gamma_rel",
            parameter,
            "price_of_robustness_pct",
            "Price of Robustness medio (%)",
            title,
            output / filename,
        )

    fig, axis = plt.subplots(figsize=(9, 5))
    axis.plot(
        budget_summary["gamma_rel"],
        budget_summary["price_of_robustness_mean"],
        marker="o",
    )
    axis.set(
        title="Sensibilità al budget di incertezza",
        xlabel=r"Budget relativo $\Gamma_{rel}$",
        ylabel="Price of Robustness medio (%)",
    )
    axis.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output / "sensibilita_budget.png", dpi=300)
    plt.close(fig)

    optimal = data[data["status"] == "OPTIMAL"].copy()
    runtime_summary = optimal.groupby(["model_type", "n", "gamma_rel"], as_index=False)[
        "runtime_sec"
    ].mean()
    runtime_summary.to_csv(output / "riepilogo_tempi.csv", index=False)

    nominal_runtime = optimal[optimal["model_type"] == "nominal"][
        ["instance_id", "runtime_sec"]
    ].rename(columns={"runtime_sec": "nominal_runtime_sec"})
    robust_runtime = optimal[optimal["model_type"] == "robust"].merge(
        nominal_runtime, on="instance_id", how="inner", validate="many_to_one"
    )
    robust_runtime["runtime_ratio"] = (
        robust_runtime["runtime_sec"] / robust_runtime["nominal_runtime_sec"]
    )
    scalability = robust_runtime.groupby(["n", "gamma_rel"], as_index=False).agg(
        robust_runtime_mean=("runtime_sec", "mean"),
        nominal_runtime_mean=("nominal_runtime_sec", "mean"),
        runtime_ratio_mean=("runtime_ratio", "mean"),
        included_pairs=("instance_id", "nunique"),
    )
    scalability.to_csv(output / "riepilogo_scalabilita.csv", index=False)
    save_line_plot(
        scalability,
        "gamma_rel",
        "n",
        "runtime_ratio_mean",
        "Rapporto medio tempo robusto/nominale",
        "Costo computazionale della formulazione robusta",
        output / "scalabilita_runtime_ratio.png",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("analisi_knapsack"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_analysis(load_results(args.csv_path), args.output_dir)


if __name__ == "__main__":
    main()
