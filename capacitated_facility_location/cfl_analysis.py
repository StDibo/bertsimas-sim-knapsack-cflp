"""Create the CFLP summaries reported in Chapter 5 of the thesis."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# Maps the last digit of the OR-Library instance name to its opening-cost level.
OPENING_COST_LEVELS = {1: 7500, 2: 12500, 3: 17500, 4: 25000}


def index_set(value: object) -> set[int]:
    """Convert a semicolon-separated collection of indices to a set."""
    if pd.isna(value) or value == "":
        return set()
    return {int(index) for index in str(value).split(";")}


def calculate_price_of_robustness(results: pd.DataFrame) -> pd.DataFrame:
    """Add nominal baselines, Price of Robustness, and opening-set changes."""
    nominal = results[results["model_type"] == "nominal"][
        ["instance_id", "objective_value", "open_facilities"]
    ].rename(
        columns={
            "objective_value": "nominal_objective",
            "open_facilities": "nominal_open_facilities",
        }
    )
    robust = results[results["model_type"] == "robust"].copy()
    robust = robust.merge(nominal, on="instance_id")

    robust["price_of_robustness_pct"] = (
        100.0 * (robust["objective_value"] / robust["nominal_objective"] - 1.0)
    ).round(12)
    robust.loc[
        robust["status"] != "OPTIMAL", "price_of_robustness_pct"
    ] = pd.NA

    nominal_open = robust["nominal_open_facilities"].map(index_set)
    robust_open = robust["open_facilities"].map(index_set)
    robust["nominal_open_count"] = nominal_open.map(len)
    robust["robust_open_count"] = robust_open.map(len)
    robust["additional_open_facilities"] = (
        robust["robust_open_count"] - robust["nominal_open_count"]
    ).astype(float)
    robust["opening_hamming_pct"] = [
        100 * len(nominal_set ^ robust_set) / facility_count
        for nominal_set, robust_set, facility_count in zip(
            nominal_open,
            robust_open,
            robust["facility_count"],
        )
    ]
    robust["opening_jaccard"] = [
        (
            len(nominal_set & robust_set) / len(nominal_set | robust_set)
            if nominal_set | robust_set
            else 1.0
        )
        for nominal_set, robust_set in zip(nominal_open, robust_open)
    ]
    numeric_solution_metrics = [
        "additional_open_facilities",
        "opening_hamming_pct",
        "opening_jaccard",
    ]
    nonoptimal = robust["status"] != "OPTIMAL"
    robust.loc[nonoptimal, numeric_solution_metrics] = float("nan")

    variants = robust["instance_id"].str[-1].astype(int)
    robust["opening_cost_level"] = variants.map(OPENING_COST_LEVELS)
    robust["facility_capacity_to_demand_ratio"] = (
        robust["capacity_to_demand_ratio"] / robust["facility_count"]
    ).round(12)
    capacity_ratios = sorted(
        robust["facility_capacity_to_demand_ratio"].unique()
    )
    regime_by_ratio = dict(
        zip(capacity_ratios, ("tight", "intermediate", "large"))
    )
    robust["capacity_regime"] = robust[
        "facility_capacity_to_demand_ratio"
    ].map(regime_by_ratio)
    return robust


def complete_panel(robust: pd.DataFrame) -> pd.DataFrame:
    """Return instances solved optimally over the complete parameter grid."""
    complete_instances = robust.groupby("instance_id")["status"].apply(
        lambda statuses: (statuses == "OPTIMAL").all()
    )
    included_ids = complete_instances[complete_instances].index
    return robust[robust["instance_id"].isin(included_ids)]


def summarize_price_of_robustness(robust: pd.DataFrame) -> pd.DataFrame:
    """Calculate mean PoR on instances optimal over the complete parameter grid."""
    return (
        complete_panel(robust)
        .groupby(["gamma_rel", "delta"])
        .agg(
            mean_por_pct=("price_of_robustness_pct", "mean"),
            mean_additional_facilities=("additional_open_facilities", "mean"),
            mean_opening_hamming_pct=("opening_hamming_pct", "mean"),
            mean_opening_jaccard=("opening_jaccard", "mean"),
        )
        .reset_index()
        .sort_values(["gamma_rel", "delta"])
    )


def summarize_capacity_trajectories(robust: pd.DataFrame) -> pd.DataFrame:
    """Summarize budget effects by capacity regime on the common panel."""

    return (
        complete_panel(robust)
        .groupby(["capacity_regime", "delta", "gamma_rel"])
        .agg(
            mean_por_pct=("price_of_robustness_pct", "mean"),
            mean_additional_facilities=("additional_open_facilities", "mean"),
        )
        .reset_index()
    )


def summarize_opening_cost_trajectories(robust: pd.DataFrame) -> pd.DataFrame:
    """Summarize budget effects by opening-cost level on the common panel."""

    return (
        complete_panel(robust)
        .groupby(["opening_cost_level", "delta", "gamma_rel"])
        .agg(
            mean_por_pct=("price_of_robustness_pct", "mean"),
            mean_additional_facilities=("additional_open_facilities", "mean"),
        )
        .reset_index()
    )


def plot_budget_trajectories(
    summary: pd.DataFrame,
    group_column: str,
    group_labels: dict[object, str],
    output_path: Path,
) -> None:
    """Plot PoR and facility-opening trajectories for the three deviations."""

    metrics = (
        ("mean_por_pct", "Mean Price of Robustness (%)"),
        ("mean_additional_facilities", "Mean additional open facilities"),
    )
    deltas = sorted(summary["delta"].unique())
    figure, axes = plt.subplots(2, len(deltas), figsize=(12, 7), sharex=True)

    for column, delta in enumerate(deltas):
        delta_data = summary[summary["delta"] == delta]
        for group_value, label in group_labels.items():
            group_data = delta_data[delta_data[group_column] == group_value]
            for row, (metric, ylabel) in enumerate(metrics):
                axes[row, column].plot(
                    group_data["gamma_rel"],
                    group_data[metric],
                    marker="o",
                    label=label,
                )
                axes[row, column].set_ylabel(ylabel if column == 0 else "")
                axes[row, column].grid(alpha=0.25)
        axes[0, column].set_title(rf"$\delta={delta:.2f}$")
        axes[1, column].set_xlabel(r"$\Gamma_{rel}$")

    axes[0, -1].legend(loc="best")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def summarize_cap4_stress_case(robust: pd.DataFrame) -> pd.DataFrame:
    """Summarize the cap41--cap44 cases."""
    cap4 = robust[
        robust["instance_id"].isin(["cap41", "cap42", "cap43", "cap44"])
        & (robust["delta"] == 0.50)
        & robust["gamma_rel"].isin([0.20, 0.30, 0.40, 1.00])
    ].copy()
    cap4["is_optimal"] = cap4["status"] == "OPTIMAL"
    cap4["is_infeasible"] = cap4["status"] == "INFEASIBLE"
    cap4["open_facilities_if_optimal"] = cap4["robust_open_count"].where(
        cap4["is_optimal"]
    )
    return (
        cap4.groupby("gamma_rel")
        .agg(
            optimal_instances=("is_optimal", "sum"),
            infeasible_instances=("is_infeasible", "sum"),
            minimum_por_pct=("price_of_robustness_pct", "min"),
            maximum_por_pct=("price_of_robustness_pct", "max"),
            minimum_open_facilities=("open_facilities_if_optimal", "min"),
            maximum_open_facilities=("open_facilities_if_optimal", "max"),
        )
        .reset_index()
    )


def calculate_scalability_price_of_robustness(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Add the nominal baseline and exact PoR to scalability results."""

    instance_columns = ["instance_id", "selection_seed"]
    nominal = results[
        (results["model_type"] == "nominal")
        & (results["status"] == "OPTIMAL")
    ][instance_columns + ["objective_value"]].rename(
        columns={"objective_value": "nominal_objective"}
    )
    robust = results[results["model_type"] == "robust"].copy()
    robust = robust.merge(nominal, on=instance_columns)
    robust["price_of_robustness_pct"] = 100.0 * (
        robust["objective_value"] / robust["nominal_objective"] - 1.0
    )
    robust.loc[
        robust["status"] != "OPTIMAL", "price_of_robustness_pct"
    ] = pd.NA
    return robust


def summarize_scalability_runtime(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize observed runtime by size and model type."""

    runtime = results.copy()
    runtime["is_time_limit"] = runtime["status"] == "TIME_LIMIT"
    return (
        runtime.groupby(
            ["facility_count", "customer_count", "model_type"]
        )
        .agg(
            mean_runtime_sec=("runtime_sec", "mean"),
            runs=("runtime_sec", "size"),
            time_limit_runs=("is_time_limit", "sum"),
        )
        .reset_index()
        .sort_values(["customer_count", "model_type"])
    )


def complete_scalability_panel(robust: pd.DataFrame) -> pd.DataFrame:
    """Keep subinstances optimal over all nine robust configurations."""

    complete = robust.groupby(
        ["instance_id", "selection_seed"]
    )["status"].transform(lambda statuses: (statuses == "OPTIMAL").all())
    return robust[complete]


def summarize_scalability_objective_effects(
    robust: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize exact PoR on a fixed panel within each problem size."""

    robust = robust.copy()
    robust["is_optimal"] = robust["status"] == "OPTIMAL"
    robust["is_time_limit"] = robust["status"] == "TIME_LIMIT"
    group_columns = [
        "facility_count",
        "customer_count",
        "delta",
        "gamma_rel",
    ]
    status_counts = (
        robust.groupby(group_columns)
        .agg(
            optimal_runs=("is_optimal", "sum"),
            time_limit_runs=("is_time_limit", "sum"),
        )
        .reset_index()
    )
    complete_summary = (
        complete_scalability_panel(robust)
        .groupby(group_columns)
        .agg(
            mean_por_pct=("price_of_robustness_pct", "mean"),
            included_subinstances=("instance_id", "size"),
        )
        .reset_index()
    )
    summary = status_counts.merge(
        complete_summary,
        on=group_columns,
        how="left",
    )
    summary["included_subinstances"] = (
        summary["included_subinstances"].fillna(0).astype(int)
    )
    return summary.sort_values(["customer_count", "delta", "gamma_rel"])


def plot_scalability_runtime(
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot nominal and robust mean observed runtimes by size."""

    figure, axis = plt.subplots(figsize=(7, 4.5))
    sizes = (
        summary[["facility_count", "customer_count"]]
        .drop_duplicates()
        .sort_values("customer_count")
    )
    size_labels = [
        f"{facilities}x{customers}"
        for facilities, customers in sizes.itertuples(index=False, name=None)
    ]
    for model_type, label in (
        ("nominal", "Nominal"),
        ("robust", "Robust"),
    ):
        model_data = summary[summary["model_type"] == model_type].sort_values(
            "customer_count"
        )
        axis.plot(
            size_labels,
            model_data["mean_runtime_sec"],
            marker="o",
            label=label,
        )

    axis.set_xlabel("Facilities x customers")
    axis.set_ylabel("Mean observed runtime (seconds, log scale)")
    axis.set_yscale("log")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_scalability_objective_effects(
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot mean exact PoR by size for each deviation and budget."""

    deltas = sorted(summary["delta"].unique())
    gamma_rels = sorted(summary["gamma_rel"].unique())
    sizes = (
        summary[["facility_count", "customer_count"]]
        .drop_duplicates()
        .sort_values("customer_count")
    )
    size_labels = [
        f"{facilities}x{customers}"
        for facilities, customers in sizes.itertuples(index=False, name=None)
    ]
    figure, axes = plt.subplots(
        1,
        len(deltas),
        figsize=(12, 4),
        sharex=True,
        sharey=True,
    )

    for axis, delta in zip(axes, deltas):
        delta_data = summary[summary["delta"] == delta]
        for gamma_rel in gamma_rels:
            budget_data = delta_data[
                delta_data["gamma_rel"] == gamma_rel
            ].sort_values("customer_count")
            axis.plot(
                size_labels,
                budget_data["mean_por_pct"],
                marker="o",
                label=rf"$\Gamma_{{rel}}={gamma_rel:.2f}$",
            )
        axis.set_title(rf"$\delta={delta:.2f}$")
        axis.set_xlabel("Facilities x customers")
        axis.grid(alpha=0.25)

    axes[0].set_ylabel("Mean Price of Robustness (%)\n(complete panels only)")
    axes[-1].legend(loc="best")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Read the experiment CSV and write the summaries."""
    directory = Path(__file__).resolve().parent
    output_directory = directory / "cfl_analysis"
    output_directory.mkdir(exist_ok=True)

    results = pd.read_csv(directory / "cfl_results.csv")
    robust = calculate_price_of_robustness(results)
    scalability_results = pd.read_csv(
        directory / "cfl_scalability_results.csv"
    )
    scalability_robust = calculate_scalability_price_of_robustness(
        scalability_results
    )

    capacity_trajectories = summarize_capacity_trajectories(robust)
    opening_cost_trajectories = summarize_opening_cost_trajectories(robust)
    scalability_runtime = summarize_scalability_runtime(scalability_results)
    scalability_objective_effects = summarize_scalability_objective_effects(
        scalability_robust
    )

    summaries = {
        "cfl_por_summary.csv": summarize_price_of_robustness(robust),
        "cfl_cap4_stress_summary.csv": summarize_cap4_stress_case(robust),
        "cfl_scalability_runtime_summary.csv": scalability_runtime,
        "cfl_scalability_por_summary.csv": scalability_objective_effects,
    }
    for filename, summary in summaries.items():
        summary.to_csv(
            output_directory / filename,
            index=False,
            float_format="%.12g",
        )

    plot_budget_trajectories(
        capacity_trajectories,
        "capacity_regime",
        {"tight": "Tight", "intermediate": "Intermediate", "large": "Large"},
        output_directory / "cfl_capacity_trajectories.png",
    )
    plot_budget_trajectories(
        opening_cost_trajectories,
        "opening_cost_level",
        {
            7500: "7,500",
            12500: "12,500",
            17500: "17,500",
            25000: "25,000",
        },
        output_directory / "cfl_opening_cost_trajectories.png",
    )
    plot_scalability_runtime(
        scalability_runtime,
        output_directory / "cfl_scalability_runtime.png",
    )
    plot_scalability_objective_effects(
        scalability_objective_effects,
        output_directory / "cfl_scalability_objective_effects.png",
    )


if __name__ == "__main__":
    main()
