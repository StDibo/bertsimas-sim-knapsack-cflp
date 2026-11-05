"""Create the Knapsack summaries and figures reported in Chapter 5."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


OPTIMAL_STATUS = "OPTIMAL"
TIME_LIMIT_STATUS = "TIME_LIMIT"
REPORTED_RUNTIME_BUDGETS = (0.05, 0.20, 0.40, 1.00)


def selected_item_set(value: object) -> set[int]:
    """Convert the semicolon-separated item indices to a set."""

    if pd.isna(value) or value == "":
        return set()
    return {int(item) for item in str(value).split(";")}


def calculate_comparison_metrics(results: pd.DataFrame) -> pd.DataFrame:
    """Compute PoR and solution similarity on the complete optimal panel."""

    robust_results = results[results["model_type"] == "robust"].copy()
    complete_instances = robust_results.groupby("instance_id")["status"].apply(
        lambda statuses: (statuses == OPTIMAL_STATUS).all()
    )
    included_ids = complete_instances[complete_instances].index

    nominal_results = results[
        (results["model_type"] == "nominal")
        & results["instance_id"].isin(included_ids)
    ][["instance_id", "objective_value", "selected_items"]].rename(
        columns={
            "objective_value": "nominal_objective",
            "selected_items": "nominal_selected_items",
        }
    )
    comparisons = robust_results[
        robust_results["instance_id"].isin(included_ids)
    ].merge(nominal_results, on="instance_id")
    comparisons["price_of_robustness_pct"] = 100 * (
        comparisons["nominal_objective"] - comparisons["objective_value"]
    ) / comparisons["nominal_objective"]

    def solution_metrics(row: pd.Series) -> pd.Series:
        nominal_items = selected_item_set(row["nominal_selected_items"])
        robust_items = selected_item_set(row["selected_items"])
        union = nominal_items | robust_items
        return pd.Series(
            {
                "hamming_pct": 100
                * len(nominal_items ^ robust_items)
                / row["n"],
                "selected_fraction": len(robust_items) / row["n"],
                "jaccard_similarity": (
                    len(nominal_items & robust_items) / len(union)
                    if union
                    else 1.0
                ),
            }
        )

    return pd.concat(
        [comparisons, comparisons.apply(solution_metrics, axis=1)], axis=1
    )


def summarize_budget(comparisons: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the three comparison metrics by uncertainty budget."""

    summary = comparisons.groupby("gamma_rel", as_index=False).agg(
        mean_por_pct=("price_of_robustness_pct", "mean"),
        mean_hamming_pct=("hamming_pct", "mean"),
        mean_jaccard=("jaccard_similarity", "mean"),
    )
    nominal_baseline = pd.DataFrame(
        {
            "gamma_rel": [0.0],
            "mean_por_pct": [0.0],
            "mean_hamming_pct": [0.0],
            "mean_jaccard": [1.0],
        }
    )
    return pd.concat([nominal_baseline, summary], ignore_index=True).sort_values(
        "gamma_rel"
    )


def summarize_capacity_saturation(comparisons: pd.DataFrame) -> pd.DataFrame:
    """Measure when each capacity-ratio group reaches full protection."""

    data = comparisons.copy()
    data["exceeds_budget"] = data["selected_fraction"] > data["gamma_rel"]
    summary = data.groupby(
        ["capacity_ratio", "gamma_rel"], as_index=False
    ).agg(
        included_solutions=("instance_id", "size"),
        max_selected_fraction=("selected_fraction", "max"),
        solutions_exceeding_budget=("exceeds_budget", "sum"),
    )
    summary["all_solutions_saturated"] = (
        summary["solutions_exceeding_budget"] == 0
    )
    return summary


def summarize_sample(
    results: pd.DataFrame, comparisons: pd.DataFrame
) -> pd.DataFrame:
    """Describe the instances retained in the complete optimal panel."""

    included = comparisons.drop_duplicates("instance_id")
    records: list[dict[str, object]] = [
        {
            "category": "complete_instances",
            "value": "all",
            "count": included["instance_id"].nunique(),
        }
    ]
    for parameter in ("n", "delta", "capacity_ratio"):
        for value, count in included[parameter].value_counts().sort_index().items():
            records.append(
                {"category": parameter, "value": value, "count": count}
            )

    all_configurations = results[results["model_type"] == "nominal"][
        ["n", "delta", "capacity_ratio"]
    ].drop_duplicates()
    included_configurations = included[
        ["n", "delta", "capacity_ratio"]
    ].drop_duplicates()
    empty_configurations = all_configurations.merge(
        included_configurations,
        on=["n", "delta", "capacity_ratio"],
        how="left",
        indicator=True,
    )
    records.append(
        {
            "category": "empty_parameter_combinations",
            "value": "all",
            "count": (empty_configurations["_merge"] == "left_only").sum(),
        }
    )
    return pd.DataFrame(records)


def summarize_runtime(results: pd.DataFrame) -> pd.DataFrame:
    """Compute the mean runtime by size for nominal and robust models."""

    nominal = (
        results[results["model_type"] == "nominal"]
        .groupby("n")["runtime_sec"]
        .mean()
        .rename("nominal_runtime_sec")
    )
    robust = (
        results[
            (results["model_type"] == "robust")
            & results["gamma_rel"].isin(REPORTED_RUNTIME_BUDGETS)
        ]
        .groupby(["n", "gamma_rel"])["runtime_sec"]
        .mean()
        .unstack()
        .rename(columns=lambda gamma: f"gamma_rel_{gamma:g}")
    )
    return nominal.to_frame().join(robust).reset_index()


def summarize_time_limits(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize time-limit frequency and final MIP gaps by size."""

    robust = results[results["model_type"] == "robust"].copy()

    def summarize(group: pd.DataFrame) -> pd.Series:
        limited = group[group["status"] == TIME_LIMIT_STATUS]
        return pd.Series(
            {
                "total_runs": len(group),
                "time_limit_runs": len(limited),
                "time_limit_pct": 100 * len(limited) / len(group),
                "mean_mip_gap_pct": 100 * limited["mip_gap"].mean(),
                "max_mip_gap_pct": 100 * limited["mip_gap"].max(),
            }
        )

    by_size = robust.groupby("n").apply(summarize, include_groups=False).reset_index()
    overall = summarize(robust).to_frame().T
    overall.insert(0, "n", "overall")
    summary = pd.concat([by_size, overall], ignore_index=True)
    summary[["total_runs", "time_limit_runs"]] = summary[
        ["total_runs", "time_limit_runs"]
    ].astype(int)
    return summary


def save_budget_plot(summary: pd.DataFrame, output_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.plot(summary["gamma_rel"], summary["mean_por_pct"], marker="o")
    axis.set(
        title="Sensitivity to the uncertainty budget",
        xlabel=r"Relative budget $\Gamma_{rel}$",
        ylabel="Mean Price of Robustness (%)",
    )
    axis.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_factor_plot(
    comparisons: pd.DataFrame,
    parameter: str,
    title: str,
    output_path: Path,
) -> None:
    summary = comparisons.groupby([parameter, "gamma_rel"], as_index=False)[
        "price_of_robustness_pct"
    ].mean()
    baseline = pd.DataFrame(
        {
            parameter: sorted(comparisons[parameter].unique()),
            "gamma_rel": 0.0,
            "price_of_robustness_pct": 0.0,
        }
    )
    summary = pd.concat([baseline, summary], ignore_index=True)

    fig, axis = plt.subplots(figsize=(9, 5))
    for value, group in summary.groupby(parameter):
        ordered = group.sort_values("gamma_rel")
        axis.plot(
            ordered["gamma_rel"],
            ordered["price_of_robustness_pct"],
            marker="o",
            label=f"{parameter}={value}",
        )
    axis.set(
        title=title,
        xlabel=r"Relative budget $\Gamma_{rel}$",
        ylabel="Mean Price of Robustness (%)",
    )
    axis.grid(True, linestyle="--", alpha=0.5)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_solution_structure_plot(
    summary: pd.DataFrame, output_path: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].plot(
        summary["gamma_rel"], summary["mean_hamming_pct"], marker="o"
    )
    axes[0].set(
        title="Normalized Hamming distance",
        xlabel=r"Relative budget $\Gamma_{rel}$",
        ylabel="Mean changed items (%)",
    )
    axes[1].plot(summary["gamma_rel"], summary["mean_jaccard"], marker="o")
    axes[1].set(
        title="Jaccard similarity",
        xlabel=r"Relative budget $\Gamma_{rel}$",
        ylabel="Mean similarity",
    )
    for axis in axes:
        axis.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    directory = Path(__file__).resolve().parent
    output_directory = directory / "knapsack_analysis"
    output_directory.mkdir(exist_ok=True)

    results = pd.read_csv(directory / "knapsack_results.csv")
    comparisons = calculate_comparison_metrics(results)
    budget_summary = summarize_budget(comparisons)

    budget_summary.to_csv(output_directory / "budget_summary.csv", index=False)
    summarize_capacity_saturation(comparisons).to_csv(
        output_directory / "capacity_saturation.csv", index=False
    )
    summarize_sample(results, comparisons).to_csv(
        output_directory / "sample_composition.csv", index=False
    )
    summarize_runtime(results).to_csv(
        output_directory / "runtime_summary.csv", index=False
    )
    summarize_time_limits(results).to_csv(
        output_directory / "time_limit_summary.csv", index=False
    )

    save_budget_plot(budget_summary, output_directory / "budget_sensitivity.png")
    save_factor_plot(
        comparisons,
        "delta",
        "Price of Robustness by maximum deviation",
        output_directory / "delta_impact.png",
    )
    save_factor_plot(
        comparisons,
        "capacity_ratio",
        "Price of Robustness by capacity ratio",
        output_directory / "capacity_ratio_impact.png",
    )
    save_solution_structure_plot(
        budget_summary, output_directory / "solution_structure.png"
    )


if __name__ == "__main__":
    main()
