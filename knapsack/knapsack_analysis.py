"""Validation and analysis of experimental Knapsack results."""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


EXPECTED_GAMMA_RELS = {0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00}

REQUIRED_COLUMNS = {
    "instance_id",
    "model_type",
    "n",
    "delta",
    "capacity_ratio",
    "seed",
    "gamma_rel",
    "objective_value",
    "runtime_sec",
    "status",
    "selected_items",
    "best_bound",
    "mip_gap",
}

def load_results(file_path: str | Path) -> pd.DataFrame:
    """Load the new CSV file and validate its minimum schema."""

    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Results file not found: {path}")

    data = pd.read_csv(path, keep_default_na=False)
    data.columns = data.columns.str.strip().str.lower()
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(
            "Outdated CSV schema; missing columns: " + ", ".join(sorted(missing))
        )

    numeric_columns = [
        "n",
        "delta",
        "capacity_ratio",
        "seed",
        "gamma_rel",
        "objective_value",
        "runtime_sec",
        "best_bound",
        "mip_gap",
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["model_type"] = data["model_type"].str.strip().str.lower()
    data["status"] = data["status"].str.strip().str.upper()
    return data


def parse_selected_items(value: object) -> set[int]:
    """Parse the semicolon-separated representation stored in the CSV file."""

    text = str(value).strip()
    if not text:
        return set()
    return {int(item) for item in text.split(";") if item != ""}


def optimal_complete_results(data: pd.DataFrame) -> pd.DataFrame:
    """Select instances solved optimally in the nominal and at every budget."""

    nominal = data[(data["model_type"] == "nominal") & (data["status"] == "OPTIMAL")]
    robust = data[(data["model_type"] == "robust") & (data["status"] == "OPTIMAL")]
    nominal_ids = {
        instance_id
        for instance_id, group in nominal.groupby("instance_id")
        if len(group) == 1
    }
    complete_ids = {
        instance_id
        for instance_id, group in robust.groupby("instance_id")
        if set(group["gamma_rel"]) == EXPECTED_GAMMA_RELS
        and not group["gamma_rel"].duplicated().any()
    }
    included_ids = nominal_ids.intersection(complete_ids)
    return pd.concat(
        [
            nominal[nominal["instance_id"].isin(included_ids)],
            robust[robust["instance_id"].isin(included_ids)],
        ],
        ignore_index=True,
    )


def add_comparison_metrics(data: pd.DataFrame) -> pd.DataFrame:
    """Match the true nominal and compute PoR, Hamming, and Jaccard per instance."""
    
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
    """Return increases that exceed the preceding solver upper bound."""
    
    violations: list[dict[str, object]] = []
    for instance_id, group in comparisons.groupby("instance_id"):
        ordered = group.sort_values("gamma_rel")
        previous = None
        for row in ordered.itertuples(index=False):
            if (
                previous is not None
                and row.objective_value > previous.best_bound + tolerance
            ):
                violations.append(
                    {
                        "instance_id": instance_id,
                        "gamma_rel_previous": previous.gamma_rel,
                        "gamma_rel_current": row.gamma_rel,
                        "best_bound_previous": previous.best_bound,
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
            "best_bound_previous",
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
    axis.set(title=title, xlabel=r"Relative budget $\Gamma_{rel}$", ylabel=ylabel)
    axis.grid(True, linestyle="--", alpha=0.5)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def create_analysis(data: pd.DataFrame, output_directory: str | Path) -> None:
    """Validate complete instances and produce summary tables and plots."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    comparisons = add_comparison_metrics(data)
    included = comparisons["instance_id"].nunique()
    total = data["instance_id"].nunique()
    print(f"Total instances: {total}")
    print(f"Instances included in complete optimal curves: {included}")
    if comparisons.empty:
        raise ValueError("No complete instance solved to optimality is available")

    nominal_runtime = data[
        (data["model_type"] == "nominal") & (data["status"] == "OPTIMAL")
    ][["instance_id", "runtime_sec"]].rename(
        columns={"runtime_sec": "nominal_runtime_sec"}
    )
    comparisons = comparisons.merge(
        nominal_runtime, on="instance_id", how="left", validate="many_to_one"
    )
    comparisons["runtime_ratio"] = comparisons["runtime_sec"] / comparisons[
        "nominal_runtime_sec"
    ].where(comparisons["nominal_runtime_sec"] > 0)

    violations = monotonicity_violations(comparisons)
    violations.to_csv(output / "monotonicity_violations.csv", index=False)
    print(f"Definite monotonicity violations: {len(violations)}")

    configuration_metrics = comparisons.groupby(
        ["n", "delta", "capacity_ratio", "gamma_rel"], as_index=False
    ).agg(
        included_seeds=("seed", "nunique"),
        robust_objective_mean=("objective_value", "mean"),
        robust_objective_std=("objective_value", "std"),
        price_of_robustness_mean=("price_of_robustness_pct", "mean"),
        price_of_robustness_std=("price_of_robustness_pct", "std"),
        hamming_normalized_mean=("hamming_distance_normalized", "mean"),
        jaccard_mean=("jaccard_similarity", "mean"),
        robust_runtime_mean=("runtime_sec", "mean"),
        runtime_ratio_mean=("runtime_ratio", "mean"),
    )
    base_configurations = data[data["model_type"] == "nominal"][
        ["n", "delta", "capacity_ratio"]
    ].drop_duplicates()
    budget_levels = pd.DataFrame({"gamma_rel": sorted(EXPECTED_GAMMA_RELS)})
    configuration_summary = base_configurations.merge(
        budget_levels, how="cross"
    ).merge(
        configuration_metrics,
        on=["n", "delta", "capacity_ratio", "gamma_rel"],
        how="left",
    )
    configuration_summary["included_seeds"] = (
        configuration_summary["included_seeds"].fillna(0).astype(int)
    )
    configuration_summary = configuration_summary.sort_values(
        ["n", "delta", "capacity_ratio", "gamma_rel"]
    )
    configuration_summary.to_csv(output / "configuration_summary.csv", index=False)
    configurations_without_complete_seeds = (
        configuration_summary.groupby(["n", "delta", "capacity_ratio"])[
            "included_seeds"
        ].max()
        == 0
    ).sum()
    print(
        "Configurations without complete optimal seeds: "
        f"{configurations_without_complete_seeds}"
    )

    budget_summary = comparisons.groupby("gamma_rel", as_index=False).agg(
        price_of_robustness_mean=("price_of_robustness_pct", "mean"),
        hamming_normalized_mean=("hamming_distance_normalized", "mean"),
        jaccard_mean=("jaccard_similarity", "mean"),
        included_instances=("instance_id", "nunique"),
    )
    nominal_baseline = pd.DataFrame(
        {
            "gamma_rel": [0.0],
            "price_of_robustness_mean": [0.0],
            "hamming_normalized_mean": [0.0],
            "jaccard_mean": [1.0],
            "included_instances": [included],
        }
    )
    budget_summary = pd.concat(
        [nominal_baseline, budget_summary], ignore_index=True
    ).sort_values("gamma_rel")
    budget_summary.to_csv(output / "budget_summary.csv", index=False)

    for parameter, filename, title in (
        (
            "capacity_ratio",
            "capacity_ratio_impact.png",
            "Price of Robustness by capacity ratio",
        ),
        ("delta", "delta_impact.png", "Price of Robustness by maximum deviation"),
    ):
        summary = comparisons.groupby([parameter, "gamma_rel"], as_index=False)[
            "price_of_robustness_pct"
        ].mean()
        parameter_baseline = pd.DataFrame(
            {
                parameter: sorted(comparisons[parameter].unique()),
                "gamma_rel": 0.0,
                "price_of_robustness_pct": 0.0,
            }
        )
        summary = pd.concat([parameter_baseline, summary], ignore_index=True)
        save_line_plot(
            summary,
            "gamma_rel",
            parameter,
            "price_of_robustness_pct",
            "Mean Price of Robustness (%)",
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
        title="Sensitivity to the uncertainty budget",
        xlabel=r"Relative budget $\Gamma_{rel}$",
        ylabel="Mean Price of Robustness (%)",
    )
    axis.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output / "budget_sensitivity.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].plot(
        budget_summary["gamma_rel"],
        budget_summary["hamming_normalized_mean"],
        marker="o",
    )
    axes[0].set(
        title="Normalized Hamming distance",
        xlabel=r"Relative budget $\Gamma_{rel}$",
        ylabel="Mean normalized distance",
    )
    axes[1].plot(
        budget_summary["gamma_rel"],
        budget_summary["jaccard_mean"],
        marker="o",
    )
    axes[1].set(
        title="Jaccard similarity",
        xlabel=r"Relative budget $\Gamma_{rel}$",
        ylabel="Mean similarity",
    )
    for axis in axes:
        axis.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output / "solution_structure.png", dpi=300)
    plt.close(fig)

    robust_runs = data[data["model_type"] == "robust"].copy()
    solver_summary = robust_runs.groupby(["n", "gamma_rel"], as_index=False).agg(
        total_runs=("instance_id", "size"),
        optimal_runs=("status", lambda values: (values == "OPTIMAL").sum()),
        time_limit_runs=("status", lambda values: (values == "TIME_LIMIT").sum()),
    )
    solver_summary["time_limit_pct"] = (
        solver_summary["time_limit_runs"] / solver_summary["total_runs"] * 100
    )
    solver_summary["other_status_runs"] = (
        solver_summary["total_runs"]
        - solver_summary["optimal_runs"]
        - solver_summary["time_limit_runs"]
    )
    time_limit_gaps = (
        robust_runs[robust_runs["status"] == "TIME_LIMIT"]
        .groupby(["n", "gamma_rel"], as_index=False)
        .agg(
            mip_gap_mean=("mip_gap", "mean"),
            mip_gap_median=("mip_gap", "median"),
            mip_gap_max=("mip_gap", "max"),
        )
    )
    solver_summary = solver_summary.merge(
        time_limit_gaps, on=["n", "gamma_rel"], how="left"
    )
    solver_summary = solver_summary.rename(
        columns={
            "mip_gap_mean": "mip_gap_mean_pct",
            "mip_gap_median": "mip_gap_median_pct",
            "mip_gap_max": "mip_gap_max_pct",
        }
    )
    for column in ("mip_gap_mean_pct", "mip_gap_median_pct", "mip_gap_max_pct"):
        solver_summary[column] *= 100
    solver_summary.to_csv(output / "solver_summary.csv", index=False)

    nominal_runs = data[data["model_type"] == "nominal"][
        ["instance_id", "runtime_sec"]
    ].rename(columns={"runtime_sec": "nominal_runtime_sec"})
    robust_runtime = robust_runs.merge(
        nominal_runs, on="instance_id", how="inner", validate="many_to_one"
    )
    robust_runtime["runtime_ratio"] = (
        robust_runtime["runtime_sec"]
        / robust_runtime["nominal_runtime_sec"].where(
            robust_runtime["nominal_runtime_sec"] > 0
        )
    )
    scalability = robust_runtime.groupby(["n", "gamma_rel"], as_index=False).agg(
        robust_runtime_mean=("runtime_sec", "mean"),
        nominal_runtime_mean=("nominal_runtime_sec", "mean"),
        runtime_ratio_mean=("runtime_ratio", "mean"),
        included_pairs=("instance_id", "nunique"),
    )
    scalability.to_csv(output / "scalability_summary.csv", index=False)

    save_line_plot(
        scalability,
        "gamma_rel",
        "n",
        "runtime_ratio_mean",
        "Mean robust/nominal runtime ratio",
        "Computational cost of the robust formulation",
        output / "runtime_ratio_scalability.png",
    )

    time_limit_count = int((robust_runs["status"] == "TIME_LIMIT").sum())
    print(f"Robust models reaching the time limit: {time_limit_count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("knapsack_analysis"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_analysis(load_results(args.csv_path), args.output_dir)


if __name__ == "__main__":
    main()
