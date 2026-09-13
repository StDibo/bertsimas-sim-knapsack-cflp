"""Validation and analysis of experimental CFLP results."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


EXPECTED_GAMMAS = {1.0, 2.0, 3.0, 5.0, 7.0}

REQUIRED_COLUMNS = {
    "instance_id",
    "model_type",
    "facility_count",
    "customer_count",
    "seed",
    "gamma",
    "gamma_rel",
    "objective_value",
    "runtime_sec",
    "status",
    "open_facilities",
    "assignments",
    "empirical_infeasibility_rate",
    "simulation_scenarios",
    "simulation_seed",
    "best_bound",
    "mip_gap",
}


def load_results(file_path: str | Path) -> pd.DataFrame:
    """Load a CFLP results CSV and validate its minimum schema."""

    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Results file not found: {path}")

    data = pd.read_csv(path, keep_default_na=False)
    data.columns = data.columns.str.strip().str.lower()
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(
            "Outdated CSV schema; missing columns: "
            + ", ".join(sorted(missing))
        )

    numeric_columns = [
        "facility_count",
        "customer_count",
        "seed",
        "gamma",
        "gamma_rel",
        "objective_value",
        "runtime_sec",
        "empirical_infeasibility_rate",
        "simulation_scenarios",
        "simulation_seed",
        "best_bound",
        "mip_gap",
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["model_type"] = data["model_type"].str.strip().str.lower()
    data["status"] = data["status"].str.strip().str.upper()
    return data


def parse_open_facilities(value: object) -> set[int]:
    """Parse the semicolon-separated facility indices stored in the CSV."""

    text = str(value).strip()
    if not text:
        return set()
    return {int(item) for item in text.split(";") if item}


def parse_assignments(value: object) -> dict[int, int]:
    """Parse the customer-to-facility assignments stored in the CSV."""

    text = str(value).strip()
    if not text:
        return {}
    assignments: dict[int, int] = {}
    for pair in text.split(";"):
        customer, facility = pair.split(":", maxsplit=1)
        assignments[int(customer)] = int(facility)
    return assignments


def optimal_complete_results(data: pd.DataFrame) -> pd.DataFrame:
    """Select instances optimal in the nominal and at every robust budget."""

    nominal = data[
        (data["model_type"] == "nominal") & (data["status"] == "OPTIMAL")
    ]
    robust = data[
        (data["model_type"] == "robust") & (data["status"] == "OPTIMAL")
    ]
    nominal_ids = {
        instance_id
        for instance_id, group in nominal.groupby("instance_id")
        if len(group) == 1
    }
    complete_ids = {
        instance_id
        for instance_id, group in robust.groupby("instance_id")
        if set(group["gamma"]) == EXPECTED_GAMMAS
        and not group["gamma"].duplicated().any()
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
    """Compute cost, location, assignment, and feasibility metrics."""

    complete = optimal_complete_results(data)
    nominal = complete[complete["model_type"] == "nominal"][
        [
            "instance_id",
            "objective_value",
            "open_facilities",
            "assignments",
            "empirical_infeasibility_rate",
        ]
    ].rename(
        columns={
            "objective_value": "nominal_objective",
            "open_facilities": "nominal_open_facilities",
            "assignments": "nominal_assignments",
            "empirical_infeasibility_rate": "nominal_infeasibility_rate",
        }
    )
    robust = complete[complete["model_type"] == "robust"].merge(
        nominal, on="instance_id", how="inner", validate="many_to_one"
    )
    robust["price_of_robustness_pct"] = (
        (robust["objective_value"] - robust["nominal_objective"])
        / robust["nominal_objective"]
        * 100
    )
    robust["infeasibility_reduction_pp"] = (
        robust["nominal_infeasibility_rate"]
        - robust["empirical_infeasibility_rate"]
    ) * 100

    def structural_metrics(row: pd.Series) -> pd.Series:
        nominal_facilities = parse_open_facilities(
            row["nominal_open_facilities"]
        )
        robust_facilities = parse_open_facilities(row["open_facilities"])
        facility_union = nominal_facilities.union(robust_facilities)
        facility_difference = nominal_facilities.symmetric_difference(
            robust_facilities
        )

        nominal_assignments = parse_assignments(row["nominal_assignments"])
        robust_assignments = parse_assignments(row["assignments"])
        expected_customers = set(range(int(row["customer_count"])))
        if set(nominal_assignments) != expected_customers:
            raise ValueError("nominal assignments are incomplete")
        if set(robust_assignments) != expected_customers:
            raise ValueError("robust assignments are incomplete")
        changed_assignments = sum(
            nominal_assignments[j] != robust_assignments[j]
            for j in expected_customers
        )

        return pd.Series(
            {
                "robust_open_count": len(robust_facilities),
                "nominal_open_count": len(nominal_facilities),
                "open_count_change": (
                    len(robust_facilities) - len(nominal_facilities)
                ),
                "facility_hamming_normalized": len(facility_difference)
                / row["facility_count"],
                "facility_jaccard_similarity": (
                    len(
                        nominal_facilities.intersection(robust_facilities)
                    )
                    / len(facility_union)
                    if facility_union
                    else 1.0
                ),
                "assignment_change_rate": changed_assignments
                / row["customer_count"],
            }
        )

    return pd.concat([robust, robust.apply(structural_metrics, axis=1)], axis=1)


def monotonicity_violations(
    comparisons: pd.DataFrame, tolerance: float = 1e-6
) -> pd.DataFrame:
    """Return objective decreases beyond the preceding solver bound."""

    violations: list[dict[str, object]] = []
    for instance_id, group in comparisons.groupby("instance_id"):
        ordered = group.sort_values("gamma")
        previous = None
        for row in ordered.itertuples(index=False):
            if (
                previous is not None
                and row.objective_value < previous.best_bound - tolerance
            ):
                violations.append(
                    {
                        "instance_id": instance_id,
                        "gamma_previous": previous.gamma,
                        "gamma_current": row.gamma,
                        "best_bound_previous": previous.best_bound,
                        "objective_current": row.objective_value,
                    }
                )
            previous = row
    return pd.DataFrame(
        violations,
        columns=[
            "instance_id",
            "gamma_previous",
            "gamma_current",
            "best_bound_previous",
            "objective_current",
        ],
    )


def size_label(data: pd.DataFrame) -> pd.Series:
    """Return compact labels for facility/customer size pairs."""

    return (
        data["facility_count"].astype(int).astype(str)
        + "x"
        + data["customer_count"].astype(int).astype(str)
    )


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

    configuration_summary = comparisons.groupby(
        ["facility_count", "customer_count", "gamma"], as_index=False
    ).agg(
        included_seeds=("seed", "nunique"),
        robust_objective_mean=("objective_value", "mean"),
        price_of_robustness_mean=("price_of_robustness_pct", "mean"),
        open_facilities_mean=("robust_open_count", "mean"),
        open_count_change_mean=("open_count_change", "mean"),
        facility_hamming_mean=("facility_hamming_normalized", "mean"),
        facility_jaccard_mean=("facility_jaccard_similarity", "mean"),
        assignment_change_mean=("assignment_change_rate", "mean"),
        empirical_infeasibility_mean=(
            "empirical_infeasibility_rate",
            "mean",
        ),
        infeasibility_reduction_pp_mean=(
            "infeasibility_reduction_pp",
            "mean",
        ),
        robust_runtime_mean=("runtime_sec", "mean"),
        runtime_ratio_mean=("runtime_ratio", "mean"),
    )
    configuration_summary.to_csv(
        output / "configuration_summary.csv", index=False
    )

    budget_summary = comparisons.groupby(
        ["facility_count", "customer_count", "gamma"], as_index=False
    ).agg(
        gamma_rel=("gamma_rel", "first"),
        price_of_robustness_mean=("price_of_robustness_pct", "mean"),
        open_facilities_mean=("robust_open_count", "mean"),
        open_count_change_mean=("open_count_change", "mean"),
        facility_hamming_mean=("facility_hamming_normalized", "mean"),
        facility_jaccard_mean=("facility_jaccard_similarity", "mean"),
        assignment_change_mean=("assignment_change_rate", "mean"),
        empirical_infeasibility_mean=(
            "empirical_infeasibility_rate",
            "mean",
        ),
        infeasibility_reduction_pp_mean=(
            "infeasibility_reduction_pp",
            "mean",
        ),
        included_instances=("instance_id", "nunique"),
    )
    nominal_rates = comparisons[
        [
            "instance_id",
            "facility_count",
            "customer_count",
            "nominal_infeasibility_rate",
            "nominal_open_count",
        ]
    ].drop_duplicates("instance_id")
    nominal_baseline = nominal_rates.groupby(
        ["facility_count", "customer_count"], as_index=False
    ).agg(
        empirical_infeasibility_mean=(
            "nominal_infeasibility_rate",
            "mean",
        ),
        included_instances=("instance_id", "nunique"),
        open_facilities_mean=("nominal_open_count", "mean"),
    )
    nominal_baseline["gamma"] = 0.0
    nominal_baseline["gamma_rel"] = 0.0
    nominal_baseline["price_of_robustness_mean"] = 0.0
    nominal_baseline["open_count_change_mean"] = 0.0
    nominal_baseline["facility_hamming_mean"] = 0.0
    nominal_baseline["facility_jaccard_mean"] = 1.0
    nominal_baseline["assignment_change_mean"] = 0.0
    nominal_baseline["infeasibility_reduction_pp_mean"] = 0.0
    budget_summary = pd.concat(
        [nominal_baseline[budget_summary.columns], budget_summary],
        ignore_index=True,
    ).sort_values(["facility_count", "customer_count", "gamma"])
    budget_summary.to_csv(output / "budget_summary.csv", index=False)

    robust_runs = data[data["model_type"] == "robust"].copy()
    solver_summary = robust_runs.groupby(
        ["facility_count", "customer_count", "gamma"], as_index=False
    ).agg(
        total_runs=("instance_id", "size"),
        optimal_runs=("status", lambda values: (values == "OPTIMAL").sum()),
        time_limit_runs=(
            "status",
            lambda values: (values == "TIME_LIMIT").sum(),
        ),
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
        .groupby(
            ["facility_count", "customer_count", "gamma"], as_index=False
        )
        .agg(
            mip_gap_mean=("mip_gap", "mean"),
            mip_gap_max=("mip_gap", "max"),
        )
    )
    solver_summary = solver_summary.merge(
        time_limit_gaps,
        on=["facility_count", "customer_count", "gamma"],
        how="left",
    )
    solver_summary["mip_gap_mean_pct"] = solver_summary.pop(
        "mip_gap_mean"
    ) * 100
    solver_summary["mip_gap_max_pct"] = solver_summary.pop(
        "mip_gap_max"
    ) * 100
    solver_summary.to_csv(output / "solver_summary.csv", index=False)

    nominal_runs = data[data["model_type"] == "nominal"][
        ["instance_id", "runtime_sec"]
    ].rename(columns={"runtime_sec": "nominal_runtime_sec"})
    robust_runtime = robust_runs.merge(
        nominal_runs, on="instance_id", how="inner", validate="many_to_one"
    )
    robust_runtime["runtime_ratio"] = robust_runtime["runtime_sec"] / (
        robust_runtime["nominal_runtime_sec"].where(
            robust_runtime["nominal_runtime_sec"] > 0
        )
    )
    scalability = robust_runtime.groupby(
        ["facility_count", "customer_count", "gamma"], as_index=False
    ).agg(
        robust_runtime_mean=("runtime_sec", "mean"),
        robust_runtime_median=("runtime_sec", "median"),
        robust_runtime_max=("runtime_sec", "max"),
        nominal_runtime_mean=("nominal_runtime_sec", "mean"),
        nominal_runtime_median=("nominal_runtime_sec", "median"),
        runtime_ratio_mean=("runtime_ratio", "mean"),
        included_pairs=("instance_id", "nunique"),
    )
    scalability.to_csv(output / "scalability_summary.csv", index=False)

    budget_summary = budget_summary.copy()
    budget_summary["size"] = size_label(budget_summary)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for label, group in budget_summary.groupby("size"):
        ordered = group.sort_values("gamma")
        axes[0].plot(
            ordered["gamma"],
            ordered["price_of_robustness_mean"],
            marker="o",
            label=label,
        )
        axes[1].plot(
            ordered["gamma"],
            ordered["empirical_infeasibility_mean"] * 100,
            marker="o",
            label=label,
        )
    axes[0].set(
        title="Cost of protection",
        xlabel=r"Capacity-row budget $\Gamma_i^d$",
        ylabel="Mean Price of Robustness (%)",
    )
    axes[1].set(
        title="Adverse-demand stress test",
        xlabel=r"Capacity-row budget $\Gamma_i^d$",
        ylabel="Mean infeasible scenarios (%)",
    )
    for axis in axes:
        axis.grid(True, linestyle="--", alpha=0.5)
        axis.legend(title="Facilities x customers")
    fig.tight_layout()
    fig.savefig(output / "budget_sensitivity.png", dpi=300)
    plt.close(fig)

    fig, axes_grid = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes_grid.ravel()
    for label, group in budget_summary.groupby("size"):
        ordered = group.sort_values("gamma")
        axes[0].plot(
            ordered["gamma"],
            ordered["open_facilities_mean"],
            marker="o",
            label=label,
        )
        axes[1].plot(
            ordered["gamma"],
            ordered["facility_hamming_mean"],
            marker="o",
            label=label,
        )
        axes[2].plot(
            ordered["gamma"],
            ordered["facility_jaccard_mean"],
            marker="o",
            label=label,
        )
        axes[3].plot(
            ordered["gamma"],
            ordered["assignment_change_mean"] * 100,
            marker="o",
            label=label,
        )
    axes[0].set(
        title="Number of open facilities",
        xlabel=r"Capacity-row budget $\Gamma_i^d$",
        ylabel="Mean open facilities",
    )
    axes[1].set(
        title="Open-facility changes",
        xlabel=r"Capacity-row budget $\Gamma_i^d$",
        ylabel="Mean normalized Hamming distance",
    )
    axes[2].set(
        title="Open-facility similarity",
        xlabel=r"Capacity-row budget $\Gamma_i^d$",
        ylabel="Mean Jaccard similarity",
    )
    axes[3].set(
        title="Changes in customer assignments",
        xlabel=r"Capacity-row budget $\Gamma_i^d$",
        ylabel="Mean changed assignments (%)",
    )
    for axis in axes:
        axis.grid(True, linestyle="--", alpha=0.5)
        axis.legend(title="Facilities x customers")
    fig.tight_layout()
    fig.savefig(output / "solution_structure.png", dpi=300)
    plt.close(fig)

    tradeoff = budget_summary[budget_summary["gamma"] > 0]
    fig, axis = plt.subplots(figsize=(8, 5))
    for label, group in tradeoff.groupby("size"):
        ordered = group.sort_values("gamma")
        axis.plot(
            ordered["price_of_robustness_mean"],
            ordered["empirical_infeasibility_mean"] * 100,
            marker="o",
            label=label,
        )
    axis.set(
        title="Cost--infeasibility trade-off by instance size",
        xlabel="Mean Price of Robustness (%)",
        ylabel="Mean infeasible scenarios (%)",
    )
    axis.grid(True, linestyle="--", alpha=0.5)
    axis.legend(title="Facilities x customers")
    fig.tight_layout()
    fig.savefig(output / "cost_infeasibility_tradeoff.png", dpi=300)
    plt.close(fig)

    scalability = scalability.copy()
    scalability["size"] = size_label(scalability)
    fig, axis = plt.subplots(figsize=(9, 5))
    for label, group in scalability.groupby("size"):
        ordered = group.sort_values("gamma")
        axis.plot(
            ordered["gamma"],
            ordered["robust_runtime_mean"],
            marker="o",
            label=label,
        )
    axis.set(
        title="Runtime of the robust formulation",
        xlabel=r"Capacity-row budget $\Gamma_i^d$",
        ylabel="Mean runtime (seconds)",
    )
    axis.grid(True, linestyle="--", alpha=0.5)
    axis.legend(title="Facilities x customers")
    fig.tight_layout()
    fig.savefig(output / "runtime_scalability.png", dpi=300)
    plt.close(fig)

    time_limit_count = int((robust_runs["status"] == "TIME_LIMIT").sum())
    print(f"Robust models reaching the time limit: {time_limit_count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("cfl_analysis"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_analysis(load_results(args.csv_path), args.output_dir)


if __name__ == "__main__":
    main()
