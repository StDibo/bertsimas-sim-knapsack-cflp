"""Create the CFLP summaries reported in Chapter 5 of the thesis."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from cfl_generator import CFLInstance, load_orlib_instance


# Maps the last digit of the OR-Library instance name to its opening-cost level.
OPENING_COST_LEVELS = {1: 7500, 2: 12500, 3: 17500, 4: 25000}
FEASIBILITY_TOLERANCE = 1e-6


def index_set(value: object) -> set[int]:
    """Convert a semicolon-separated collection of indices to a set."""
    if pd.isna(value) or value == "":
        return set()
    return {int(index) for index in str(value).split(";")}


def allocation_map(value: object) -> dict[tuple[int, int], float]:
    """Convert serialized positive allocations to an index-value mapping."""
    if pd.isna(value) or value == "":
        return {}
    allocations: dict[tuple[int, int], float] = {}
    for entry in str(value).split(";"):
        facility, customer, fraction = entry.split(":")
        allocations[(int(facility), int(customer))] = float(fraction)
    return allocations


def calculate_price_of_robustness(results: pd.DataFrame) -> pd.DataFrame:
    """Add nominal baselines and the Price of Robustness."""
    nominal = results[results["model_type"] == "nominal"][
        [
            "instance_id",
            "objective_value",
            "open_facilities",
            "allocation_fractions",
        ]
    ].rename(
        columns={
            "objective_value": "nominal_objective",
            "open_facilities": "nominal_open_facilities",
            "allocation_fractions": "nominal_allocation_fractions",
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

    robust["robust_open_count"] = robust["open_facilities"].map(
        lambda value: len(index_set(value))
    )

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


def load_benchmark_instances(
    results: pd.DataFrame, directory: Path
) -> dict[str, CFLInstance]:
    """Load the nominal OR-Library data used by the benchmark campaign."""

    instance_ids = results.loc[
        results["model_type"] == "nominal", "instance_id"
    ].unique()
    orlib_directory = directory / "or_library_files"
    return {
        instance_id: load_orlib_instance(
            orlib_directory / f"{instance_id}.txt",
            deviation_rates=0.0,
        )
        for instance_id in instance_ids
    }


def objective_components(
    instance: CFLInstance,
    open_facilities: object,
    allocation_fractions: object,
) -> tuple[float, float]:
    """Return the opening and assignment components of one objective."""

    opening_cost = sum(
        instance.opening_costs[i] for i in index_set(open_facilities)
    )
    assignment_cost = sum(
        instance.assignment_costs[i][j] * fraction
        for (i, j), fraction in allocation_map(allocation_fractions).items()
    )
    return opening_cost, assignment_cost


def add_objective_components(
    robust: pd.DataFrame,
    instances: dict[str, CFLInstance],
) -> pd.DataFrame:
    """Decompose the robust cost increase into opening and assignment costs."""

    data = robust.copy()
    nominal_components: dict[str, tuple[float, float]] = {}
    for row in data.drop_duplicates("instance_id").itertuples():
        nominal_components[row.instance_id] = objective_components(
            instances[row.instance_id],
            row.nominal_open_facilities,
            row.nominal_allocation_fractions,
        )

    robust_components = [
        objective_components(
            instances[row.instance_id],
            row.open_facilities,
            row.allocation_fractions,
        )
        for row in data.itertuples()
    ]
    data["opening_cost_change"] = [
        robust_opening - nominal_components[instance_id][0]
        for (robust_opening, _), instance_id in zip(
            robust_components, data["instance_id"]
        )
    ]
    data["assignment_cost_change"] = [
        robust_assignment - nominal_components[instance_id][1]
        for (_, robust_assignment), instance_id in zip(
            robust_components, data["instance_id"]
        )
    ]
    data["total_cost_change"] = (
        data["opening_cost_change"] + data["assignment_cost_change"]
    )
    nonoptimal = data["status"] != "OPTIMAL"
    data.loc[
        nonoptimal,
        [
            "opening_cost_change",
            "assignment_cost_change",
            "total_cost_change",
        ],
    ] = float("nan")
    return data


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
        .agg(mean_por_pct=("price_of_robustness_pct", "mean"))
        .reset_index()
        .sort_values(["gamma_rel", "delta"])
    )


def summarize_capacity_trajectories(robust: pd.DataFrame) -> pd.DataFrame:
    """Summarize budget effects by capacity regime on the common panel."""

    return (
        complete_panel(robust)
        .groupby(["capacity_regime", "delta", "gamma_rel"])
        .agg(mean_por_pct=("price_of_robustness_pct", "mean"))
        .reset_index()
    )


def summarize_opening_cost_trajectories(robust: pd.DataFrame) -> pd.DataFrame:
    """Summarize budget effects by opening-cost level on the common panel."""

    return (
        complete_panel(robust)
        .groupby(["opening_cost_level", "delta", "gamma_rel"])
        .agg(mean_por_pct=("price_of_robustness_pct", "mean"))
        .reset_index()
    )


def summarize_full_protection_feasibility(
    robust: pd.DataFrame,
    instances: dict[str, CFLInstance],
) -> pd.DataFrame:
    """Count intermediate-budget solutions already feasible at full protection."""

    data = complete_panel(robust).copy()

    def is_fully_protected(row: pd.Series) -> bool:
        instance = instances[row["instance_id"]]
        open_facilities = index_set(row["open_facilities"])
        allocations = allocation_map(row["allocation_fractions"])
        for i in range(instance.facility_count):
            full_load = sum(
                (1.0 + row["delta"])
                * instance.nominal_demands[j]
                * allocations.get((i, j), 0.0)
                for j in range(instance.customer_count)
            )
            available_capacity = (
                instance.capacities[i] if i in open_facilities else 0.0
            )
            if full_load > available_capacity + FEASIBILITY_TOLERANCE:
                return False
        return True

    data["full_protection_feasible"] = data.apply(
        is_fully_protected, axis=1
    )
    summary = (
        data.groupby(["gamma_rel", "delta"])
        .agg(
            included_solutions=("instance_id", "size"),
            full_protection_feasible_solutions=(
                "full_protection_feasible",
                "sum",
            ),
        )
        .reset_index()
    )
    summary["failed_full_protection_test_solutions"] = (
        summary["included_solutions"]
        - summary["full_protection_feasible_solutions"]
    )
    return summary.sort_values(["gamma_rel", "delta"])


def summarize_opening_cost_decomposition(
    robust: pd.DataFrame,
) -> pd.DataFrame:
    """Decompose the full-protection cost increase by opening-cost level."""

    full_protection = complete_panel(robust)
    full_protection = full_protection[
        full_protection["gamma_rel"] == 1.0
    ]
    return (
        full_protection.groupby(["opening_cost_level", "delta"])
        .agg(
            included_instances=("instance_id", "size"),
            mean_opening_cost_change=("opening_cost_change", "mean"),
            mean_assignment_cost_change=("assignment_cost_change", "mean"),
            mean_total_cost_change=("total_cost_change", "mean"),
            mean_por_pct=("price_of_robustness_pct", "mean"),
        )
        .reset_index()
        .sort_values(["delta", "opening_cost_level"])
    )


def plot_budget_trajectories(
    summary: pd.DataFrame,
    group_column: str,
    group_labels: dict[object, str],
    output_path: Path,
) -> None:
    """Plot mean PoR trajectories for the three deviations."""

    deltas = sorted(summary["delta"].unique())
    figure, axes = plt.subplots(
        1, len(deltas), figsize=(12, 4), sharex=True
    )

    for column, delta in enumerate(deltas):
        delta_data = summary[summary["delta"] == delta]
        for group_value, label in group_labels.items():
            group_data = delta_data[delta_data[group_column] == group_value]
            axes[column].plot(
                group_data["gamma_rel"],
                group_data["mean_por_pct"],
                marker="o",
                label=label,
            )
        axes[column].set_title(rf"$\delta={delta:.2f}$")
        axes[column].set_xlabel(r"$\Gamma_{rel}$")
        axes[column].grid(alpha=0.25)
    axes[0].set_ylabel("Mean Price of Robustness (%)")
    axes[-1].legend(loc="best")
    figure.tight_layout(pad=1.5)
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
    """Summarize observed runtime by size, deviation, and budget."""

    runtime = results.copy()
    runtime["is_time_limit"] = runtime["status"] == "TIME_LIMIT"
    size_columns = ["facility_count", "customer_count"]
    nominal = (
        runtime[runtime["model_type"] == "nominal"]
        .groupby(size_columns)
        .agg(
            mean_runtime_sec=("runtime_sec", "mean"),
            runs=("runtime_sec", "size"),
            time_limit_runs=("is_time_limit", "sum"),
        )
        .reset_index()
    )
    nominal["model_type"] = "nominal"
    nominal["delta"] = pd.NA
    nominal["gamma_rel"] = pd.NA

    robust = (
        runtime[runtime["model_type"] == "robust"]
        .groupby(size_columns + ["delta", "gamma_rel"])
        .agg(
            mean_runtime_sec=("runtime_sec", "mean"),
            runs=("runtime_sec", "size"),
            time_limit_runs=("is_time_limit", "sum"),
        )
        .reset_index()
    )
    robust["model_type"] = "robust"
    return (
        pd.concat([nominal, robust], ignore_index=True)
        .sort_values(
            ["customer_count", "model_type", "delta", "gamma_rel"],
            na_position="first",
        )
        .reset_index(drop=True)
    )


def summarize_scalability_objective_effects(
    robust: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize PoR from the optimal runs available in each grid cell."""

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
    optimal_summary = (
        robust[robust["is_optimal"]]
        .groupby(group_columns)
        .agg(
            mean_por_pct=("price_of_robustness_pct", "mean"),
            included_subinstances=("instance_id", "size"),
        )
        .reset_index()
    )
    summary = status_counts.merge(
        optimal_summary,
        on=group_columns,
        how="left",
    )
    summary["included_subinstances"] = (
        summary["included_subinstances"].fillna(0).astype(int)
    )
    return summary.sort_values(["customer_count", "delta", "gamma_rel"])


def summarize_scalability_solver_outcomes(
    robust: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize optimal runs, time limits, and final gaps by size."""

    data = robust.copy()
    data["is_optimal"] = data["status"] == "OPTIMAL"
    data["is_time_limit"] = data["status"] == "TIME_LIMIT"
    data["time_limit_gap_pct"] = (
        100.0 * data["mip_gap"].where(data["is_time_limit"])
    )
    return (
        data.groupby(["facility_count", "customer_count"])
        .agg(
            robust_runs=("status", "size"),
            optimal_runs=("is_optimal", "sum"),
            time_limit_runs=("is_time_limit", "sum"),
            mean_time_limit_gap_pct=("time_limit_gap_pct", "mean"),
            maximum_time_limit_gap_pct=("time_limit_gap_pct", "max"),
        )
        .reset_index()
        .sort_values("customer_count")
    )


def plot_scalability_runtime(
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot runtime by size and budget, with one panel per deviation."""

    nominal = summary[summary["model_type"] == "nominal"]
    robust = summary[summary["model_type"] == "robust"]
    deltas = sorted(robust["delta"].unique())
    gamma_rels = sorted(robust["gamma_rel"].unique())
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

    nominal = nominal.sort_values("customer_count")
    for axis, delta in zip(axes, deltas):
        axis.plot(
            size_labels,
            nominal["mean_runtime_sec"],
            marker="o",
            label="Nominal",
        )
        delta_data = robust[robust["delta"] == delta]
        for gamma_rel in gamma_rels:
            budget_data = delta_data[
                delta_data["gamma_rel"] == gamma_rel
            ].sort_values("customer_count")
            axis.plot(
                size_labels,
                budget_data["mean_runtime_sec"],
                marker="o",
                label=rf"$\Gamma_{{rel}}={gamma_rel:.2f}$",
            )
        axis.set_title(rf"$\delta={delta:.2f}$")
        axis.set_xlabel("Facilities x customers")
        axis.set_yscale("log")
        axis.grid(alpha=0.25)

    axes[0].set_ylabel("Mean observed runtime (seconds, log scale)")
    axes[-1].legend(loc="best")
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

    axes[0].set_ylabel("Mean Price of Robustness (%)\n(optimal runs in each cell)")
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
    instances = load_benchmark_instances(results, directory)
    robust = add_objective_components(robust, instances)
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
    scalability_solver_outcomes = summarize_scalability_solver_outcomes(
        scalability_robust
    )

    summaries = {
        "cfl_por_summary.csv": summarize_price_of_robustness(robust),
        "cfl_cap4_stress_summary.csv": summarize_cap4_stress_case(robust),
        "cfl_full_protection_feasibility.csv": (
            summarize_full_protection_feasibility(robust, instances)
        ),
        "cfl_opening_cost_decomposition.csv": (
            summarize_opening_cost_decomposition(robust)
        ),
        "cfl_scalability_runtime_summary.csv": scalability_runtime,
        "cfl_scalability_por_summary.csv": scalability_objective_effects,
        "cfl_scalability_status_summary.csv": scalability_solver_outcomes,
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
