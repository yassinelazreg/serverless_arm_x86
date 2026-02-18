import csv

import matplotlib.pyplot as plt
import numpy as np

from evaluation.utils import (
    BENCHMARKS, MEMORY_SIZES, LINE_COLORS, ALPHABET_LABELS,
    get_benchmark_files, get_all_costs, calculate_bootstrap_ci,
    add_ggplot_title, setup_subplots, save_and_show_figure
)

plt.style.use("../scientific.mplstyle")


def calculate_costs(files, invocation_key):
    files_arm_cold = [f"../cost/{f}" for f in files["arm_cold"]]
    files_arm_warm = [f"../cost/{f}" for f in files["arm_warm"]]
    files_x86_cold = [f"../cost/{f}" for f in files["x86_cold"]]
    files_x86_warm = [f"../cost/{f}" for f in files["x86_warm"]]

    arm_cold_raw = [get_all_costs(f, invocation_key, is_arm=True) for f in files_arm_cold]
    arm_warm_raw = [get_all_costs(f, invocation_key, is_arm=True) for f in files_arm_warm]
    x86_cold_raw = [get_all_costs(f, invocation_key, is_arm=False) for f in files_x86_cold]
    x86_warm_raw = [get_all_costs(f, invocation_key, is_arm=False) for f in files_x86_warm]

    costs_means = {
        "arm_cold": [calculate_bootstrap_ci(data)[0] for data in arm_cold_raw],
        "arm_warm": [calculate_bootstrap_ci(data)[0] for data in arm_warm_raw],
        "x86_cold": [calculate_bootstrap_ci(data)[0] for data in x86_cold_raw],
        "x86_warm": [calculate_bootstrap_ci(data)[0] for data in x86_warm_raw],
    }
    costs_cis = {
        "arm_cold": [calculate_bootstrap_ci(data)[1] for data in arm_cold_raw],
        "arm_warm": [calculate_bootstrap_ci(data)[1] for data in arm_warm_raw],
        "x86_cold": [calculate_bootstrap_ci(data)[1] for data in x86_cold_raw],
        "x86_warm": [calculate_bootstrap_ci(data)[1] for data in x86_warm_raw],
    }
    return costs_means, costs_cis


def create_summary_data(benchmark_name, memory_sizes, costs):
    results = []
    for j, mem in enumerate(memory_sizes):
        results.append([
            benchmark_name,
            mem,
            round(costs["arm_cold"][j], 8),
            round(costs["arm_warm"][j], 8),
            round(costs["x86_cold"][j], 8),
            round(costs["x86_warm"][j], 8)
        ])
    return results


def plot_benchmark(ax, memory_sizes, costs_means, costs_cis, alphabet_label, benchmark_name):
    x_positions = np.arange(len(memory_sizes))
    lines = []

    mean = np.array(costs_means["x86_cold"])
    ci_lower = np.array([ci[0] for ci in costs_cis["x86_cold"]])
    ci_upper = np.array([ci[1] for ci in costs_cis["x86_cold"]])
    lines.append(ax.plot(x_positions, mean, marker='o', color=LINE_COLORS["x86 cold"], label="x86 cold")[0])
    ax.fill_between(x_positions, ci_lower, ci_upper, color=LINE_COLORS["x86 cold"], alpha=0.2)

    mean = np.array(costs_means["arm_cold"])
    ci_lower = np.array([ci[0] for ci in costs_cis["arm_cold"]])
    ci_upper = np.array([ci[1] for ci in costs_cis["arm_cold"]])
    lines.append(ax.plot(x_positions, mean, marker='o', color=LINE_COLORS["ARM cold"], label="ARM cold")[0])
    ax.fill_between(x_positions, ci_lower, ci_upper, color=LINE_COLORS["ARM cold"], alpha=0.2)

    mean = np.array(costs_means["x86_warm"])
    ci_lower = np.array([ci[0] for ci in costs_cis["x86_warm"]])
    ci_upper = np.array([ci[1] for ci in costs_cis["x86_warm"]])
    lines.append(
        ax.plot(x_positions, mean, marker='o', linestyle='--', color=LINE_COLORS["x86 warm"], label="x86 warm")[0])
    ax.fill_between(x_positions, ci_lower, ci_upper, color=LINE_COLORS["x86 warm"], alpha=0.2)

    mean = np.array(costs_means["arm_warm"])
    ci_lower = np.array([ci[0] for ci in costs_cis["arm_warm"]])
    ci_upper = np.array([ci[1] for ci in costs_cis["arm_warm"]])
    lines.append(
        ax.plot(x_positions, mean, marker='o', linestyle='--', color=LINE_COLORS["ARM warm"], label="ARM warm")[0])
    ax.fill_between(x_positions, ci_lower, ci_upper, color=LINE_COLORS["ARM warm"], alpha=0.2)

    add_ggplot_title(ax, f"{alphabet_label} {benchmark_name}")
    ax.set_xlabel("")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(size) for size in memory_sizes])

    return lines


def main():
    fig, axes = setup_subplots(2, 3)

    summary_data = [["Benchmark", "Memory Size", "ARM Cold Avg (USD)", "ARM Warm Avg (USD)",
                     "x86 Cold Avg (USD)", "x86 Warm Avg (USD)"]]

    legend_handles = None
    legend_labels = ["x86 cold", "ARM cold", "x86 warm", "ARM warm"]

    for i, (benchmark_name, invocation_key) in enumerate(BENCHMARKS.items()):
        memory_sizes = MEMORY_SIZES[benchmark_name]
        files = get_benchmark_files(benchmark_name, memory_sizes)
        costs_means, costs_cis = calculate_costs(files, invocation_key)
        summary_data.extend(create_summary_data(benchmark_name, memory_sizes, costs_means))

        ax = axes[i]
        lines = plot_benchmark(ax, memory_sizes, costs_means, costs_cis, ALPHABET_LABELS[i], benchmark_name)

        _, top = ax.get_ylim()
        ax.set_ylim(bottom=0, top=top * 1.1)

        if i == 0:
            legend_handles = lines

    fig.supxlabel("Memory Size (MB)", x=0.537, y=0.03)
    fig.supylabel("Cost (USD)")
    fig.legend(legend_handles, legend_labels, loc='lower center', ncol=4, bbox_to_anchor=(0.537, -0.08), frameon=True,
               edgecolor='black')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)

    save_and_show_figure(fig, "combined_cost_comparison.pdf")

    csv_file = "summary_cost_table.csv"
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(summary_data)


if __name__ == "__main__":
    main()
