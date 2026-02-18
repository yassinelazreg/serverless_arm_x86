import csv

import matplotlib.pyplot as plt
import numpy as np

from evaluation.utils import (
    BENCHMARKS, MEMORY_SIZES, LINE_COLORS, ALPHABET_LABELS,
    get_benchmark_files, get_all_costs, calculate_bootstrap_ci,
    add_ggplot_title, setup_subplots, save_and_show_figure
)

plt.style.use("../scientific.mplstyle")


def create_cost_plots(use_total_cost=False):
    fig, axes = setup_subplots(2, 3)

    summary_data = [["Benchmark", "Memory Size", "ARM Cold Avg (USD)", "ARM Warm Avg (USD)",
                     "x86 Cold Avg (USD)", "x86 Warm Avg (USD)"]]

    for i, (benchmark_name, invocation_key) in enumerate(BENCHMARKS.items()):
        memory_sizes = MEMORY_SIZES[benchmark_name]
        files = get_benchmark_files(benchmark_name, memory_sizes)

        costs_means = {}
        costs_cis = {}
        is_arm_map = {"arm_cold": True, "arm_warm": True, "x86_cold": False, "x86_warm": False}

        for key, file_list in files.items():
            raw_costs_per_mem = [get_all_costs(f, invocation_key, is_arm_map[key], use_total_cost) for f in file_list]
            bootstrap_results = [calculate_bootstrap_ci(data) for data in raw_costs_per_mem]
            costs_means[key] = [res[0] for res in bootstrap_results]
            costs_cis[key] = [res[1] for res in bootstrap_results]

        for j, mem in enumerate(memory_sizes):
            summary_data.append([
                benchmark_name,
                mem,
                round(costs_means["arm_cold"][j], 8),
                round(costs_means["arm_warm"][j], 8),
                round(costs_means["x86_cold"][j], 8),
                round(costs_means["x86_warm"][j], 8)
            ])

        ax = axes[i]
        x_positions = np.arange(len(memory_sizes))

        for arch in ["ARM", "x86"]:
            key_cold = f"{arch.lower()}_cold"
            mean_cold = np.array(costs_means[key_cold])
            ci_cold_lower = np.array([ci[0] for ci in costs_cis[key_cold]])
            ci_cold_upper = np.array([ci[1] for ci in costs_cis[key_cold]])
            ax.plot(x_positions, mean_cold, marker='o', color=LINE_COLORS[f"{arch} cold"], label=f"{arch} cold")
            ax.fill_between(x_positions, ci_cold_lower, ci_cold_upper,
                            color=LINE_COLORS[f"{arch} cold"], alpha=0.2)

            key_warm = f"{arch.lower()}_warm"
            mean_warm = np.array(costs_means[key_warm])
            ci_warm_lower = np.array([ci[0] for ci in costs_cis[key_warm]])
            ci_warm_upper = np.array([ci[1] for ci in costs_cis[key_warm]])
            ax.plot(x_positions, mean_warm, marker='o', color=LINE_COLORS[f"{arch} warm"],
                    label=f"{arch} warm", linestyle='--')
            ax.fill_between(x_positions, ci_warm_lower, ci_warm_upper,
                            color=LINE_COLORS[f"{arch} warm"], alpha=0.2)

        add_ggplot_title(ax, f"{ALPHABET_LABELS[i]} {benchmark_name}")
        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(size) for size in memory_sizes])

        _, top = ax.get_ylim()
        ax.set_ylim(bottom=0, top=top * 1.1)

    fig.supxlabel("Memory Size (MB)", x=0.537, y=0.03)
    fig.supylabel("Cost (USD)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.537, -0.08), frameon=True,
               edgecolor='black')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)

    file_prefix = "total_" if use_total_cost else ""
    fig_name = f"{file_prefix}cost.pdf"
    csv_file = f"summary_{file_prefix}cost.csv"

    save_and_show_figure(fig, fig_name)
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(summary_data)


if __name__ == "__main__":
    create_cost_plots(use_total_cost=False)
    create_cost_plots(use_total_cost=True)
