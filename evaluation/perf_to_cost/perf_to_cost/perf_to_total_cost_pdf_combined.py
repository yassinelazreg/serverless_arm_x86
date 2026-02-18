import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation.utils import (
    BENCHMARKS, LINE_COLORS, MEMORY_SIZES, ALPHABET_LABELS,
    setup_subplots, save_and_show_figure, set_scientific_notation,
    get_all_costs, add_ggplot_title
)

plt.style.use("../../scientific.mplstyle")


def get_raw_client_times(benchmark, arch, memory, run_type):
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "perf"))
    file_path = os.path.join(results_dir, f"result_{arch.lower()}_{benchmark}.csv")
    if not os.path.exists(file_path):
        return []

    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    filtered_df = df[(df['memory'] == memory) & (df['type'] == run_type)]
    return filtered_df['client_time'].tolist()


def bootstrap_ratio_of_means(data1, data2, n_bootstraps=1000, ci=95):
    if not data1 or not data2:
        return 0, (0, 0)

    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    bootstrapped_ratios = []

    for _ in range(n_bootstraps):
        sample1 = np.random.choice(data1, size=len(data1), replace=True)
        sample2 = np.random.choice(data2, size=len(data2), replace=True)
        mean1 = np.mean(sample1)
        mean2 = np.mean(sample2)
        if mean2 > 0:
            bootstrapped_ratios.append(mean1 / mean2)

    if not bootstrapped_ratios:
        mean_ratio = np.mean(data1) / np.mean(data2) if np.mean(data2) != 0 else 0
        return mean_ratio, (mean_ratio, mean_ratio)

    mean_ratio = np.mean(data1) / np.mean(data2) if np.mean(data2) != 0 else 0
    lower_percentile = (100 - ci) / 2
    upper_percentile = 100 - lower_percentile
    ci_lower = np.percentile(bootstrapped_ratios, lower_percentile)
    ci_upper = np.percentile(bootstrapped_ratios, upper_percentile)

    return mean_ratio, (ci_lower, ci_upper)


def calculate_all_ratios_with_ci(use_total_cost=False):
    all_ratios = {}
    for benchmark_name, invocation_key in BENCHMARKS.items():
        memory_sizes = MEMORY_SIZES[benchmark_name]
        for mem in memory_sizes:
            key = (benchmark_name, str(mem))
            all_ratios[key] = {"means": {}, "cis": {}}

            for arch in ["ARM", "x86"]:
                for run_type in ["cold", "warm"]:
                    label = f"{arch} {run_type}"
                    client_times_raw = get_raw_client_times(benchmark_name, arch, mem, run_type)

                    cost_file_path = os.path.abspath(os.path.join(
                        os.path.dirname(__file__), "..", "..", "cost",
                        benchmark_name, arch.lower(), f"{run_type}_results_{mem}-processed.json"
                    ))

                    costs_raw = []
                    if os.path.exists(cost_file_path):
                        costs_raw = get_all_costs(cost_file_path, invocation_key, is_arm=(arch == "ARM"),
                                                  use_total_cost=use_total_cost)

                    mean_ratio, ci_ratio = bootstrap_ratio_of_means(client_times_raw, costs_raw)
                    all_ratios[key]["means"][label] = mean_ratio
                    all_ratios[key]["cis"][label] = ci_ratio
    return all_ratios


def save_ratios_to_csv(ratios_data, output_file):
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ["Benchmark", "Memory Size", "ARM Cold Ratio", "ARM Warm Ratio", "x86 Cold Ratio", "x86 Warm Ratio"])
        sorted_keys = sorted(ratios_data.keys(), key=lambda x: (x[0], int(x[1])))
        for (benchmark, memory) in sorted_keys:
            means = ratios_data[(benchmark, memory)]["means"]
            writer.writerow([
                benchmark, memory,
                means.get('ARM cold', 'N/A'), means.get('ARM warm', 'N/A'),
                means.get('x86 cold', 'N/A'), means.get('x86 warm', 'N/A')
            ])


def plot_combined_ratios(ratios_data, memory_sizes_dict, output_basename):
    benchmarks = sorted(list(BENCHMARKS.keys()))
    fig, axes = setup_subplots(2, 3)

    for idx, benchmark in enumerate(benchmarks):
        ax = axes[idx]
        memory_sizes = [str(size) for size in memory_sizes_dict[benchmark]]
        x_positions = np.arange(len(memory_sizes))

        for arch in ["ARM", "x86"]:
            for run_type in ["cold", "warm"]:
                label = f"{arch} {run_type}"
                linestyle = '--' if run_type == "warm" else '-'
                means, cis_lower, cis_upper = [], [], []
                for mem in memory_sizes:
                    key = (benchmark, mem)
                    means.append(ratios_data[key]["means"][label])
                    cis_lower.append(ratios_data[key]["cis"][label][0])
                    cis_upper.append(ratios_data[key]["cis"][label][1])
                ax.plot(x_positions, means, label=label, marker='o', linestyle=linestyle, color=LINE_COLORS[label])
                ax.fill_between(x_positions, cis_lower, cis_upper, color=LINE_COLORS[label], alpha=0.2)

        add_ggplot_title(ax, f"{ALPHABET_LABELS[idx]} {benchmark}")
        ax.set_xlabel("")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(memory_sizes)
        set_scientific_notation(ax, scilimits=(4, 4))

        _, top = ax.get_ylim()
        ax.set_ylim(bottom=0, top=top * 1.1)

    for ax in axes:
        if ax.get_legend():
            ax.get_legend().remove()

    fig.supxlabel("Memory Size (MB)", x=0.537, y=0.03)
    fig.supylabel("Performance-Cost Ratio (μs/USD)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.537, -0.08),
               frameon=True, edgecolor='black', ncol=4)
    save_and_show_figure(fig, f"{output_basename}.pdf")


def create_plots(ratios_data, memory_sizes_dict, output_dir, is_warm=False):
    benchmarks = sorted(list(BENCHMARKS.keys()))
    fig, axes = setup_subplots(2, 3)
    run_type = "warm" if is_warm else "cold"

    for idx, benchmark in enumerate(benchmarks):
        ax = axes[idx]
        memory_sizes = [str(size) for size in memory_sizes_dict[benchmark]]
        x_positions = np.arange(len(memory_sizes))
        for arch in ["ARM", "x86"]:
            label = f"{arch} {run_type}"
            means, cis_lower, cis_upper = [], [], []
            for mem in memory_sizes:
                key = (benchmark, mem)
                means.append(ratios_data[key]["means"][label])
                cis_lower.append(ratios_data[key]["cis"][label][0])
                cis_upper.append(ratios_data[key]["cis"][label][1])
            ax.plot(x_positions, means, label=label, marker='o', linestyle='-', color=LINE_COLORS[label])
            ax.fill_between(x_positions, cis_lower, cis_upper, color=LINE_COLORS[label], alpha=0.2)

        add_ggplot_title(ax, f"{ALPHABET_LABELS[idx]} {benchmark}")
        ax.set_xlabel("")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(memory_sizes)
        set_scientific_notation(ax, scilimits=(4, 4))

        _, top = ax.get_ylim()
        ax.set_ylim(bottom=0, top=top * 1.1)

    for ax in axes:
        if ax.get_legend():
            ax.get_legend().remove()

    fig.supxlabel("Memory Size (MB)", x=0.537, y=0.03)
    fig.supylabel("Performance-Cost Ratio (μs/USD)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.537, -0.08),
               frameon=True, edgecolor='black', ncol=len(handles))
    filename = os.path.join(output_dir, f"perf_to_cost_{'warm' if is_warm else 'cold'}.pdf")
    save_and_show_figure(fig, filename)


def main():
    memory_sizes_dict = {name: sizes for name, sizes in MEMORY_SIZES.items()}

    ratios_with_ci = calculate_all_ratios_with_ci(use_total_cost=False)
    total_cost_ratios_with_ci = calculate_all_ratios_with_ci(use_total_cost=True)

    save_ratios_to_csv(ratios_with_ci, os.path.join("..", "performance_cost_ratios.csv"))
    save_ratios_to_csv(total_cost_ratios_with_ci, os.path.join("..", "performance_total_cost_ratios.csv"))

    plot_combined_ratios(ratios_with_ci, memory_sizes_dict, os.path.join("..", "perf_to_cost_combined"))
    plot_combined_ratios(total_cost_ratios_with_ci, memory_sizes_dict,
                         os.path.join("..", "perf_to_total_cost_combined"))

    create_plots(ratios_with_ci, memory_sizes_dict, "..", is_warm=False)
    create_plots(ratios_with_ci, memory_sizes_dict, "..", is_warm=True)


if __name__ == "__main__":
    main()
