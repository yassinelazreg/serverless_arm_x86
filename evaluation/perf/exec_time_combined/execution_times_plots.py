import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from evaluation.utils import (
    BENCHMARKS, ALPHABET_LABELS, MEMORY_SIZES, LINE_COLORS,
    load_benchmark_data, set_scientific_notation, save_and_show_figure,
    add_ggplot_title, calculate_bootstrap_ci, setup_subplots
)

plt.style.use("../../scientific.mplstyle")


def create_execution_time_boxplots(data_combined, output_dir):
    fig_boxplot, axes_boxplot = setup_subplots(2, 3)

    for idx, benchmark in enumerate(BENCHMARKS.keys()):
        warm_data = data_combined[benchmark]
        mem_sizes = MEMORY_SIZES[benchmark]
        ax = axes_boxplot[idx]

        sns.boxplot(
            x="memory",
            y="exec_time",
            hue="architecture",
            data=warm_data,
            palette="Set2",
            order=mem_sizes,
            ax=ax
        )
        add_ggplot_title(ax, f"{ALPHABET_LABELS[idx]} {benchmark}")
        set_scientific_notation(ax, scilimits=(4, 4))
        ax.set_xlabel("")
        ax.set_ylabel("")

        _, top = ax.get_ylim()
        ax.set_ylim(bottom=0, top=top * 1.1)

    fig_boxplot.supxlabel("Memory Size (MB)", x=0.537, y=0.03)
    fig_boxplot.supylabel("Execution Time (μs)")
    handles, labels = next((ax for ax in axes_boxplot if ax.get_legend()),
                           axes_boxplot[0]).get_legend_handles_labels()

    labels = [f"{l} warm" for l in labels]

    for ax in axes_boxplot:
        if ax.get_legend():
            ax.get_legend().remove()
    fig_boxplot.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.537, -0.08), frameon=True,
                       edgecolor='black', ncol=len(handles))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)

    save_and_show_figure(fig_boxplot, os.path.join(output_dir, "combined_boxplots_execution_time.pdf"))


def create_execution_time_lineplots(data_combined, output_dir):
    fig_lineplot, axes_lineplot = setup_subplots(2, 3)

    for idx, benchmark in enumerate(BENCHMARKS.keys()):
        mem_sizes = MEMORY_SIZES[benchmark]
        df = data_combined[benchmark]
        ax = axes_lineplot[idx]
        x_positions = np.arange(len(mem_sizes))

        for arch in ["ARM", "x86"]:
            subset = df[df["architecture"] == arch]
            if subset.empty:
                continue

            means = []
            lowers = []
            uppers = []

            for mem in mem_sizes:
                vals = subset[subset["memory"] == mem]["exec_time"].values
                if len(vals) > 0:
                    mean, (lower, upper) = calculate_bootstrap_ci(vals)
                    means.append(mean)
                    lowers.append(lower)
                    uppers.append(upper)
                else:
                    means.append(np.nan)
                    lowers.append(np.nan)
                    uppers.append(np.nan)

            color = LINE_COLORS[f"{arch} warm"]
            ax.plot(x_positions, means, marker='o', color=color, label=f"{arch} warm", linestyle='--')
            ax.fill_between(x_positions, lowers, uppers, color=color, alpha=0.2)

        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(m) for m in mem_sizes])

        add_ggplot_title(ax, f"{ALPHABET_LABELS[idx]} {benchmark}")
        ax.set_xlabel("")
        ax.set_ylabel("")
        set_scientific_notation(ax, scilimits=(4, 4))

        _, top = ax.get_ylim()
        ax.set_ylim(bottom=0, top=top * 1.1)

    fig_lineplot.supxlabel("Memory Size (MB)", x=0.537, y=0.03)
    fig_lineplot.supylabel("Execution Time (μs)")
    handles, labels = axes_lineplot[0].get_legend_handles_labels()
    fig_lineplot.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.537, -0.08), frameon=True,
                        edgecolor='black', ncol=len(handles))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)

    save_and_show_figure(fig_lineplot, os.path.join(output_dir, "combined_lineplots_execution_time.pdf"))


def prepare_data(results_dir):
    data_combined = {}

    for benchmark in BENCHMARKS.keys():
        all_data = load_benchmark_data(results_dir, benchmark)
        warm_data = all_data[all_data["type"] == "warm"]
        data_combined[benchmark] = warm_data

    return data_combined


def main():
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    os.makedirs(output_dir, exist_ok=True)

    data_combined = prepare_data(results_dir)

    create_execution_time_boxplots(data_combined, output_dir)
    create_execution_time_lineplots(data_combined, output_dir)


if __name__ == "__main__":
    main()
