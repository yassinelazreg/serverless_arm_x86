import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from evaluation.utils import (
    BENCHMARKS, ALPHABET_LABELS, LINE_COLORS, MEMORY_SIZES,
    load_benchmark_data, setup_subplots, set_scientific_notation,
    save_and_show_figure, add_ggplot_title, calculate_bootstrap_ci
)

plt.style.use("../../scientific.mplstyle")


def create_cold_boxplots(data_combined, output_dir):
    fig_boxplot_cold, axes_boxplot_cold = setup_subplots(2, 3)

    for idx, benchmark in enumerate(BENCHMARKS.keys()):
        cold_data = data_combined[benchmark][data_combined[benchmark]["type"] == "cold"]
        mem_sizes = MEMORY_SIZES[benchmark]
        ax = axes_boxplot_cold[idx]

        sns.boxplot(
            x="memory",
            y="client_time",
            hue="architecture",
            data=cold_data,
            palette="Set2",
            order=mem_sizes,
            ax=ax
        )
        add_ggplot_title(ax, f"{ALPHABET_LABELS[idx]} {benchmark}")
        ax.set_xlabel("")
        set_scientific_notation(ax, scilimits=(4, 4))
        ax.set_ylabel("")

        _, top = ax.get_ylim()
        ax.set_ylim(bottom=0, top=top * 1.1)

    fig_boxplot_cold.supxlabel("Memory Size (MB)", x=0.537, y=0.03)
    fig_boxplot_cold.supylabel("Client Time (μs)")
    handles, labels = next((ax for ax in axes_boxplot_cold if ax.get_legend()),
                           axes_boxplot_cold[0]).get_legend_handles_labels()
    for ax in axes_boxplot_cold:
        if ax.get_legend():
            ax.get_legend().remove()
    fig_boxplot_cold.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.537, -0.08), frameon=True,
                            edgecolor='black', ncol=len(handles))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)

    save_and_show_figure(fig_boxplot_cold, os.path.join(output_dir, "combined_boxplots_cold_client_time.pdf"))


def create_warm_boxplots(data_combined, output_dir):
    fig_boxplot_warm, axes_boxplot_warm = setup_subplots(2, 3)

    for idx, benchmark in enumerate(BENCHMARKS.keys()):
        warm_data = data_combined[benchmark][data_combined[benchmark]["type"] == "warm"]
        mem_sizes = MEMORY_SIZES[benchmark]
        ax = axes_boxplot_warm[idx]

        sns.boxplot(
            x="memory",
            y="client_time",
            hue="architecture",
            data=warm_data,
            palette="Set2",
            order=mem_sizes,
            ax=ax
        )
        add_ggplot_title(ax, f"{ALPHABET_LABELS[idx]} {benchmark}")
        ax.set_xlabel("")
        set_scientific_notation(ax, scilimits=(4, 4))
        ax.set_ylabel("")

        _, top = ax.get_ylim()
        ax.set_ylim(bottom=0, top=top * 1.1)

    fig_boxplot_warm.supxlabel("Memory Size (MB)", x=0.537, y=0.03)
    fig_boxplot_warm.supylabel("Client Time (μs)")
    handles, labels = next((ax for ax in axes_boxplot_warm if ax.get_legend()),
                           axes_boxplot_warm[0]).get_legend_handles_labels()
    for ax in axes_boxplot_warm:
        if ax.get_legend():
            ax.get_legend().remove()
    fig_boxplot_warm.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.537, -0.08), frameon=True,
                            edgecolor='black', ncol=len(handles))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)

    save_and_show_figure(fig_boxplot_warm, os.path.join(output_dir, "combined_boxplots_warm_client_time.pdf"))


def create_lineplots(data_combined, output_dir):
    fig_lineplot, axes_lineplot = setup_subplots(2, 3)

    for idx, benchmark in enumerate(BENCHMARKS.keys()):
        mem_sizes = MEMORY_SIZES[benchmark]
        df = data_combined[benchmark]
        ax = axes_lineplot[idx]
        x_positions = np.arange(len(mem_sizes))

        for label in ["ARM cold", "ARM warm", "x86 cold", "x86 warm"]:
            subset = df[df["label"] == label]
            if subset.empty:
                continue

            means = []
            lowers = []
            uppers = []

            for mem in mem_sizes:
                vals = subset[subset["memory"] == mem]["client_time"].values
                if len(vals) > 0:
                    mean, (lower, upper) = calculate_bootstrap_ci(vals)
                    means.append(mean)
                    lowers.append(lower)
                    uppers.append(upper)
                else:
                    means.append(np.nan)
                    lowers.append(np.nan)
                    uppers.append(np.nan)

            color = LINE_COLORS[label]
            linestyle = '--' if 'warm' in label else '-'

            ax.plot(x_positions, means, marker='o', color=color, label=label, linestyle=linestyle)
            ax.fill_between(x_positions, lowers, uppers, color=color, alpha=0.2)

        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(m) for m in mem_sizes])

        add_ggplot_title(ax, f"{ALPHABET_LABELS[idx]} {benchmark}")
        ax.set_xlabel("")
        set_scientific_notation(ax, scilimits=(4, 4))
        ax.set_ylabel("")

        _, top = ax.get_ylim()
        ax.set_ylim(bottom=0, top=top * 1.1)

    fig_lineplot.supxlabel("Memory Size (MB)", x=0.537, y=0.03)
    fig_lineplot.supylabel("Client Time (μs)")

    handles, labels = axes_lineplot[0].get_legend_handles_labels()
    fig_lineplot.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.537, -0.08), frameon=True,
                        edgecolor='black', ncol=4)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)

    save_and_show_figure(fig_lineplot, os.path.join(output_dir, "combined_lineplots_client_time.pdf"))


def main():
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    os.makedirs(output_dir, exist_ok=True)

    data_combined = {}
    for benchmark in BENCHMARKS.keys():
        df = load_benchmark_data(results_dir, benchmark)
        df["label"] = df["architecture"] + " " + df["type"]
        data_combined[benchmark] = df

    create_cold_boxplots(data_combined, output_dir)
    create_warm_boxplots(data_combined, output_dir)
    create_lineplots(data_combined, output_dir)


if __name__ == "__main__":
    main()
