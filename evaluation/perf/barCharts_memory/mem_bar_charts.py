import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from evaluation.utils import (
    BENCHMARKS, ALPHABET_LABELS, LINE_COLORS, MEMORY_SIZES,
    load_benchmark_data, setup_subplots, save_and_show_figure, add_ggplot_title,
    calculate_bootstrap_ci
)

plt.style.use("../../scientific.mplstyle")


def create_bar_plots(bar_data, output_dir):
    fig_bar, axes_bar = setup_subplots(2, 3)
    hue_order = ["ARM cold", "ARM warm", "x86 cold", "x86 warm"]

    for i, benchmark in enumerate(BENCHMARKS.keys()):
        ax = axes_bar[i]
        benchmark_data = bar_data[bar_data["benchmark"] == benchmark]
        sns.barplot(x="memory", y="mem_used", hue="label", data=benchmark_data, ax=ax, dodge=True,
                    palette=LINE_COLORS, hue_order=hue_order, errorbar=("ci", 95))
        add_ggplot_title(ax, f"{ALPHABET_LABELS[i]} {benchmark}")
        ax.set_xlabel("")
        if ax.get_legend():
            ax.get_legend().remove()
        ax.set_ylabel("")

        _, top = ax.get_ylim()
        ax.set_ylim(bottom=0, top=top * 1.1)

    fig_bar.supxlabel("Memory Size (MB)", x=0.537, y=0.03)
    fig_bar.supylabel("Peak Memory Usage (MB)")
    handles, labels = axes_bar[0].get_legend_handles_labels()
    fig_bar.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.537, -0.08), frameon=True,
                   edgecolor='black')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    save_and_show_figure(fig_bar, os.path.join(output_dir, "memory_usage_bar_charts.pdf"))


def create_line_plots(line_data, output_dir):
    fig_line, axes_line = setup_subplots(2, 3)

    for i, benchmark in enumerate(BENCHMARKS.keys()):
        ax = axes_line[i]
        benchmark_data = line_data[line_data["benchmark"] == benchmark]
        mem_sizes = MEMORY_SIZES[benchmark]
        x_positions = np.arange(len(mem_sizes))

        for arch in ["ARM", "x86"]:
            for type_ in ["cold", "warm"]:
                label = f"{arch} {type_}"
                subset = benchmark_data[benchmark_data["label"] == label]

                if subset.empty:
                    continue

                means = []
                lowers = []
                uppers = []

                for mem in mem_sizes:
                    vals = subset[subset["memory"] == mem]["mem_used"].values
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
                linestyle = '--' if type_ == "warm" else '-'

                ax.plot(x_positions, means, marker='o', color=color, label=label, linestyle=linestyle)
                ax.fill_between(x_positions, lowers, uppers, color=color, alpha=0.2)

        add_ggplot_title(ax, f"{ALPHABET_LABELS[i]} {benchmark}")
        ax.set_xlabel("")
        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(m) for m in mem_sizes])
        ax.set_ylabel("")

        _, top = ax.get_ylim()
        ax.set_ylim(bottom=0, top=top * 1.1)

    fig_line.supxlabel("Memory Size (MB)", x=0.537, y=0.03)
    fig_line.supylabel("Peak Memory Usage (MB)")

    handles, labels = axes_line[0].get_legend_handles_labels()
    fig_line.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.537, -0.08), ncol=4,
                    frameon=True, edgecolor='black')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    save_and_show_figure(fig_line, os.path.join(output_dir, "memory_usage_line_plots.pdf"))

    summary_cold = line_data[line_data["type"] == "cold"].groupby(["benchmark", "memory", "architecture"])[
        "mem_used"].mean().reset_index(name="cold_avg_mem_used")
    summary_warm = line_data[line_data["type"] == "warm"].groupby(["benchmark", "memory", "architecture"])[
        "mem_used"].mean().reset_index(name="warm_avg_mem_used")

    summary = pd.merge(summary_cold, summary_warm, on=["benchmark", "memory", "architecture"])
    summary = summary[["benchmark", "memory", "architecture", "cold_avg_mem_used", "warm_avg_mem_used"]]
    summary.to_csv(os.path.join(output_dir, "summary_memory_usage.csv"), index=False)


def main():
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))

    os.makedirs(output_dir, exist_ok=True)

    data_frames = []
    for benchmark in BENCHMARKS.keys():
        df = load_benchmark_data(results_dir, benchmark)
        df["benchmark"] = benchmark
        df["label"] = df["architecture"] + " " + df["type"]
        data_frames.append(df)

    all_data = pd.concat(data_frames, ignore_index=True)

    create_bar_plots(all_data, output_dir)
    create_line_plots(all_data, output_dir)


if __name__ == "__main__":
    main()
