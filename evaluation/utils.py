import json
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnchoredText
from matplotlib.transforms import ScaledTranslation
from mpl_toolkits.axes_grid1 import make_axes_locatable

BENCHMARKS = {
    "110.dynamic-html": "110_dynamic_html_python_3_8",
    "120.uploader": "120_uploader_python_3_8",
    "210.thumbnailer": "210_thumbnailer_python_3_8",
    "220.video-processing": "220_video_processing_python_3_8",
    "311.compression": "311_compression_python_3_8",
    "501.graph-pagerank": "501_graph_pagerank_python_3_8"
}

MEMORY_SIZES = {
    "110.dynamic-html": [128, 256, 512, 1024],
    "120.uploader": [128, 256, 512, 1024],
    "210.thumbnailer": [128, 256, 512, 1024],
    "220.video-processing": [512, 1024, 2048, 4096],
    "311.compression": [256, 512, 1024, 2048],
    "501.graph-pagerank": [128, 256, 512, 1024]
}

LINE_COLORS = {
    "ARM cold": "#1f77b4",
    "ARM warm": "#a6cee3",
    "x86 cold": "#ff7f0e",
    "x86 warm": "#ffbb78"
}

ALPHABET_LABELS = [f"{chr(97 + i)})" for i in range(len(BENCHMARKS))]

COST_MULTIPLIER_ARM = 0.0000133334
COST_MULTIPLIER_X86 = 0.0000166667
REQUEST_COST_TOTAL = 0.2048


def set_scientific_notation(ax, scilimits=(0, 0), dx=-70, dy=20):
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="y", scilimits=scilimits)

    offset = ax.yaxis.get_offset_text()

    text_transform = offset.get_transform() + ScaledTranslation(
        dx / 72, dy / 72, ax.figure.dpi_scale_trans
    )
    offset.set_transform(text_transform)


def add_ggplot_title(ax, title, color="#d3d3d3", fontsize=20):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="19%", pad=0)

    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    cax.set_facecolor(color)

    at = AnchoredText(
        title,
        loc="center",
        frameon=False,
        prop=dict(
            fontsize=fontsize,
            backgroundcolor=color
        )
    )

    cax.add_artist(at)
    return cax


def calculate_cost_arm(gb_seconds):
    return gb_seconds * COST_MULTIPLIER_ARM


def calculate_cost_x86(gb_seconds):
    return gb_seconds * COST_MULTIPLIER_X86


def calculate_total_cost_arm(gb_seconds):
    return gb_seconds * COST_MULTIPLIER_ARM + REQUEST_COST_TOTAL


def calculate_total_cost_x86(gb_seconds):
    return gb_seconds * COST_MULTIPLIER_X86 + REQUEST_COST_TOTAL


def get_benchmark_files(benchmark_name, memory_sizes):
    arm_cold = [os.path.join(benchmark_name, "arm", f"cold_results_{mem}-processed.json") for mem in memory_sizes]
    arm_warm = [os.path.join(benchmark_name, "arm", f"warm_results_{mem}-processed.json") for mem in memory_sizes]
    x86_cold = [os.path.join(benchmark_name, "x86", f"cold_results_{mem}-processed.json") for mem in memory_sizes]
    x86_warm = [os.path.join(benchmark_name, "x86", f"warm_results_{mem}-processed.json") for mem in memory_sizes]

    return {
        "arm_cold": arm_cold,
        "arm_warm": arm_warm,
        "x86_cold": x86_cold,
        "x86_warm": x86_warm
    }


def get_all_costs(file_path, invocation_key, is_arm, use_total_cost=False):
    with open(file_path, 'r') as f:
        data = json.load(f)

    invocations = data['_invocations'].get(invocation_key, {})
    costs = []

    for invocation in invocations.values():
        gb_seconds = invocation['billing']['_gb_seconds']
        if gb_seconds > 0:
            if use_total_cost:
                cost = calculate_total_cost_arm(gb_seconds) if is_arm else calculate_total_cost_x86(gb_seconds)
            else:
                cost = calculate_cost_arm(gb_seconds) if is_arm else calculate_cost_x86(gb_seconds)
            costs.append(cost)
    return costs


def calculate_bootstrap_ci(data, n_bootstraps=1000, ci=95):
    if not hasattr(data, '__len__') or len(data) < 2:
        mean_val = np.mean(data) if hasattr(data, '__len__') and len(data) > 0 else 0
        return mean_val, (mean_val, mean_val)

    data = np.asarray(data)
    bootstrapped_means = np.array([
        np.mean(np.random.choice(data, size=len(data), replace=True))
        for _ in range(n_bootstraps)
    ])

    mean = np.mean(data)
    lower_percentile = (100 - ci) / 2
    upper_percentile = 100 - lower_percentile

    ci_lower = np.percentile(bootstrapped_means, lower_percentile)
    ci_upper = np.percentile(bootstrapped_means, upper_percentile)

    return mean, (ci_lower, ci_upper)


def setup_subplots(rows, cols):
    fig, axes = plt.subplots(nrows=rows, ncols=cols)
    return fig, axes.flatten()


def load_benchmark_data(results_dir, benchmark):
    data_arm = pd.read_csv(os.path.join(results_dir, f"result_arm_{benchmark}.csv"))
    data_x86 = pd.read_csv(os.path.join(results_dir, f"result_x86_{benchmark}.csv"))

    data_arm.columns = data_arm.columns.str.strip()
    data_x86.columns = data_x86.columns.str.strip()

    data_arm["architecture"] = "ARM"
    data_x86["architecture"] = "x86"

    data_combined = pd.concat([data_arm, data_x86])
    data_combined["label"] = data_combined["architecture"] + " " + data_combined["type"]

    return data_combined


def load_csv_data(file_path, data_type):
    data = {}
    with open(file_path, 'r') as file:
        reader = pd.read_csv(file)
        for _, row in reader.iterrows():
            benchmark = row['Benchmark']
            memory = row['Memory Size']
            key = (benchmark, str(memory))

            if data_type == 'client_times':
                data[key] = {
                    'ARM Cold': float(row['ARM Cold Avg (ms)']),
                    'ARM Warm': float(row['ARM Warm Avg (ms)']),
                    'x86 Cold': float(row['x86 Cold Avg (ms)']),
                    'x86 Warm': float(row['x86 Warm Avg (ms)'])
                }
            elif data_type == 'cost':
                data[key] = {
                    'ARM Cold': float(row['ARM Cold Avg (USD)']),
                    'ARM Warm': float(row['ARM Warm Avg (USD)']),
                    'x86 Cold': float(row['x86 Cold Avg (USD)']),
                    'x86 Warm': float(row['x86 Warm Avg (USD)'])
                }
    return data


def save_and_show_figure(fig, filename):
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)
