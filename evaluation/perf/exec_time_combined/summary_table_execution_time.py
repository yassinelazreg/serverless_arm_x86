import pandas as pd
import os

# List of benchmarks to process (excluding 502.graph-mst and 503.graph-bfs)
benchmarks = [
    "110.dynamic-html", "120.uploader", "210.thumbnailer",
    "220.video-processing", "311.compression", "501.graph-pagerank"
]

results_dir = ".."
output_dir = "."

# DataFrame to store summary statistics for execution times
exec_time_summary = pd.DataFrame(
    columns=["benchmark", "memory", "architecture", "mean_exec_time", "median_exec_time", "std_exec_time", "advantage"])

# Loop through each benchmark to compute execution times and create plots
for benchmark in benchmarks:
    # Load the ARM and x86 data from CSVs
    data_arm = pd.read_csv(f"{results_dir}/result_arm_{benchmark}.csv")
    data_x86 = pd.read_csv(f"{results_dir}/result_x86_{benchmark}.csv")

    # Ensure columns are correctly named and types are as expected
    data_arm.columns = data_arm.columns.str.strip()
    data_x86.columns = data_x86.columns.str.strip()

    # Add columns to distinguish between ARM and x86, and keep only warm invocations
    data_arm["architecture"] = "ARM"
    data_x86["architecture"] = "x86"
    data_combined = pd.concat([data_arm, data_x86])
    warm_data = data_combined[data_combined["type"] == "warm"]

    # Calculate summary statistics for execution times
    summary_stats = warm_data.groupby(["memory", "architecture"])["exec_time"].agg(
        ["mean", "median", "std"]).reset_index()
    summary_stats["benchmark"] = benchmark
    summary_stats.rename(columns={"mean": "mean_exec_time", "median": "median_exec_time", "std": "std_exec_time"},
                         inplace=True)

    # Now we calculate the advantage between ARM and x86 for each memory size
    advantage_list = []
    for memory_size in summary_stats["memory"].unique():
        # Get the ARM and x86 stats for the current memory size
        arm_row = summary_stats[(summary_stats["memory"] == memory_size) & (summary_stats["architecture"] == "ARM")]
        x86_row = summary_stats[(summary_stats["memory"] == memory_size) & (summary_stats["architecture"] == "x86")]

        if not arm_row.empty and not x86_row.empty:
            arm_mean = arm_row["mean_exec_time"].values[0]
            x86_mean = x86_row["mean_exec_time"].values[0]

            # Calculate the percentage advantage
            if arm_mean < x86_mean:
                advantage = f"ARM, {((x86_mean - arm_mean) / arm_mean) * 100:.2f}% faster"
            else:
                advantage = f"x86, {((arm_mean - x86_mean) / x86_mean) * 100:.2f}% faster"

            # Add the advantage information to the list for both ARM and x86 rows
            advantage_list.append(advantage)
            advantage_list.append(advantage)  # Duplicate for the second row (x86 or ARM counterpart)

    # Add the 'advantage' column to the summary table
    summary_stats["advantage"] = advantage_list
    exec_time_summary = pd.concat([exec_time_summary, summary_stats], ignore_index=True)

# Save the execution time summary table as a CSV file
exec_time_summary_filename = f"{output_dir}/summary_execution_time_with_advantage.csv"
exec_time_summary.to_csv(exec_time_summary_filename, index=False)
print(f"Saved execution time summary table with advantage column as CSV: {exec_time_summary_filename}")
