import pandas as pd
import os

# List of benchmarks to process (excluding 502.graph-mst and 503.graph-bfs)
benchmarks = [
    "110.dynamic-html", "120.uploader", "210.thumbnailer",
    "220.video-processing", "311.compression", "501.graph-pagerank"
]

# Get the project directory by going two levels up from the current script location
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Results and output directories relative to the project directory
results_dir = os.path.join(project_dir, "perf")
output_dir = os.path.join(project_dir, "perf", "client_times_combined")

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Initialize lists to store summary data for cold and warm runs
summary_data_cold = []
summary_data_warm = []

# Function to calculate the advantage between architectures
def calculate_advantage(mean_arm, mean_x86):
    if mean_arm < mean_x86:
        percentage_faster = ((mean_x86 - mean_arm) / mean_arm) * 100
        return f"ARM, {percentage_faster:.2f}% faster"
    else:
        percentage_faster = ((mean_arm - mean_x86) / mean_x86) * 100
        return f"x86, {percentage_faster:.2f}% faster"

# Loop through each benchmark to compute summary statistics
for benchmark in benchmarks:
    # Load the ARM and x86 data from CSVs
    data_arm = pd.read_csv(os.path.join(results_dir, f"result_arm_{benchmark}.csv"))
    data_x86 = pd.read_csv(os.path.join(results_dir, f"result_x86_{benchmark}.csv"))

    # Ensure columns are correctly named and types are as expected
    data_arm.columns = data_arm.columns.str.strip()
    data_x86.columns = data_x86.columns.str.strip()

    # Add columns to distinguish between ARM and x86
    data_arm["architecture"] = "ARM"
    data_x86["architecture"] = "x86"
    data_combined = pd.concat([data_arm, data_x86])

    # Separate cold and warm runs
    cold_data = data_combined[data_combined["type"] == "cold"]
    warm_data = data_combined[data_combined["type"] == "warm"]

    # Function to compute summary statistics for a given subset of data
    def compute_summary(data, run_type):
        grouped = data.groupby(["memory", "architecture"])
        summary = grouped["client_time"].agg(
            mean_client_time="mean",
            median_client_time="median",
            std_client_time="std"
        ).reset_index()

        # Add benchmark information
        summary["benchmark"] = benchmark

        # Calculate advantage for each memory group
        memory_groups = summary.groupby("memory")
        advantage_mapping = {}
        for memory, group in memory_groups:
            mean_arm = group[group["architecture"] == "ARM"]["mean_client_time"].values[0]
            mean_x86 = group[group["architecture"] == "x86"]["mean_client_time"].values[0]
            advantage_mapping[memory] = calculate_advantage(mean_arm, mean_x86)

        # Map the calculated advantage back to the summary DataFrame
        summary["advantage"] = summary["memory"].map(advantage_mapping)

        # Reorder columns
        summary = summary[[
            "benchmark", "memory", "architecture",
            "mean_client_time", "median_client_time", "std_client_time", "advantage"
        ]]
        return summary

    # Compute summaries for cold and warm runs
    summary_cold = compute_summary(cold_data, "cold")
    summary_warm = compute_summary(warm_data, "warm")

    # Append summaries to the respective lists
    summary_data_cold.append(summary_cold)
    summary_data_warm.append(summary_warm)

# Combine all summaries into single DataFrames
final_summary_cold = pd.concat(summary_data_cold).reset_index(drop=True)
final_summary_warm = pd.concat(summary_data_warm).reset_index(drop=True)

# Save the summary tables as CSV files
cold_table_path = os.path.join(output_dir, "summary_table_cold_runs.csv")
warm_table_path = os.path.join(output_dir, "summary_table_warm_runs.csv")
final_summary_cold.to_csv(cold_table_path, index=False)
final_summary_warm.to_csv(warm_table_path, index=False)

print(f"Saved cold run summary table: {cold_table_path}")
print(f"Saved warm run summary table: {warm_table_path}")
