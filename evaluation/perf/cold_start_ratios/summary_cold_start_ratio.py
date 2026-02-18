import pandas as pd
import os

# List of benchmarks and their specific memory sizes
benchmark_memory_sizes = {
    "110.dynamic-html": [128, 256, 512, 1024],
    "120.uploader": [128, 256, 512, 1024],
    "210.thumbnailer": [128, 256, 512, 1024],
    "220.video-processing": [512, 1024, 2048, 4096],
    "311.compression": [256, 512, 1024, 2048],
    "501.graph-pagerank": [128, 256, 512, 1024],
}

# Get the project directory by going two levels up from the current script location
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Results and output directories relative to the project directory
results_dir = os.path.join(project_dir, "perf")
output_dir = os.path.join(project_dir, "perf", "cold_start_ratios")
os.makedirs(output_dir, exist_ok=True)

# Initialize a list to store summary data
summary_data = []

# Loop through each benchmark and its specific memory sizes
for benchmark, memory_sizes in benchmark_memory_sizes.items():
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

    # Calculate cold-to-warm ratios for each memory size and architecture
    for memory in memory_sizes:
        for arch in ["ARM", "x86"]:
            filtered_data = data_combined[
                (data_combined["memory"] == memory) & (data_combined["architecture"] == arch)
            ]

            # Separate cold and warm start client times
            cold_times = filtered_data[filtered_data["type"] == "cold"]["client_time"]
            warm_times = filtered_data[filtered_data["type"] == "warm"]["client_time"]

            # Ensure both cold and warm times exist
            if len(cold_times) > 0 and len(warm_times) > 0:
                # Compute the mean cold-to-warm ratio
                ratio = cold_times.mean() / warm_times.mean()
                summary_data.append([benchmark, arch, memory, ratio])

# Create the summary table
summary_table = pd.DataFrame(summary_data, columns=["Benchmark", "Architecture", "Memory", "Cold-to-Warm Ratio"])

# Reshape the table to have "Memory Size 1" through "Memory Size 4" columns
reshaped_data = []
for benchmark, group in summary_table.groupby("Benchmark"):
    # Get the memory sizes for this benchmark
    memory_sizes = benchmark_memory_sizes[benchmark]
    for arch in ["ARM", "x86"]:
        row_data = {"Benchmark": benchmark, "Architecture": arch}
        arch_group = group[group["Architecture"] == arch]
        for i, mem in enumerate(memory_sizes, start=1):
            ratio = arch_group[arch_group["Memory"] == mem]["Cold-to-Warm Ratio"].mean()
            row_data[f"Memory Size {i}"] = ratio
        reshaped_data.append(row_data)

# Convert reshaped data into a DataFrame
final_table = pd.DataFrame(reshaped_data)

# Save the table as a CSV
output_table_path = os.path.join(output_dir, "summary_cold_warm_ratio.csv")
final_table.to_csv(output_table_path, index=False)
print(f"Summary table saved to: {output_table_path}")

# Display the table
print(final_table)
