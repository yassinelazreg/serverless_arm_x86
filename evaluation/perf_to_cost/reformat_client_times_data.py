import csv
import os
from collections import defaultdict

def calculate_averages(input_file):
    """Calculate average times for ARM and x86 data."""
    averages = defaultdict(lambda: {"ARM": None, "x86": None})

    with open(input_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            benchmark = row['benchmark']
            memory = row['memory']
            architecture = row['architecture']
            mean_client_time = float(row['mean_client_time'])

            # Build a unique key combining benchmark and memory size
            key = (benchmark, memory)

            # Add data to the corresponding architecture
            averages[key][architecture] = mean_client_time

    return averages


def merge_and_write_output(cold_averages, warm_averages, output_file):
    """Merge data and write output to a file."""
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow([
            "Benchmark",
            "Memory Size",
            "ARM Cold Avg (ms)",
            "ARM Warm Avg (ms)",
            "x86 Cold Avg (ms)",
            "x86 Warm Avg (ms)"
        ])

        # Write the data rows
        all_keys = set(cold_averages.keys()).union(warm_averages.keys())

        # Sort by benchmark and memory size (the second element of the key tuple)
        sorted_keys = sorted(all_keys, key=lambda x: (x[0], int(x[1])))

        for key in sorted_keys:
            benchmark, memory = key
            cold_data = cold_averages.get(key, {"ARM": None, "x86": None})
            warm_data = warm_averages.get(key, {"ARM": None, "x86": None})

            writer.writerow([
                benchmark,
                memory,
                cold_data.get("ARM", "N/A"),
                warm_data.get("ARM", "N/A"),
                cold_data.get("x86", "N/A"),
                warm_data.get("x86", "N/A")
            ])


# Dynamically set the paths based on the current script's directory
base_dir = os.path.dirname(__file__)  # Get the directory of the current script

cold_data_file = os.path.join(base_dir, "summary_table_cold_runs.csv")  # The cold client time data (table)
warm_data_file = os.path.join(base_dir, "summary_table_warm_runs.csv")  # The warm client time data (table)
output_file = os.path.join(base_dir, "merged_client_times_data.csv")  # Output file for merged data

# Process cold and warm data
cold_averages = calculate_averages(cold_data_file)
warm_averages = calculate_averages(warm_data_file)

# Merge and write output
merge_and_write_output(cold_averages, warm_averages, output_file)

print(f"Merged data written to {output_file}")
