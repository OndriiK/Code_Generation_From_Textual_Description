from datasets import load_dataset
import json
import os

# Define save path
save_path = "/mnt/d/wsl_workspace/data/MandM"
os.makedirs(save_path, exist_ok=True)

# Load dataset in streaming mode
dataset = load_dataset("mandms/benchmark", split="train", streaming=True)

# Take a subset (e.g., first 500 examples)
subset_size = 500
subset = dataset.take(subset_size)

# Convert the subset to a list
subset_list = list(subset)

# Save to JSON file
output_file = os.path.join(save_path, "mandm_subset.json")
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(subset_list, f, indent=4)

print(f"Saved {subset_size} examples from m&m's Benchmark dataset to {output_file}")
