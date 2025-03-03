from datasets import load_dataset
import json
import os

# Define save path
save_path = "/mnt/d/wsl_workspace/data/TAPAS"
os.makedirs(save_path, exist_ok=True)

# Load dataset in streaming mode to avoid downloading the entire dataset
dataset = load_dataset("MiguelZamoraM/TAPAS", split="train", streaming=True)

# Take a subset (e.g., first 1000 examples)
subset_size = 150
subset = dataset.take(subset_size)

# Convert the subset to a list
subset_list = list(subset)

# Save to JSON file
output_file = os.path.join(save_path, "tapas_subset.json")
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(subset_list, f, indent=4)

print(f"Saved {subset_size} examples from TAPAS dataset to {output_file}")
