from datasets import load_dataset
import json
import os

# Define save path
save_path = "/mnt/d/wsl_workspace/data/CodeAlpaca"
os.makedirs(save_path, exist_ok=True)  # Ensure the directory exists

# Load the dataset in streaming mode to avoid downloading everything
dataset = load_dataset("sahil2801/CodeAlpaca-20k", split="train", streaming=True)

# Take a subset (e.g., first 1000 examples)
subset_size = 1000
subset = dataset.take(subset_size)

# Convert the subset to a list
subset_list = list(subset)

# Define output file path
output_file = os.path.join(save_path, "code_alpaca_subset.json")

# Save to JSON
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(subset_list, f, indent=4)

print(f"Saved {subset_size} examples to {output_file}")
