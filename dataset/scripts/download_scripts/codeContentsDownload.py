from datasets import load_dataset
import json
import os

# Define the save path
save_path = "/mnt/d/wsl_workspace/data/CodeContents"
os.makedirs(save_path, exist_ok=True)  # Ensure the directory exists

# Load the CodeContests dataset in streaming mode
dataset = load_dataset("deepmind/code_contests", split="train", streaming=True)

# Take the first 1000 examples
subset_size = 1000
subset = dataset.take(subset_size)

# Convert to a list
subset_list = list(subset)

# Save the subset as a JSON file
output_file = os.path.join(save_path, "code_contests_subset.json")
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(subset_list, f, indent=4)

print(f"Saved the first {subset_size} examples to {output_file}")
