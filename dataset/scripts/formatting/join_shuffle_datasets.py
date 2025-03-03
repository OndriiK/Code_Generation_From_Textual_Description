import json
import random

# File paths
dataset1_path = "/mnt/d/wsl_workspace/data/updated_upsampled_github_commit_dataset.json"
dataset2_path = "./updated_intent_dataset.json"
output_file = "/mnt/d/wsl_workspace/data/final_intent_dataset.json"

# Load the datasets
with open(dataset1_path, 'r') as file:
    dataset1 = json.load(file)

with open(dataset2_path, 'r') as file:
    dataset2 = json.load(file)

# Combine the datasets
combined_dataset = dataset1 + dataset2

# Shuffle the combined dataset
random.shuffle(combined_dataset)

# Save the shuffled dataset to a new file
with open(output_file, 'w') as file:
    json.dump(combined_dataset, file, indent=4)

print(f"Combined and shuffled dataset saved to {output_file}")
print(f"Total number of examples: {len(combined_dataset)}")
