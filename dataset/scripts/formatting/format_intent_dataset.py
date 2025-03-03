import json

# Load the JSON dataset
with open('/mnt/d/wsl_workspace/data/upsampled_github_commit_dataset_validation.json', 'r') as file:
    dataset = json.load(file)

# Process the dataset
for entry in dataset:
    # Rename "label" to "labels"
    if 'commit' in entry:
        entry['text'] = entry.pop('commit')

# Save the updated dataset
output_file = '/mnt/d/wsl_workspace/data/updated_upsampled_github_commit_dataset_validation.json'
with open(output_file, 'w') as file:
    json.dump(dataset, file, indent=4)

print(f"Dataset has been updated and saved to {output_file}")
