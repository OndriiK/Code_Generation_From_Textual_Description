import json

# Path to the aggregated dataset
input_file = "/mnt/d/wsl_workspace/data/upsampled_github_commit_dataset_validation.json"
output_file = "/mnt/d/wsl_workspace/data/cleaned_github_commit_dataset_validation.json"

# Load the dataset
with open(input_file, 'r') as file:
    data = json.load(file)

# Initialize counters
label_counts = {
    "debug_code": 0,
    "add_feature": 0,
    "write_tests": 0,
    "optimize_performance": 0,
    "document_code": 0
}
total_entries = 0

# Clean and count
cleaned_data = []
for entry in data:
    # Replace \n\n and \r\n in the commit message
    # cleaned_commit1 = entry["commit"].replace("\n\n\0", ".").replace("\r\n\0", ".")
    # cleaned_commit2 = cleaned_commit1.replace("\n\n\n\n", ", ").replace("\r\n\r\n", ", ")
    # cleaned_commit3 = cleaned_commit2.replace("\n\n", ", ").replace("\r\n", ", ")
    # entry["commit"] = cleaned_commit3

    # Count labels
    for label in entry["labels"]:
        if label in label_counts:
            label_counts[label] += 1

    cleaned_data.append(entry)
    total_entries += 1

# Save cleaned data to a new file
# with open(output_file, 'w') as file:
#     json.dump(cleaned_data, file, indent=4)

# Print counts
print("Label counts:")
for label, count in label_counts.items():
    print(f"{label}: {count}")

print(f"Total dataset entries: {total_entries}")
# print(f"Cleaned dataset saved to: {output_file}")
