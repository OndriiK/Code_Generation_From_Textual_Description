import json
import random

# Load the dataset
input_file = "/mnt/d/wsl_workspace/data/cleaned_github_commit_dataset_validation.json"
output_file = "/mnt/d/wsl_workspace/data/upsampled_github_commit_dataset_validation.json"

with open(input_file, 'r') as file:
    data = json.load(file)

# Predefined transformations for augmentation
def augment_message(message):
    test_synonym = random.choice(["test case", "unit test", "regression test"])
    # Replace synonyms, shuffle phrasing, or apply simple tweaks
    replacements = {"fix": "resolve", "add": "include", "optimize": "enhance", "test": test_synonym}
    words = message.split()
    augmented = [replacements.get(word, word) for word in words]
    return " ".join(augmented)

# Separate data by labels
class_data = {
    "debug_code": [],
    "add_feature": [],
    "write_tests": [],
    "optimize_performance": [],
    "document_code": []
}

for entry in data:
    for label in entry["labels"]:
        if label in class_data:
            class_data[label].append(entry)
            break

# Determine maximum class size
max_count = max(len(examples) for examples in class_data.values())

# Upsample classes with augmentation
upsampled_data = []
for label, examples in class_data.items():
    if len(examples) < max_count:
        additional_examples = []
        while len(additional_examples) + len(examples) < max_count:
            original = random.choice(examples)
            augmented = {
                "commit": augment_message(original["commit"]),
                "labels": original["labels"]
            }
            additional_examples.append(augmented)
        upsampled_data.extend(examples + additional_examples)
    else:
        upsampled_data.extend(examples)

# Shuffle the dataset
random.shuffle(upsampled_data)

# Save the dataset
with open(output_file, 'w') as file:
    json.dump(upsampled_data, file, indent=4)

# Print stats
print("Upsampled dataset saved to:", output_file)
print(f"Total upsampled entries: {len(upsampled_data)}")
for label, examples in class_data.items():
    print(f"{label}: {len([entry for entry in upsampled_data if label in entry['labels']])}")
