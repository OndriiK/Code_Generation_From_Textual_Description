import os
import json

# Predefined intents
predefined_intents = [
    "debug_code",
    "add_feature",
    "write_tests",
    "optimize_performance",
    "document_code"
]

# Define keywords or patterns for each intent
intent_keywords = {
    "debug_code": ["fix", "bug", "error", "issue", "crash"],
    "add_feature": ["add", "support", "implement", "enable"],
    "write_tests": ["test", "tests", "coverage"],
    "optimize_performance": ["optimize", "performance", "efficient", "speed", "refactor", "unnecessary"],
    "document_code": ["doc", "documentation", "comments", "readme"]
}

# Function to check for non-ASCII characters
def contains_non_ascii(s):
    return any(ord(c) > 127 for c in s)

# Function to label commit messages
def label_commit_message(message):
    labels = []
    message_lower = message.lower()
    for intent, keywords in intent_keywords.items():
        if any(keyword in message_lower for keyword in keywords):
            labels.append(intent)
    return labels

# Function to filter and process commit messages from a single file
def filter_commits(input_file):
    with open(input_file, 'r') as file:
        commit_messages = json.load(file)
    
    filtered_commits = []
    for msg in commit_messages:
        if "merge branch" in msg.lower() or "merge pull request" in msg.lower():
            continue
        labels = label_commit_message(msg)
        if len(msg) <= 250 and len(msg) >= 50 and len(labels) < 4 and len(labels) > 0 and not contains_non_ascii(msg):
            filtered_commits.append({"commit": msg, "labels": labels})
    return filtered_commits

# Function to process all files in a directory and aggregate results
def process_directory(input_dir, output_file):
    all_filtered_commits = []
    
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".json"):
                input_file = os.path.join(root, file)
                print(f"Processing: {input_file}")
                filtered_commits = filter_commits(input_file)
                all_filtered_commits.extend(filtered_commits)
    
    # Save all aggregated results into a single JSON file
    with open(output_file, 'w') as file:
        json.dump(all_filtered_commits, file, indent=4)
    
    print(f"Filtered dataset saved to: {output_file}")

# Paths
input_directory = "/mnt/d/wsl_workspace/commit_messages2"
output_file = "/mnt/d/wsl_workspace/data/github_commit_dataset_validation.json"

# Process the directory and save the aggregated dataset
process_directory(input_directory, output_file)
