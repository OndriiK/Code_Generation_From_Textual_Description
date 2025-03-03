import json
import pandas as pd

# Load commit messages from the provided JSON file
with open('./commit_messages/huggingface_transformers_commits.json', 'r') as file:
    commit_messages = json.load(file)

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
    "optimize_performance": ["optimize", "performance", "efficient", "speed"],
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

# Filter and label commit messages
filtered_commits = []
for msg in commit_messages:
    labels = label_commit_message(msg)
    if len(msg) <= 250 and len(msg) >= 50 and len(labels) < 4 and not contains_non_ascii(msg):  # Filtering criteria
        filtered_commits.append({"commit": msg, "labels": labels})

# Save the filtered commits to a JSON file
output_file = '/mnt/d/wsl_workspace/data/filtered_commits.json'
with open(output_file, 'w') as file:
    json.dump(filtered_commits, file, indent=4)

print(f"Filtered commits saved to {output_file}")
