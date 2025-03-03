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

# Function to label commit messages
def label_commit_message(message):
    labels = []
    message_lower = message.lower()
    for intent, keywords in intent_keywords.items():
        if any(keyword in message_lower for keyword in keywords):
            labels.append(intent)
    return labels

# Apply labeling to commit messages
labeled_commits = [{"commit": msg, "labels": label_commit_message(msg)} for msg in commit_messages]

# Convert to DataFrame for better visualization
labeled_commits_df = pd.DataFrame(labeled_commits)

# Save the labeled commits to a JSON file
output_file = '/mnt/d/wsl_workspace/data/labeled_commits.json'
with open(output_file, 'w') as file:
    json.dump(labeled_commits, file, indent=4)

print(f"Labeled commits saved to {output_file}")

# # Display a batch of labeled commits to the user
# import ace_tools as tools; tools.display_dataframe_to_user(name="Labeled Commit Messages", dataframe=labeled_commits_df.head(20))
