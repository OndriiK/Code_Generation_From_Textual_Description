import json
import os

# Load MBPP dataset
from datasets import load_dataset

# Define the save path
save_path = "/mnt/d/wsl_workspace/data/MBPP"
os.makedirs(save_path, exist_ok=True)  # Ensure the directory exists

# Load the MBPP dataset in streaming mode
dataset = load_dataset("google-research-datasets/mbpp", split="train", streaming=True)

# Function to generate dynamic steps
def generate_dynamic_steps(task_description, code_snippet, test_cases, related_file):
    steps = []

    # Step 1: Understand the problem
    steps.append({
        "step": len(steps) + 1,
        "action": f"Understand the task: {task_description}",
        "file": related_file,
        "line_hint": "N/A"
    })

    # Step 2: Write the solution
    if code_snippet:
        steps.append({
            "step": len(steps) + 1,
            "action": f"Implement the solution code: {code_snippet.strip()}",
            "file": related_file,
            "line_hint": "Lines 1-20"
        })

    # Step 3: Add test cases if available
    if test_cases:
        steps.append({
            "step": len(steps) + 1,
            "action": f"Add test cases to verify the implementation: {test_cases}",
            "file": f"test_{related_file}",
            "line_hint": "Lines 1-10"
        })

    return steps

# Function to convert MBPP examples to the unified format
def convert_to_unified_format(data, subset_size=1000):
    reformatted_data = []
    count = 0

    for example in data:
        if count >= subset_size:
            break
        count += 1

        # Extract relevant fields
        task_description = example.get("text", "No description provided.")
        code_snippet = example.get("code", "")
        test_cases = example.get("test_list", [])
        related_file = f"task_{example.get('task_id', 'unknown_task')}.py"

        # Generate dynamic steps
        dynamic_steps = generate_dynamic_steps(task_description, code_snippet, test_cases, related_file)

        # Create reformatted entry
        reformatted_entry = {
            "instruction": task_description,
            "input": {
                "context": "Task from MBPP dataset.",
                "related_files": [related_file]
            },
            "output": dynamic_steps
        }

        reformatted_data.append(reformatted_entry)

    return reformatted_data

# Convert the MBPP dataset to unified format
subset_size = 1000
reformatted_dataset = convert_to_unified_format(dataset, subset_size)

# Save the reformatted dataset as a JSON file
output_file = os.path.join(save_path, "mbpp_unified_dataset.json")
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(reformatted_dataset, f, indent=4)

print(f"Saved the unified dataset with {subset_size} examples to {output_file}")
