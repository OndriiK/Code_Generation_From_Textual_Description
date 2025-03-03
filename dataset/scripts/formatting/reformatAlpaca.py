import json
import os

# Load CodeAlpaca data
code_alpaca_path = "/mnt/d/wsl_workspace/data/CodeAlpaca/code_alpaca_subset.json"
with open(code_alpaca_path, "r", encoding="utf-8") as f:
    code_alpaca_data = json.load(f)

# Save path for the reformatted dataset
output_path = "/mnt/d/wsl_workspace/data/CodeAlpaca/unified_dynamic_dataset.json"

# Helper function to generate dynamic steps
def generate_dynamic_steps(instruction, code_output, related_file):
    steps = []

    # Analyze the task dynamically
    if "create" in instruction.lower():
        steps.append({
            "step": len(steps) + 1,
            "action": f"Create a new file named `{related_file}`.",
            "file": related_file,
            "line_hint": "Start from Line 1."
        })
    if "array" in instruction.lower():
        steps.append({
            "step": len(steps) + 1,
            "action": f"Write code to define an array as described: {code_output}.",
            "file": related_file,
            "line_hint": "Write on Lines 1-3."
        })
    if "function" in instruction.lower():
        steps.append({
            "step": len(steps) + 1,
            "action": f"Implement the function as per the requirements in the instruction.",
            "file": related_file,
            "line_hint": "Lines 5-15."
        })
    if "test" in instruction.lower():
        steps.append({
            "step": len(steps) + 1,
            "action": "Write unit tests for the implementation in a separate test file.",
            "file": f"test_{related_file}",
            "line_hint": "Lines 1-10."
        })
    
    # Add a fallback step **only if no other steps were created**
    if not steps:
        steps.append({
            "step": 1,
            "action": f"Write the code snippet: {code_output}",
            "file": related_file,
            "line_hint": "Finalize on Lines 10-20."
        })

    return steps

# Function to convert CodeAlpaca examples to a dynamic unified format
def convert_to_dynamic_format(data):
    reformatted_data = []

    for example in data:
        instruction = example["instruction"]
        code_output = example["output"]
        input_field = example.get("input", "")

        # Dynamically generate related file name
        related_file = f"task_{hash(instruction) % 1000}.py"

        # Generate dynamic steps
        dynamic_steps = generate_dynamic_steps(instruction, code_output, related_file)

        # Create reformatted entry
        reformatted_entry = {
            "instruction": instruction,
            "input": {
                "context": input_field if input_field else "No additional context provided.",
                "related_files": [related_file]
            },
            "output": dynamic_steps
        }

        reformatted_data.append(reformatted_entry)

    return reformatted_data

# Convert and save the reformatted data
reformatted_dataset = convert_to_dynamic_format(code_alpaca_data)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(reformatted_dataset, f, indent=4)

print(f"Reformatted dataset with dynamic steps saved to {output_path}")
