import json
import os

# Define input and output file paths
INPUT_FILE = r"D:\bakalarka\data\glaive_code_assist\subset12_glaive_code_assistant.json"
FIXED_OUTPUT_FILE = r"D:\bakalarka\data\glaive_code_assist\FIXEDFINAL_glaive_code_assistant.json"

# Save to JSON
def save_to_json(data, output_file):
    """Saves the data to a JSON file as a structured JSON list."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"Saving {len(data)} examples to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print(f"Saved successfully to {output_file}.")

# Fix dataset formatting
def fix_json_format(input_file, output_file):
    """Loads improperly formatted JSON lines and stores them as a structured JSON list."""
    print(f"Fixing JSON format from {input_file}...")
    fixed_data = []
    
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)  # Parse each line as a JSON object
                fixed_data.append(entry)
            except json.JSONDecodeError as e:
                print(f"Skipping invalid line due to error: {e}")
    
    print(f"Fixed {len(fixed_data)} valid JSON entries.")
    save_to_json(fixed_data, output_file)

if __name__ == "__main__":
    try:
        # Fix the dataset format
        fix_json_format(INPUT_FILE, FIXED_OUTPUT_FILE)
    except Exception as e:
        print(f"An error occurred: {e}")
