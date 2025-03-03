import json
import os

# Define input and output file paths
INPUT_FILE = r"D:\bakalarka\data\glaive_code_assist\FIXEDFINAL_glaive_code_assistant.json"
FILTERED_OUTPUT_FILE = r"D:\bakalarka\data\glaive_code_assist\FILTEREDFINAL_glaive_code_assistant.json"

# Save to JSON
def save_to_json(data, output_file):
    """Saves the data to a JSON file as a structured JSON list."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"Saving {len(data)} examples to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print(f"Saved successfully to {output_file}.")

# Filter dataset for step-by-step plans
def filter_step_by_step(input_file, output_file):
    """Filters examples with answers containing step-by-step plans."""
    print(f"Filtering dataset for step-by-step plans from {input_file}...")
    step_by_step_keywords = ["1. First", "2. ", "3. ", "First", "Next", "Finally"]

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)  # Load the entire dataset as a list
    
    # filtered_data = [entry for entry in data if any(keyword in entry.get("answer", "") for keyword in step_by_step_keywords)]
    invalid = 0
    continued = 0
    filtered_data = []
    lengths = []
    max_length = 0
    min_length = 3000
    for entry in data:
        if len(entry.get("question", "")) < 10 or len(entry.get("question", "")) > 600:
            continued += 1
            if len(entry.get("question", "")) > max_length:
                max_length = len(entry.get("question", ""))
            if len(entry.get("question", "")) < min_length:
                min_length = len(entry.get("question", ""))
            # lengths.append(len(entry.get("question", "")))
            continue

        if "\n" in entry.get("quetsion", ""):
            continue
        
        if "1. First" in entry.get("answer", ""):
            filtered_data.append(entry)
        elif "follow these steps" in entry.get("answer", "").lower() or "following the steps" in entry.get("answer", "").lower():
            filtered_data.append(entry)
        elif "First, " in entry.get("answer", "") or "firstly, " in entry.get("answer", "").lower():
            filtered_data.append(entry)
        elif "step-by-step" in entry.get("answer", "").lower() or "step by step" in entry.get("answer", "").lower():
            filtered_data.append(entry)
        elif "Next, " in entry.get("answer", ""):
            filtered_data.append(entry)
        elif "here are the steps" in entry.get("answer", "").lower():
            filtered_data.append(entry)
        elif "finally, " in entry.get("answer", "").lower():
            filtered_data.append(entry)
        # elif ":\n\n1. " in entry.get("answer", "").lower():
        #     filtered_data.append(entry)
        elif "Secondly, " in entry.get("answer", ""):
            filtered_data.append(entry)
        else:
            invalid += 1

    print(f"Invalid: {invalid}")
    print(f"Continued: {continued}")
    print(f"Max length: {max_length}")
    print(f"Min length: {min_length}")
    # print(f"Lengths: {lengths}")

    print(f"Filtered {len(filtered_data)} examples containing step-by-step plans.")
    
    # Save filtered data
    save_to_json(filtered_data, output_file)


# Calculate average character length of "question" field
def calculate_avg_question_length(input_file):
    """Calculates and prints the average character length of the 'question' field."""
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)  # Load dataset
    
    question_lengths = [len(entry.get("question", "")) for entry in data]
    avg_length = sum(question_lengths) / len(question_lengths) if question_lengths else 0
    print(f"Average character length of 'question': {avg_length:.2f}")

if __name__ == "__main__":
    try:
        # Filter the dataset for step-by-step plans
        filter_step_by_step(INPUT_FILE, FILTERED_OUTPUT_FILE)
        calculate_avg_question_length(INPUT_FILE)
    except Exception as e:
        print(f"An error occurred: {e}")
