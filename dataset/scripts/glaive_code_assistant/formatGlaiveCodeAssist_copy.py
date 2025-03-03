import json
import os
import re

# Define input and output file paths
INPUT_FILE = "/mnt/d/wsl_workspace/glaive_code_assistant/test.json"
FORMATTED_OUTPUT_FILE = "/mnt/d/wsl_workspace/glaive_code_assistant/formatted2_glaive_code_assistant.json"
MISFIT_OUTPUT_FILE = "/mnt/d/wsl_workspace/glaive_code_assistant/misfit_glaive_code_assistant.json"

# Save to JSON
def save_to_json(data, output_file):
    """Saves the data to a JSON file as a structured JSON list."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"Saving {len(data)} examples to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print(f"Saved successfully to {output_file}.")

# Extract first sentence from each step
def extract_first_sentence(text):
    """Extracts the first sentence from a given text, accounting for numbered steps."""
    result_text = ""
    sentences = re.split(r'(?<=[.!?:])\s+', text.strip())
    
    print(f"Sentences: {sentences}")
    # Handle cases where the step starts with "1. ", "2. ", etc.
    if re.match(r'^[1-9]\.', sentences[0]):
        # print("ANO")
        print(f"Sentences: {sentences}")
        if len(sentences) > 1:
            print("WHAT")
            print(f"Sentences[1]: {sentences[1]}")
            print(f"Sentences[1][-1]: {sentences[1][-1]}")
            if (sentences[1][-1] == ":"):
                print(f"Sentences[1]: {sentences[1]}")
                print(f"Sentences[1][0:len(sentences[1])-2]: {sentences[1][0:len(sentences[1])-1]}")
                result_text = sentences[0] + ' ' + sentences[1][0:len(sentences[1])-2]
            else:
                result_text = sentences[0] + ' ' + sentences[1]  # Combine first two segments
        return result_text

    if (sentences[0][-1] == ":"):
        result_text = sentences[0][0:len(sentences[0])-2] if sentences else text
    else:
        result_text = sentences[0] if sentences else text

    return result_text

# Extract structured step-by-step plans in correct order
def format_structured_plan(input_file, output_file):
    """Formats the dataset to extract structured plans with only the first sentence of each step, ensuring order and avoiding duplicates."""
    print(f"Formatting dataset for structured step-by-step plans from {input_file}...")
    step_groups = [
        ["First, ", "Firstly, ", "Step 1", "1. "],
        ["Second, ", "Secondly, ", "Step 2", "2. ", "Next, ", "Then, "],
        ["Third, ", "Thirdly, ", "Step 3", "3. ", "Next, ", "Then, "],
        ["Fourth, ", "Step 4", "4. ", "Next, ", "Then, "],
        ["Fifth, ", "Step 5", "5. ", "Finally, ", "Lastly, ", "Now, "],
        ["Sixth, ", "Step 6", "6. "],
        ["Seventh, ", "Step 7", "7. "]
    ]
    
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)  # Load the dataset
    
    formatted_data = []
    undetected_data = []
    for entry in data:
        structured_steps = []
        answer_text = entry.get("answer", "")
        last_position = 0  # Pointer to track last appended step position
        
        for group in step_groups:
            matches = []
            
            # Collect matches for all keywords in the group
            for keyword in group:
                if keyword.startswith("Step"):
                    # Adjust regex to exclude colon as sentence boundary for "Step N"
                    pattern = rf'{re.escape(keyword)}.*?(?:[\.!](?=\s|$)|[!?])'
                else:
                    # Regular regex for other keywords
                    pattern = rf'{re.escape(keyword)}.*?(?:[\.!:](?=\s|$)|[!?])'
                
                match = re.search(pattern, answer_text[last_position:], re.IGNORECASE)
                if match:
                    matches.append((match, answer_text.find(match.group(0), last_position)))
            
            # Process the match with the smallest index
            if matches:
                closest_match = min(matches, key=lambda x: x[1])  # Select match with smallest index
                match, step_position = closest_match
                step_sentence = match.group(0)
                
                if step_position >= last_position:  # Ensure valid order
                    if step_sentence[-1] == ":":  # Remove trailing colon
                        step_sentence = step_sentence[:-1]
                    structured_steps.append(step_sentence)
                    last_position = step_position + len(step_sentence)
        
        formatted_entry = {
            "question": entry.get("question", ""),
            "answer_steps": structured_steps
        }
        if structured_steps:
            formatted_data.append(formatted_entry)
        else:
            undetected_data.append(formatted_entry)

        print(f"Formatted entry: {formatted_entry}")
    
    print(f"Formatted {len(formatted_data)} examples with structured plans.")
    # save_to_json(formatted_data, output_file)
    # save_to_json(undetected_data, MISFIT_OUTPUT_FILE)

if __name__ == "__main__":
    try:
        # Format the dataset to structured step-by-step plans
        format_structured_plan(INPUT_FILE, FORMATTED_OUTPUT_FILE)
    except Exception as e:
        print(f"An error occurred: {e}")
