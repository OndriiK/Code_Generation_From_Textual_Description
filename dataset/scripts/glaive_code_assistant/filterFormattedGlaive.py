import json
import os
import re

# Define input and output file paths
INPUT_FILE = r"D:\bakalarka\data\glaive_code_assist\task666667789432_decomposition_dataset.json"
OUTPUT_FILE = r"D:\bakalarka\data\glaive_code_assist\task666667789432123_decomposition_dataset.json"
STATIC_PROMPT = "Decompose this query from the user into an actionable plan with steps: "

# Save to JSON
def save_to_json(data, output_file):
    """Saves the data to a JSON file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"Saving {len(data)} examples to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print(f"Saved successfully to {output_file}.")


# Calculate average character length of "question" field
def calculate_avg_question_length(input_file):
    """Calculates and prints the average character length of the 'question' field."""
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)  # Load dataset
    
    question_lengths = [len(entry.get("question", "")) for entry in data]
    avg_length = sum(question_lengths) / len(question_lengths) if question_lengths else 0
    print(f"Average character length of 'question': {avg_length:.2f}")

# Filter dataset
def filter_dataset(input_file, output_file):
    """Filters the dataset based on the given criteria."""
    print(f"Filtering dataset from {input_file}...")
    
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)  # Load the dataset
    max_length = 0
    min_length = 3000
    min_len_step = 3000
    max_len_step = 0
    filtered_data = []
    for entry in data:
        answer_steps = entry.get("answer_steps", [])
        question = entry.get("question", "")

        if len(question) > max_length:
            max_length = len(question)
        if len(question) < min_length:
            min_length = len(question)
        # print(len(question))

        if ("\n" in question) or ("\\" in question) or ("\"" in question) or ("http" in question):
            continue

        if len(question) > 512:
            continue
        
        # Exclude examples with less than 2 actionable steps
        if len(answer_steps) < 3:
            continue
        
        number = False
        mixed_bullets_flag = False

        

        asterisk_count = 0
        for step in answer_steps:
            if step[-2:] == "**": # Exclude steps with no content
                asterisk_count += 1
            
        if asterisk_count >= len(answer_steps)-1:
            continue

        # Exclude examples where the first step starts with specific prefixes
        first_step = answer_steps[0].strip()
        invalid_prefixes = ["Next, ", "Then, ", "Finally, ", "2. ", "3. ", "4. ", "5. ", "Step 2", "Step 3"]
        if any(first_step.startswith(prefix) for prefix in invalid_prefixes):
            continue
        
        number_prefixes = ["1. ", "2. ", "3. ", "4. ", "5. ", "6. ", "7. ", "8. ", "9. ", "10. ", "1, ", "2, ", "3, ", "4, ", "5, ", "6, ", "7, ", "8, ", "9, ", "10, "]
        if first_step.startswith("1. ") or first_step.startswith("1, "):
            number = True
        
        step_index = 0

        for step in answer_steps[1:]:
            if number:
                if not(any(step.startswith(prefix) for prefix in number_prefixes)):
                    if (step.startswith("Now, ") or step.startswith("Finally, ")) and step_index == len(answer_steps)-1:
                        continue
                    mixed_bullets_flag = True
                    break
            else:
                if any(step.startswith(prefix) for prefix in number_prefixes):
                    mixed_bullets_flag = True
                    break
            step_index += 1

        if mixed_bullets_flag:
            continue

        faulty_step_flag = False
        idx = 0
        for step in answer_steps:
            if ("\"" in step) or ("\\" in step) or ("http" in step):
                faulty_step_flag = True
                break
            if len(step) > 400:
                faulty_step_flag = True
                break
            if len(step) < 50:
                faulty_step_flag = True
                break

            if (idx == len(answer_steps)-1) and (step.startswith("Next, ") or step.startswith(f"{idx+1}. Next, ")):
                faulty_step_flag = True
                break
                
            idx += 1

            # if len(step) > max_len_step:
            #     max_len_step = len(step)
            # if len(step) < min_len_step:
            #     min_len_step = len(step)
        
        if faulty_step_flag:
            continue

        idx = 1
        for i in range(len(answer_steps)):
            answer_steps[i] = re.sub(r'^\S+', f"{idx}.", answer_steps[i])
            idx += 1
        # entry["question"] = STATIC_PROMPT + entry["question"]
        
        # Include the entry if it passes all filters
        filtered_data.append(entry)
    
    print(f"Max length: {max_length}")
    print(f"Min length: {min_length}")
    print(f"Max step length: {max_len_step}")
    print(f"Min step length: {min_len_step}")
    print(f"Filtered dataset to {len(filtered_data)} examples.")
    save_to_json(filtered_data, output_file)

if __name__ == "__main__":
    try:
        # Filter the dataset based on the criteria
        filter_dataset(INPUT_FILE, OUTPUT_FILE)
        calculate_avg_question_length(INPUT_FILE)
    except Exception as e:
        print(f"An error occurred: {e}")
