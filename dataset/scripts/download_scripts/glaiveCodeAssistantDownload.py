import datasets
import json
import os

# Define dataset and desired output
DATASET_NAME = "glaiveai/glaive-code-assistant"
OUTPUT_FILE = r"D:\bakalarka\data\glaive_code_assist\subset12_glaive_code_assistant.json"
START_INDEX = 0  # Start from after the last downloaded batch
NUM_EXAMPLES = 136109  # Number of new examples to download

# Load the dataset
def load_dataset(dataset_name, split="train", start_index=START_INDEX, num_examples=NUM_EXAMPLES):
    """Loads the dataset and retrieves examples from start_index to start_index + num_examples."""
    print(f"Loading {dataset_name} dataset...")
    dataset = datasets.load_dataset(dataset_name, split=split)
    print(f"Dataset loaded. Total examples in the split: {len(dataset)}")

    # Get the new subset
    end_index = min(start_index + num_examples, len(dataset))
    subset = dataset.select(range(start_index, end_index))
    return subset

# Save to JSON
def save_to_json(subset, output_file):
    """Saves the subset to a JSON file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print(f"Saving {len(subset)} examples to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        for example in subset:
            f.write(json.dumps(example) + "\n")
    print(f"Saved successfully to {output_file}.")

if __name__ == "__main__":
    try:
        # Load and save the dataset
        subset = load_dataset(DATASET_NAME)
        save_to_json(subset, OUTPUT_FILE)
    except Exception as e:
        print(f"An error occurred: {e}")