import json
import random

# Define keyword substitutions
KEYWORD_REPLACEMENTS = {
    " optimize ": [" refactor ", " improve ", " enhance "],
    " fix ": [" debug ", " resolve ", " correct "],
    # " test ": [" cover by tests ", " validate "],
    " validate ": [" test ", " verify ", " check ", " ensure "],
    " analyze ": [" examine ", " inspect ", " study "],
    " deploy ": [" release ", " launch ", " deliver "],
    " build ": [" develop ", " create ", " construct "],
    " classify ": [" categorize ", " group ", " label "],
    " evaluate ": [" assess ", " measure ", " analyze "],
    " train ": [" fine-tune ", " build and improve "],
    " generate ": [" create ", " produce ", " synthesize "],
    " code ": [" implementation ", " script ", " program "],
    " debug ": [" troubleshoot ", " fix ", " resolve "]
}

KEYWORD_REPLACEMENTS2 = {
    " optimize ": [" refactor ", " improve ", " enhance "],
    " fix ": [" debug ", " resolve "],
    " test ": [" cover by tests ", " validate "],
    " validate ": [" test ", " verify ", " ensure "],
    " analyze ": [" examine ", " inspect "],
    " deploy ": [" release ", " launch "],
    " build ": [" develop ", " construct "],
    " classify ": [" categorize ", " label "],
    " evaluate ": [" assess ", " analyze "],
    " train ": [" fine-tune "],
    " generate ": [" produce ", " synthesize "],
    " code ": [" script ", " program "],
    " debug ": [" troubleshoot ", " resolve "]
}

def augment_question_with_keywords(question):
    """
    Generate variations of a given question by replacing keywords.
    """
    variations = []
    for keyword, replacements in KEYWORD_REPLACEMENTS.items():
        if keyword in question:
            change = random.choice(replacements)
            question = question.replace(keyword, change)
    return variations

def augment_answer_steps(answer_steps):
    """
    Generate variations of individual answer steps by replacing keywords.
    """
    augmented_steps = []
    flag = False
    for step in answer_steps:
        flag2 = False
        variations = [step]  # Include the original step
        for keyword, replacements in KEYWORD_REPLACEMENTS.items():
            if keyword in step:
                new_word = random.choice(replacements)
                augmented_steps.append(step.replace(keyword, new_word))
                flag = True
                flag2 = True
        if not flag2:
            augmented_steps.append(step)
    # print(augmented_steps)
    return flag, list(set(augmented_steps))

def augment_dataset(input_file, output_file, sampling_fraction=0.5):
    """
    Augment the dataset by adding variations to the questions.
    """
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    print(f"Loaded dataset with {len(data)} entries.")

    rephrasings = ["What is the process to", "Could you guide me on how to"]

    step_augment = 0
    question_augment = 0

    augmented_data = []
    for entry in data:
        original_question = entry["question"]
        original_answer_steps = entry["answer_steps"]

        # Add the original entry
        augmented_data.append({
            "question": original_question,
            "answer_steps": original_answer_steps
        })

        flag = False
        augmented_answer_steps = []
        # Generate variations for answer steps
        flag, augmented_answer_steps = augment_answer_steps(original_answer_steps)
        if flag:
            # sample_size = max(1, int(len(augmented_answer_steps) * 0.05))  # Ensure at least one variation is selected
            # sampled_step_variations = random.sample(augmented_answer_steps, sample_size)
            # for steps in sampled_step_variations:
            augmented_data.append({
                "question": original_question,
                "answer_steps": augmented_answer_steps
            })
            step_augment += 1

        keyword_variations = []
        # Generate keyword-based variations
        keyword_variations = augment_question_with_keywords(original_question)
        if len(keyword_variations) != 0:
            # Randomly sample a subset of variations
            sample_size = max(1, int(len(keyword_variations) * 0.3))  # Ensure at least one variation is selected
            sampled_variations = random.sample(keyword_variations, sample_size)
            for variation in sampled_variations:
                augmented_data.append({
                    "question": variation,
                    "answer_steps": original_answer_steps
                })
                question_augment += 1

        if ("How can I" in original_question):
            phrase = random.choice(rephrasings)
            final_question = original_question.replace("How can I", phrase)
            augmented_data.append({
                "question": final_question,
                "answer_steps": original_answer_steps
            })
            question_augment += 1

    random.shuffle(augmented_data)

    print(f"Saving augmented dataset with {len(augmented_data)} entries.")
    # Save the augmented dataset
    with open(output_file, 'w', encoding='utf-8') as file:
        print(f"Saving augmented dataset with {len(augmented_data)} entries.")
        json.dump(augmented_data, file, indent=4, ensure_ascii=False)

    print(f"Augmented dataset saved with {question_augment} question augmentations and {step_augment} step augmentations.")

# Specify the input and output file paths
input_file = r"D:\bakalarka\data\glaive_code_assist\task666667789432123_decomposition_dataset.json"  # Replace with the actual path if needed
output_file = r"D:\bakalarka\data\glaive_code_assist\augmented9_task_decomposition_dataset.json"

# Run the augmentation
augment_dataset(input_file, output_file)

print(f"Augmented dataset saved to {output_file}")
