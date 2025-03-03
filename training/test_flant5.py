import json
import torch
from datasets import Dataset
import evaluate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load BLEU, ROUGE, and F1 metrics
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")

# Load the fine-tuned model
model_path = "/mnt/d/wsl_workspace/flan_t5_finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
model.eval()  # Set to evaluation mode

# Load the validation dataset
dataset_path = "/mnt/d/wsl_workspace/glaive_code_assistant/validation_task_decomposition_dataset.json"
with open(dataset_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Convert the dataset into Hugging Face format
dataset = Dataset.from_list([
    {
        "input": entry["question"],
        "output": "\n".join(entry["answer_steps"])
    }
    for entry in data
])

# Function to generate model outputs
def generate_prediction(question):
    inputs = tokenizer(question, return_tensors="pt", max_length=300, truncation=True)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=300)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Collect predictions and references
predictions, references = [], []
for entry in dataset:
    question = entry["input"]
    reference = entry["output"]
    
    # Generate model response
    prediction = generate_prediction(question)
    print(f"Question: {question}")
    
    # Store outputs
    predictions.append(prediction)
    references.append(reference)

predictions = [" ".join(pred.split()) for pred in predictions]
references = [" ".join(ref.split()) for ref in references]

# Ensure BLEU formatting
bleu_predictions = predictions  # Predictions should already be full sentences
bleu_references = [[ref] for ref in references] 

try:
    bleu_score = bleu.compute(predictions=predictions, references=[[ref] for ref in references])["bleu"]
except ValueError as e:
    print("Error computing BLEU:", e)
    print("Predictions (Sample):", predictions[:5])  # Print a sample for debugging
    print("References (Sample):", [[ref] for ref in references][:5])
    bleu_score = 0.0  # Fallback


# Compute ROUGE
rouge_score = rouge.compute(predictions=predictions, references=references)

# Handle cases where ROUGE returns a single float instead of an object
def safe_rouge_get(rouge_dict, key):
    """Handle cases where ROUGE returns a float instead of an object"""
    return rouge_dict[key].mid.fmeasure if isinstance(rouge_dict[key], dict) else rouge_dict[key]

# Compute F1
def compute_f1(predictions, references):
    exact_matches = [pred.strip() == ref.strip() for pred, ref in zip(predictions, references)]
    tp = sum(exact_matches)
    fp = len(predictions) - tp
    fn = len(references) - tp

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return {"f1": f1, "precision": precision, "recall": recall}

f1_metrics = compute_f1(predictions, references)

# Print Evaluation Results
print("\n📊 **Model Evaluation Results**")
print(f"🔹 BLEU Score: {bleu_score:.4f}")
print(f"🔹 ROUGE-1: {safe_rouge_get(rouge_score, 'rouge1'):.4f}")
print(f"🔹 ROUGE-2: {safe_rouge_get(rouge_score, 'rouge2'):.4f}")
print(f"🔹 ROUGE-L: {safe_rouge_get(rouge_score, 'rougeL'):.4f}")
print(f"🔹 F1 Score: {f1_metrics['f1']:.4f}")
print(f"🔹 Precision: {f1_metrics['precision']:.4f}")
print(f"🔹 Recall: {f1_metrics['recall']:.4f}")