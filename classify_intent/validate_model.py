import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, multilabel_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json

# Load the validation dataset
val_data_path = "/mnt/d/wsl_workspace/data/updated_upsampled_github_commit_dataset_validation.json"
val_data = pd.read_json(val_data_path)

# Define intents and mapping
predefined_intents = [
    "debug_code", 
    "add_feature", 
    "write_tests", 
    "optimize_performance",
    "document_code"
]
label_mapping = {label: idx for idx, label in enumerate(predefined_intents)}

# Convert labels to one-hot encoding
def one_hot_encode(labels, num_classes, label_mapping):
    encoding = [0] * num_classes
    for label in labels:
        encoding[label_mapping[label]] = 1
    return encoding

val_data["labels"] = val_data["labels"].apply(lambda labels: one_hot_encode(labels, len(predefined_intents), label_mapping))
val_dataset = Dataset.from_pandas(val_data)

# Ensure labels are floats
val_dataset = val_dataset.map(lambda x: {"labels": [float(v) for v in x["labels"]]})

# Load the fine-tuned model
model_path = "/mnt/d/wsl_workspace/fine_tuned_bert_v4"
model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    problem_type="multi_label_classification"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Tokenize dataset and ensure labels are floats
def tokenize_function(examples):
    tokens = tokenizer(
        examples["text"], 
        padding="max_length", 
        truncation=True, 
        max_length=256
    )
    tokens["labels"] = torch.tensor(examples["labels"], dtype=torch.float).tolist()  # Convert labels to float
    return tokens

tokenized_val = val_dataset.map(tokenize_function, batched=True)

# Define metrics for evaluation
def compute_metrics(pred):
    sigmoid = torch.nn.Sigmoid()
    predictions = sigmoid(torch.tensor(pred.predictions))
    predictions = (predictions > 0.5).int().numpy()  # Threshold for multi-label classification
    labels = torch.tensor(pred.label_ids).numpy()

    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    precision = precision_score(labels, predictions, average="weighted")
    recall = recall_score(labels, predictions, average="weighted")

    # Generate and save confusion matrix as a PNG file
    cm = multilabel_confusion_matrix(labels, predictions)
    fig, axes = plt.subplots(1, len(predefined_intents), figsize=(20, 5))
    for i, intent in enumerate(predefined_intents):
        sns.heatmap(cm[i], annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(f"Confusion Matrix for {intent}")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("True")
    plt.tight_layout()
    plt.savefig("validation_confusion_matrix.png")

    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# Training arguments for evaluation
training_args = TrainingArguments(
    output_dir="/mnt/d/wsl_workspace/validation/results",
    per_device_eval_batch_size=4,
    logging_dir="/mnt/d/wsl_workspace/validation/logs",
)

# Initialize the trainer
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Explicitly convert labels to Float
        labels = inputs["labels"].to(model.device).to(torch.float)
        inputs["labels"] = labels  # Ensure labels are correctly typed and device-aligned
        # print(inputs)
        outputs = model(**inputs)
        logits = outputs.logits

        # Use BCEWithLogitsLoss for multi-label classification
        loss_fn = torch.nn.BCEWithLogitsLoss()
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

trainer = CustomTrainer(
    model=model,
    args=training_args,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Evaluate the model
results = trainer.evaluate()

# Print and save results
print("Validation Results:")
print(results)

with open("/mnt/d/wsl_workspace/validation/fine_tuned_model_validation.json", "w") as file:
    json.dump(results, file, indent=4)

print("Validation metrics and confusion matrix saved.")
