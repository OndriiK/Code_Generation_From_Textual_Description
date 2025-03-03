import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
dataset_path = "/mnt/d/wsl_workspace/data/final_intent_dataset.json"
data = pd.read_json(dataset_path)

data = data.sample(frac=1, random_state=42).reset_index(drop=True)

predefined_intents = [
    "debug_code", 
    "add_feature", 
    "write_tests", 
    "optimize_performance", 
    "document_code"
]

# Ensure consistent column names
data = data.rename(columns={"text": "text", "label": "label"})

# Define mapping
label_mapping = {label: idx for idx, label in enumerate(predefined_intents)}

# Convert labels into one-hot encoded format
def one_hot_encode(labels, num_classes):
    encoding = np.zeros(num_classes)
    for label in labels:
        encoding[label_mapping[label]] = 1
    return encoding

data["labels"] = data["labels"].apply(lambda labels: one_hot_encode(labels, len(predefined_intents)))

# Split dataset
train_test_split = 0.85
train_data = data.sample(frac=train_test_split, random_state=42)
val_data = data.drop(train_data.index)

train_dataset = Dataset.from_pandas(train_data)
val_dataset = Dataset.from_pandas(val_data)

# Compute class weights
all_labels = np.stack(train_data["labels"].values)  # Convert list of arrays to 2D numpy array
flattened_labels = []
for row in all_labels:
    flattened_labels.extend(np.where(row == 1)[0])  # Extract class indices for each row

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(predefined_intents)),
    y=flattened_labels
)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
print("Class Weights:", class_weights_tensor)

# Load model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(predefined_intents),
    problem_type="multi_label_classification"
)

# Modify the loss function to include class weights
def compute_loss(inputs, targets):
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights_tensor)
    return loss_fn(inputs, targets)

model.config.loss_fn = compute_loss

# Tokenization
def tokenize_function(examples):
    return tokenizer(
        examples["text"], 
        padding="max_length", 
        truncation=True, 
        max_length=256
    )

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)

# Metrics
def compute_metrics(pred):
    sigmoid = torch.nn.Sigmoid()
    predictions = sigmoid(torch.tensor(pred.predictions))
    predictions = (predictions > 0.5).int().numpy()
    labels = torch.tensor(pred.label_ids).numpy()

    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    precision = precision_score(labels, predictions, average="weighted")

    # Confusion Matrix
    if hasattr(pred, "confusion_matrix"):
        cm = confusion_matrix(np.argmax(labels, axis=1), np.argmax(predictions, axis=1))
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=predefined_intents, yticklabels=predefined_intents)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()

    return {"accuracy": acc, "f1": f1, "precision": precision}

# Training arguments
training_args = TrainingArguments(
    output_dir="/mnt/d/wsl_workspace/results4",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_steps=500,
    save_total_limit=5,
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=8,
    weight_decay=0.01,
    lr_scheduler_type="linear",
    logging_dir="/mnt/d/wsl_workspace/logs4",
    logging_steps=50,
    warmup_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    gradient_accumulation_steps=8,
    fp16=True,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

# Train
trainer.train()

# Save the fine-tuned model
model.save_pretrained("/mnt/d/wsl_workspace/fine_tuned_bert_v4")
tokenizer.save_pretrained("/mnt/d/wsl_workspace/fine_tuned_bert_v4")
