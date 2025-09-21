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
dataset_path = "./intent_classification_dataset.json"
data = pd.read_json(dataset_path)

data = data.sample(frac=1, random_state=42).reset_index(drop=True)

predefined_intents = [
    "debug_code",
    "add_feature", 
    "write_tests",  
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
 
# Load model and tokenizer for the BERT BASE UNCASED model
model_name = "microsoft/codebert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(predefined_intents),
    problem_type="multi_label_classification"
)

# Custom Trainer to apply weighted loss
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Move labels to the same device as the model's logits
        labels = inputs.pop("labels").to(model.module.device)
        outputs = model(**inputs)
        logits = outputs.logits

        # Ensure class_weights_tensor is on the same device
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights_tensor.to(model.module.device))
        loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss

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
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_steps=500,
    save_total_limit=3,
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=6,
    weight_decay=0.01,
    lr_scheduler_type="linear",
    logging_dir="./logs",
    logging_steps=50,
    warmup_steps=200,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    gradient_accumulation_steps=4,
    fp16=True,
)

# Trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# Train
trainer.train()

# Create output directory if it doesn't exist
import os
save_path = "./fine_tuned_codebert"
os.makedirs(save_path, exist_ok=True)

# Save the fine-tuned model
print(f"Saving model to {save_path}...")
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"Model saved successfully to {save_path}")

# Verify the saved model
if os.path.exists(os.path.join(save_path, "pytorch_model.bin")):
    print("Model file confirmed to exist.")
else:
    print("WARNING: Model file not found after saving!")
