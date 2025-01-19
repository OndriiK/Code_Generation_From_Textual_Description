import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np

# Load your custom dataset (JSON format)
dataset_path = "intent_dataset.json"  # Replace with your dataset file path
data = pd.read_json(dataset_path)

predefined_intents = [
    "debug_code", 
    "add_feature", 
    "write_tests", 
    "optimize_performance", 
    "document_code"
]

# Ensure consistent column names
data = data.rename(columns={"text": "text", "label": "label"})

# Normalize labels to ensure all are lists
def normalize_labels(label):
    if isinstance(label, list):  # Already a list
        return label
    return [label]  # Convert single label to a list

# Apply normalization
data["label"] = data["label"].apply(normalize_labels)

# Define mapping
label_mapping = {label: idx for idx, label in enumerate(predefined_intents)}

# Convert labels into one-hot encoded format
def one_hot_encode(labels, num_classes):
    encoding = np.zeros(num_classes)
    for label in labels:
        encoding[label_mapping[label]] = 1
    return encoding

data["label"] = data["label"].apply(lambda labels: one_hot_encode(labels, len(predefined_intents)))

# Split dataset into train and validation sets
train_test_split = 0.8
train_data = data.sample(frac=train_test_split, random_state=42)
val_data = data.drop(train_data.index)

# Convert to Hugging Face Dataset format
train_dataset = Dataset.from_pandas(train_data)
val_dataset = Dataset.from_pandas(val_data)

# Load tokenizer and model
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=len(predefined_intents), problem_type="multi_label_classification"
)

# Tokenization function
def tokenize_function(examples):
    return tokenizer(
        examples["text"], 
        padding="max_length", 
        truncation=True, 
        max_length=128  # Adjust max_length as needed for your dataset
    )

# Tokenize datasets
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

# Define evaluation metric
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(pred):
    sigmoid = torch.nn.Sigmoid()
    predictions = sigmoid(torch.tensor(pred.predictions))
    predictions = (predictions > 0.5).int().numpy()  # Convert to numpy array
    labels = torch.tensor(pred.label_ids).numpy()  # Convert to numpy array

    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": acc, "f1": f1}


# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_bert")
tokenizer.save_pretrained("./fine_tuned_bert")
