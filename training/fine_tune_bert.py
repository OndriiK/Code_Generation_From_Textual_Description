import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
import numpy as np
from sklearn.metrics import precision_score

# Load your custom dataset (JSON format)
dataset_path = "/mnt/d/wsl_workspace/data/final_intent_dataset.json"  # Replace with your dataset file path
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

# Split dataset into train and validation sets
train_test_split = 0.85
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
        max_length=256  # Adjust max_length as needed for your dataset
    )

# Tokenize datasets
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="/mnt/d/wsl_workspace/results3",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_steps=500,
    save_total_limit=5,  # Keep more checkpoints
    learning_rate=2e-5,
    lr_scheduler_type="linear",  # Dynamic learning rate
    warmup_steps=500,  # Gradual increase in learning rate
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="/mnt/d/wsl_workspace/logs3",
    logging_steps=50,  # More frequent logs
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    gradient_accumulation_steps=4,
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
    precision = precision_score(labels, predictions, average="weighted")
    return {"accuracy": acc, "f1": f1, "precision": precision}


# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("/mnt/d/wsl_workspace/fine_tuned_bert_v3")
tokenizer.save_pretrained("/mnt/d/wsl_workspace/fine_tuned_bert_v3")
