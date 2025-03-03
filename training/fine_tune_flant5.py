import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
import evaluate

# Load the BLEU metric
bleu = evaluate.load("bleu")

# Load your dataset
dataset_path = "/mnt/d/wsl_workspace/glaive_code_assistant/task2_decomposition_dataset.json"
with open(dataset_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Prepare the dataset in Hugging Face format
dataset = Dataset.from_list([
    {
        "input": entry["question"],
        "output": "\n".join(entry["answer_steps"])
    }
    for entry in data
])

# Split into train and validation sets
dataset = dataset.train_test_split(test_size=0.1, seed=42)

# Load the tokenizer
model_name = "google/flan-t5-small"  # Adjust if you want to use 'base' or 'large'
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenization
def preprocess_function(examples):
    inputs = tokenizer(
        examples["input"],
        max_length=256,
        truncation=True,
        padding="max_length"
    )
    targets = tokenizer(
        examples["output"],
        max_length=256,
        truncation=True,
        padding="max_length"
    )
    inputs["labels"] = targets["input_ids"]
    return inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Load the model
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Define a data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True
)

# Metrics: BLEU and F1 computation
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Tokenize for BLEU
    bleu_preds = [pred.split() for pred in decoded_preds]
    bleu_labels = [[label.split()] for label in decoded_labels]
    bleu_score = bleu.compute(predictions=bleu_preds, references=bleu_labels)["bleu"]

    # F1 computation
    exact_matches = [
        pred.strip() == label.strip()
        for pred, label in zip(decoded_preds, decoded_labels)
    ]
    tp = sum(exact_matches)
    fp = len(decoded_preds) - tp
    fn = len(decoded_labels) - tp

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return {
        "bleu": bleu_score,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

# Define training arguments optimized for low GPU memory
training_args = TrainingArguments(
    output_dir="/mnt/d/wsl_workspace/results_flant5",
    evaluation_strategy="no",
    save_strategy="epoch",
    logging_dir="/mnt/d/wsl_workspace/logs",
    learning_rate=2e-5,
    per_device_train_batch_size=1,  # Reduce batch size to save memory
    gradient_accumulation_steps=8,  # Accumulate gradients to simulate a larger batch size
    num_train_epochs=6,
    weight_decay=0.01,
    save_total_limit=1,
    logging_steps=50,
    fp16=True,  # Enable mixed precision if supported
    dataloader_num_workers=0,
)

# Modify Trainer to handle predictions for BLEU
class CustomTrainer(Trainer):
    def predict(self, test_dataset):
        """Override predict to manually handle generation."""
        dataloader = self.get_test_dataloader(test_dataset)
        predictions = []
        references = []

        for batch in dataloader:
            inputs = {k: v.to(self.args.device) for k, v in batch.items() if k != "labels"}
            with torch.no_grad():
                generated_tokens = self.model.generate(
                    **inputs,
                    max_length=512,
                    num_beams=3  # Reduce number of beams for efficiency
                )
            predictions.extend(generated_tokens.cpu().numpy())
            references.extend(batch["labels"].numpy())

        return (predictions, references)

# Define the Trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("/mnt/d/wsl_workspace/flan_t5_finetuned")
tokenizer.save_pretrained("/mnt/d/wsl_workspace/flan_t5_finetuned")

print("Fine-tuning complete. Model saved.")
