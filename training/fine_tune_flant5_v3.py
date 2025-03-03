import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)

# Load your dataset
dataset_path = "/mnt/d/wsl_workspace/glaive_code_assistant/task3_decomposition_dataset.json"
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
model_name = "google/flan-t5-large"  # Using the large variant
tokenizer = AutoTokenizer.from_pretrained(model_name)

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

# Load the model with gradient checkpointing enabled
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.gradient_checkpointing_enable()

# Define a data collator with dynamic padding
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding="longest"
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="/mnt/d/wsl_workspace/results_flant5_large",
    evaluation_strategy="no",
    save_strategy="epoch",
    logging_dir="/mnt/d/wsl_workspace/logs",
    learning_rate=1.5e-5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=20,
    num_train_epochs=8,
    weight_decay=0.01,
    save_total_limit=1,
    logging_steps=100,
    fp16=True,
    dataloader_num_workers=0,
)

# Define the Trainer (NO manual accelerator usage)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("/mnt/d/wsl_workspace/flan_t5_finetuned_large")
tokenizer.save_pretrained("/mnt/d/wsl_workspace/flan_t5_finetuned_large")

print("Fine-tuning complete. Model saved.")
