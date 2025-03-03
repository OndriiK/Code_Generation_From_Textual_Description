import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

# Load the Flan-T5 Large model and tokenizer
model_name = "google/flan-t5-large"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Set the pad token if it's not defined
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Apply QLoRA
lora_config = LoraConfig(
    r=16,  # Rank size for LoRA
    lora_alpha=32,  # Scaling factor
    target_modules=["q", "v", "k", "o", "ffn"],  # Apply LoRA on specific layers
    lora_dropout=0.1,
    bias="all",
    task_type="SEQ_2_SEQ_LM",
)
model = get_peft_model(model, lora_config)

# Load and preprocess the dataset
dataset_path = r"D:\bakalarka\data\glaive_code_assist\augmented9_task_decomposition_dataset.json"
dataset = load_dataset("json", data_files=dataset_path)["train"].train_test_split(test_size=0.1, seed=42)

def preprocess_function(examples):
    inputs = [f"Task: {q}" for q in examples["question"]]
    targets = ["; ".join(steps) for steps in examples["answer_steps"]]
    model_inputs = tokenizer(inputs, max_length=256, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=256, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Tokenize the train and test splits
tokenized_datasets = {
    "train": dataset["train"].map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names),
    "test": dataset["test"].map(preprocess_function, batched=True, remove_columns=dataset["test"].column_names),
}

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=r"D:\bakalarka\flan5LARGE_result_v2",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,  # Adjust for VRAM limitations
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,  # Simulates a larger batch size
    num_train_epochs=8,
    logging_dir=r"D:\bakalarka\logs",
    logging_steps=50,
    save_total_limit=2,
    warmup_steps=200,
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    predict_with_generate=True,  # Enables text generation for evaluation
    # Disable fp16 for now
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

# Define the trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=6)],
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
trainer.save_model(r"D:\bakalarka\flant5LARGE_v2")
print("Fine-tuning complete. Model saved.")
