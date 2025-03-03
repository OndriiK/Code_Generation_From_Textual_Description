import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset

# Check if CUDA is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the DeepSeek model and tokenizer
model_name = "DeepSeek/1.1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model = model.to(device)

# Enable gradient checkpointing to save memory
model.gradient_checkpointing_enable()

# Load your dataset (replace 'path_to_your_dataset' with the actual path)
dataset = load_dataset("json", data_files="path_to_your_dataset")

dataset = dataset["train"].train_test_split(test_size=0.1, seed=42) 

# Preprocess the dataset
def preprocess_function(examples):
    # Format the question as input
    inputs = [f"Question: {q}" for q in examples["question"]]
    
    # Combine answer steps into a single string
    targets = ["\n".join(steps) if isinstance(steps, list) else steps for steps in examples["answer_steps"]]
    
    # Tokenize inputs
    model_inputs = tokenizer(inputs, max_length=256, truncation=True, padding="max_length")
    
    # Tokenize targets (labels)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=256, truncation=True, padding="max_length")
    
    # Add labels to the model inputs
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_datasets = {
    "train": dataset["train"].map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names),
    "test": dataset["test"].map(preprocess_function, batched=True, remove_columns=dataset["test"].column_names),
}

# Set up data collator for padding
from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding="max_length",
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./deepseek-finetuned",
    overwrite_output_dir=True,
    evaluation_strategy="steps",
    eval_steps=500,
    per_device_train_batch_size=1,  # Low batch size to save VRAM
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,  # Accumulate gradients to simulate a larger batch size
    learning_rate=5e-5,
    num_train_epochs=3,
    save_steps=1000,
    save_total_limit=2,
    fp16=True,  # Enable mixed precision
    logging_dir="./logs",
    logging_steps=100,
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train the model
trainer.train()

# Save the fine-tuned model
trainer.save_model("./deepseek-finetuned")
