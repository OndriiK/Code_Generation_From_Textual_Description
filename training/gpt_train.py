import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# Load the GPT-Neo model with 4-bit quantization
model_name = "EleutherAI/gpt-neo-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Enable 4-bit quantization for memory savings
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")

# Ensure pad token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# Apply QLoRA for trainable adapters
lora_config = LoraConfig(
    r=8,  # Rank of LoRA matrices (low-rank adaptation)
    lora_alpha=16,  # Scaling factor for LoRA updates
    target_modules=["attn.q_proj", "attn.v_proj", "attn.k_proj", "attn.out_proj", "mlp.c_proj"],  # Applies LoRA to attention layers
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# Load and split dataset
dataset = load_dataset("json", data_files=r"D:\bakalarka\data\glaive_code_assist\task32_decomposition_dataset.json")
dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)

# Preprocessing function
def preprocess_function(examples):
    inputs = [f"Question: {q}" for q in examples["question"]]
    targets = [f"Answer Steps: {' '.join(steps)}" for steps in examples["answer_steps"]]
    model_inputs = tokenizer(inputs, max_length=200, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=200, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Tokenize both splits
tokenized_datasets = {
    "train": dataset["train"].map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names),
    "test": dataset["test"].map(preprocess_function, batched=True, remove_columns=dataset["test"].column_names),
}

# Training arguments with QLoRA
training_args = TrainingArguments(
    output_dir=r"D:\bakalarka\result_gpt_neo_1_3_B",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-4,  # Higher LR is often better with LoRA
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,  # Higher steps to reduce memory usage
    num_train_epochs=3,
    logging_dir=r"D:\bakalarka\logs",
    logging_steps=50,
    save_total_limit=2,
    fp16=True,  # Mixed precision
    optim="paged_adamw_8bit",  # Optimized optimizer for low-memory training
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
)

# Train and save LoRA adapters
trainer.train()
trainer.save_model(r"D:\bakalarka\gpt_neo_1_3_B_lora_finetuned")
print("Fine-tuning complete. Model saved.")
