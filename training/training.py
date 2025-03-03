import logging
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
import torch
import os

# Step 1: Setup Logging
log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(filename=os.path.join(log_dir, "training.log")),
    ],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
logger.info("Logging is set up.")

# Step 2: Load the dataset
logger.info("Loading dataset...")
dataset = load_dataset("code_x_glue_ct_code_refinement", "small")
train_data = dataset["train"]
val_data = dataset["validation"]

# Step 3: Initialize the model and tokenizer
logger.info("Initializing model and tokenizer...")
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Step 4: Preprocessing function
def preprocess_function(examples):
    inputs = [example["buggy_code"] for example in examples]
    targets = [example["fixed_code"] for example in examples]
    model_inputs = tokenizer(
        inputs, max_length=512, truncation=True, padding="max_length"
    )
    labels = tokenizer(
        targets, max_length=512, truncation=True, padding="max_length"
    )["input_ids"]
    model_inputs["labels"] = labels
    return model_inputs

logger.info("Preprocessing datasets...")
train_dataset = train_data.map(preprocess_function, batched=True)
val_dataset = val_data.map(preprocess_function, batched=True)

# Step 5: Define training arguments
logger.info("Setting up training arguments...")
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    save_steps=500,
    eval_steps=500,
    logging_steps=100,
    save_total_limit=3,
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True,
    report_to="none",
)

# Step 6: Initialize Trainer
logger.info("Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

# Step 7: Train the model
logger.info("Starting training...")
trainer.train()

# Step 8: Evaluate the model
logger.info("Evaluating model...")
results = trainer.evaluate()
logger.info("Evaluation Results: %s", results)

# Step 9: Save the model and tokenizer
logger.info("Saving model and tokenizer...")
model.save_pretrained("./fine_tuned_gpt2")
tokenizer.save_pretrained("./fine_tuned_gpt2")

# Optional: Print a success message
logger.info("Fine-tuning complete. Model saved to './fine_tuned_gpt2'.")
