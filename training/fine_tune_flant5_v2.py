import json
import torch
from datasets import Dataset
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)
from accelerate import Accelerator  # Importing accelerate
from transformers import GenerationConfig

# Load your dataset
dataset_path = r"D:\bakalarka\data\glaive_code_assist\task32_decomposition_dataset.json"
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
model_name = "google/flan-t5-base"  # Using the large variant

# Reduce max_length further to save memory
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

def preprocess_function(examples):
    inputs = tokenizer(
        examples["input"],
        max_length=290,  # Further reduced from 300
        truncation=True,
        padding="max_length"
    )
    targets = tokenizer(
        examples["output"],
        max_length=290,  # Further reduced from 300
        truncation=True,
        padding="max_length"
    )
    targets["input_ids"] = [
        [(token if token != tokenizer.pad_token_id else -100) for token in seq]
        for seq in targets["input_ids"]
    ]
    inputs["labels"] = targets["input_ids"]
    return inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# print(f'Input IDs: {tokenized_datasets["train"][0]["input_ids"]}')
# print(f'Labels: {tokenized_datasets["train"][0]["labels"]}')

# print("Dataset tokenization complete:")
# print(tokenized_datasets["train"][0])
# print("Validation set tokenization complete:")
# print(tokenized_datasets["test"][0])

# Load the model with gradient checkpointing enabled
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.gradient_checkpointing_enable()  # Saves memory at the cost of extra computation

# generation_config = GenerationConfig.from_pretrained(model_name)
# model.config.update(vars(generation_config)) 

# print("Model Config:")
# print(model.config)
# print("\nGeneration Config:")
# print(model.generation_config)

# Offload model and optimizer to Accelerator
# model, optimizer = accelerator.prepare(model, torch.optim.AdamW(model.parameters(), lr=1.5e-5))

# Define a data collator with dynamic padding
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding="longest",
    label_pad_token_id=-100
)

model.config.reduction = "mean"

# Define training arguments optimized for low GPU memory
training_args = Seq2SeqTrainingArguments(
    output_dir=r"D:\bakalarka\result_flant5_large_v2",
    eval_strategy="epoch",  # Use `evaluation_strategy` for the latest library version
    save_strategy="epoch",
    logging_dir=r"D:\bakalarka\logs",
    learning_rate=5e-6,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    num_train_epochs=8,
    weight_decay=0.01,
    max_grad_norm=1.0,
    save_total_limit=1,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    fp16=False,
    report_to="none",
    predict_with_generate=True,  # Seq2Seq-specific argument for text generation
    use_cache=False,
)

# print("Final Model Config:")
# print(model.config)

# print("\nFinal Training Arguments:")
# print(training_args)

# 🚨 Patch TrainingArguments to ensure no 'generation_config' is referenced
# if hasattr(transformers.TrainingArguments, "generation_config"):
#     del transformers.TrainingArguments.generation_config

losses = []

class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")  # Remove labels from inputs
        outputs = model(**inputs)  # Forward pass
        logits = outputs.logits  # Model predictions
        # print("Training logits shape:", outputs.logits.shape)
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))  # Compute loss
        # Log the loss explicitly
        self.state.log_history.append({"loss": loss.item()})
        # losses.append(loss.item())

        return (loss, outputs) if return_outputs else loss
    
    def training_step(self, model, inputs, num_items_in_batch):
        model.train()
        inputs = self._prepare_inputs(inputs)
        # Forward pass
        outputs = model(**inputs)
        loss = self.compute_loss(model, inputs)
        # Normalize loss if gradient accumulation is used
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        
        if loss is None:
            raise RuntimeError("Loss is None, something went wrong in the forward pass.")

        loss.backward()  # Ensure backward is called here
        # Update model parameters

        # Log gradients for debugging
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         print(f"{name}: Gradient norm = {param.grad.norm().item()}")
        #     else:
        #         print(f"{name}: No gradient")
        
        self.optimizer.step()
        
        # Clear previous gradients
        if (self.state.global_step + 1) % self.args.gradient_accumulation_steps == 0:
            self.lr_scheduler.step()
        
        self.optimizer.zero_grad()
        

        
        return loss.detach()



trainer = CustomSeq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# trainer = Seq2SeqTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_datasets["train"],
#     eval_dataset=tokenized_datasets["test"],
#     tokenizer=tokenizer,
#     data_collator=data_collator,
#     callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
# )

print(f"Training on device: {training_args.device}")
print(f"Model device: {next(model.parameters()).device}")

# def check_grad_norm(trainer):
#     for name, param in trainer.model.named_parameters():
#         if param.grad is not None:
#             print(f"{name} grad norm: {param.grad.norm()}")

# trainer.add_callback(check_grad_norm)

# def check_gradients(trainer):
#     # print("Checking gradients...")
#     for name, param in trainer.model.named_parameters():
#         if param.grad is not None:
#             grad_norm = param.grad.norm().item()
#             print(f"Gradient norm for {name}: {grad_norm}")
#         else:
#             print(f"No gradient for {name}")

# class LossMonitorCallback(TrainerCallback):
#     def on_log(self, args, state, control, logs=None, **kwargs):
#         if logs is not None:
#             print(f"Raw Loss Logged: {logs.get('loss', 'N/A')}")
# trainer.add_callback(LossMonitorCallback())

# class GradientCheckCallback(TrainerCallback):
#     def on_step_end(self, args, state, control, logs=None, **kwargs):
#         model = kwargs["model"]
#         for name, param in model.named_parameters():
#             if param.grad is not None:
#                 print(f"{name}: Gradient norm = {param.grad.norm().item()}")
#             else:
#                 print(f"{name}: No gradient")

# # Add the callback to your trainer
# trainer.add_callback(GradientCheckCallback())

# model.train()

# batch = next(iter(trainer.get_train_dataloader()))
# batch = {k: v.to(training_args.device) for k, v in batch.items()}
# outputs = model(**batch)
# # print(batch["attention_mask"])
# print("Batch Loss:", outputs.loss)

# print("TrainingArguments:")
# print(training_args.__dict__)

# print("\nSeq2SeqTrainer Parameters:")
# print(trainer.__dict__)

# trainer.evaluate()

# Test Loss Computation on a Single Batch
# print("\nTesting loss computation on a single batch:")

# Get a small batch from the train dataset
# batch = next(iter(trainer.get_train_dataloader()))
# # Move the batch to the correct device
# batch = {k: v.to(training_args.device) for k, v in batch.items()}

# # Perform forward pass
# with torch.no_grad():
#     outputs = trainer.model(**batch)

# # Print the computed loss
# print("Computed loss on test batch:", outputs.loss.item())

# Use Accelerator to manage training
# with accelerator.main_process_first():
trainer.train()

metrics = trainer.evaluate()
print(metrics)

# Save the fine-tuned model
model.save_pretrained(r"D:\bakalarka\fine_tuned_flant5_large_v2")
tokenizer.save_pretrained(r"D:\bakalarka\fine_tuned_flant5_large_v2")

# Save the final list into a txt file
with open('losses.txt', 'w') as file:
    for loss in losses:
        file.write(f"{loss}\n")

print("Fine-tuning complete. Model saved.")
