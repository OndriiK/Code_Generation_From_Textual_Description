from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "EleutherAI/gpt-neo-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

from transformers import BitsAndBytesConfig

# Quantization configuration
config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=config,
    device_map="auto",
)

# Set pad_token_id to eos_token_id to avoid warnings
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Query input
query = "Break this task into a numbered step-by-step guide (in the format: '1. <first-step> 2. <second-step, 3. <third-step>, ...): How do I use scikit-learn to classify the wine dataset?"

# Tokenize input with attention mask
inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=512)
inputs = {key: tensor.to("cuda" if torch.cuda.is_available() else "cpu") for key, tensor in inputs.items()}

# Generate response
outputs = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_length=128,
    pad_token_id=tokenizer.pad_token_id,
)

# Decode and print the output
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(output_text)
