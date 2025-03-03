import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the fine-tuned model and tokenizer
model_name = r"D:\bakalarka\gpt_neo_1_3_B_lora_finetuned"  # Path to your fine-tuned model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Query to test the model
query = "Decompose this query from the user into an actionable plan with steps: How do I use scikit-learn to classify the wine dataset?"

# Tokenize the input query
inputs = tokenizer(f"Question: {query}", return_tensors="pt").to(device)

# Generate the response
outputs = model.generate(
    **inputs,
    max_length=256,
    num_beams=5,
    early_stopping=True
)

# Decode and print the response
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Model Response:")
print(response)
