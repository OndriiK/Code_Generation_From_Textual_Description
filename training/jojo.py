from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

model_name = "EleutherAI/gpt-neo-2.7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model with CPU/GPU Offloading
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Use fp16 to save memory
    device_map="auto",  # Auto-assign layers between CPU & GPU
    # offload_folder=r"D:\bakalarka\offload",  # Save large layers to disk
)

query = "Break this task into a numbered step-by-step guide (in the format: '1. <first-step> 2. <second-step, 3. <third-step>, ...): How do I use scikit-learn to classify the wine dataset?"
inputs = tokenizer(query, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
output = model.generate(**inputs, max_length=512)

print(tokenizer.decode(output[0], skip_special_tokens=True))
