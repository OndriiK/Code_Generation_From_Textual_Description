from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_path = "/mnt/d/wsl_workspace/flan_t5_finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

input_text = "How can I install Python 3 on AWS EC2?"
inputs = tokenizer(input_text, return_tensors="pt")

outputs = model.generate(**inputs, max_length=300)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))