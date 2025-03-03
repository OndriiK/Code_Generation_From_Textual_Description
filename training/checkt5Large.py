from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the fine-tuned model and tokenizer
model_name = r"D:\bakalarka\flant5LARGE_v2"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Define the query
# query = "Decompose this query from the user into an actionable plan with steps: How can I use datasets to fine-tune a pre-trained large language model?"
# Prompt the user to input the query
query = input("Please enter your query: ")

# Tokenize the input
inputs = tokenizer(query, return_tensors="pt", max_length=512, truncation=True, padding="max_length")

# Generate a response
outputs = model.generate(inputs["input_ids"], max_length=512, num_beams=4, early_stopping=True)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Model Response:")
print(response)