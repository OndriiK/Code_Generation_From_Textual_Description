from transformers import AutoTokenizer, AutoModelForCausalLM

# Step 1: Load the model and tokenizer
model_name = "gpt2"  # GPT-2 model for text generation
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Step 2: Define the prompt
prompt = "It was a rainy Saturday and Molly decided to stay home because she was feeling under the weather. "

# Step 3: Tokenize the prompt
inputs = tokenizer(prompt, return_tensors="pt")

# Step 4: Generate the continuation
output = model.generate(
    inputs.input_ids,
    max_length=200,  # Limit the length of the generated story
    num_return_sequences=1,  # Generate one story
    temperature=0.7,  # Control creativity (lower = less creative, higher = more)
    top_k=50,  # Limit to top-k most probable tokens
    top_p=0.95,  # Nucleus sampling for probabilistic selection
    do_sample=True  # Enable sampling for more varied outputs
)

# Step 5: Decode the generated text
story = tokenizer.decode(output[0], skip_special_tokens=True)

# Print the short story
print("Generated Short Story:")
print(story)
