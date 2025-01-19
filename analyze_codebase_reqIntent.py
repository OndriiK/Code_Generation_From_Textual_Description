import os
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM

# Initialize CodeBERT and Flan-T5
codebert_model_name = "microsoft/codebert-base"
tokenizer_codebert = AutoTokenizer.from_pretrained(codebert_model_name)
codebert = AutoModel.from_pretrained(codebert_model_name)

flan_t5_model_name = "google/flan-t5-base"
tokenizer_flan_t5 = AutoTokenizer.from_pretrained(flan_t5_model_name)
flan_t5 = AutoModelForSeq2SeqLM.from_pretrained(flan_t5_model_name)

def parse_repository(repo_path):
    """
    Parses a Python repository to extract code structures.
    Returns a dictionary with file paths and corresponding code.
    """
    repo_data = {}
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        repo_data[file_path] = f.read()
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    return repo_data

def get_hierarchical_embeddings(repo_data, tokenizer, model):
    """
    Generates embeddings for each file and aggregates them by directory.
    """
    embeddings = {}
    for file_path, code in repo_data.items():
        inputs = tokenizer(code, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        file_embedding = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
        embeddings[file_path] = file_embedding

    # Aggregate embeddings by directory
    directory_embeddings = {}
    for file_path, embedding in embeddings.items():
        directory = os.path.dirname(file_path)
        if directory not in directory_embeddings:
            directory_embeddings[directory] = []
        directory_embeddings[directory].append(embedding)

    # Average embeddings per directory
    for directory, embedding_list in directory_embeddings.items():
        directory_embeddings[directory] = torch.stack(embedding_list).mean(dim=0)

    return embeddings, directory_embeddings

def process_user_query_with_flan_t5(query):
    """
    Processes a user query using Flan-T5 to extract structured intent.
    """
    # Prompt Flan-T5 for intent extraction
    prompt = (
        f"Analyze the following user request and provide its intent: \n\n" # TODO: Include predefined intent classes???
        f"Request: {query}\n"
        f"Output format: Intent: <intent_description>"
    )
    inputs = tokenizer_flan_t5(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = flan_t5.generate(inputs.input_ids, max_length=100, num_return_sequences=1)
    generated_text = tokenizer_flan_t5.decode(outputs[0], skip_special_tokens=True)

    # Extract intent from output
    try:
        intent = generated_text.split("Intent: ")[1].strip()
    except IndexError:
        intent = "unknown_intent"

    return intent


def match_query_to_repository(intent, module_embeddings, file_embeddings, tokenizer, model):
    """
    Matches the user's intent to the most relevant module and file.
    """
    # Generate intent embedding
    inputs = tokenizer(intent, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    intent_embedding = outputs.last_hidden_state.mean(dim=1)

    # Match intent to modules
    module_similarities = {
        module: torch.nn.functional.cosine_similarity(intent_embedding, embedding.unsqueeze(0)).item()
        for module, embedding in module_embeddings.items()
    }
    best_module = max(module_similarities, key=module_similarities.get)

    # Match intent to files within the best module
    file_similarities = {
        file: torch.nn.functional.cosine_similarity(intent_embedding, embedding.unsqueeze(0)).item()
        for file, embedding in file_embeddings.items()
        if file.startswith(best_module)
    }
    best_file = max(file_similarities, key=file_similarities.get)

    return best_module, best_file

def decide_which_agents_to_awaken(intent):
    """
    Determines which multi-agent system modules to activate based on the intent.
    """
    # Map intents to agent modules
    intent_to_agent_map = {
        "debug_code": ["debugger_agent"],
        "add_feature": ["feature_generator_agent"],
        "write_tests": ["test_generator_agent"],
        "optimize_performance": ["optimizer_agent"],
        "document_code": ["documentation_agent"],
    }

    return intent_to_agent_map.get(intent, ["default_agent"])

if __name__ == "__main__":
    # Path to the repository
    repo_path = "path_to_your_repository"

    # Parse the repository
    print("Parsing repository...")
    repository_data = parse_repository(repo_path)

    # Generate embeddings
    print("Generating embeddings...")
    file_embeddings, module_embeddings = get_hierarchical_embeddings(repository_data, tokenizer_codebert, codebert)

    # Example User Query
    user_query = "How can I fetch user data from the database?"

    # Process query with Flan-T5
    print("Processing user query...")
    intent = process_user_query_with_flan_t5(user_query)
    print(f"Extracted Intent: {intent}")

    # Match Intent to Repository
    print("Matching query to repository...")
    best_module, best_file = match_query_to_repository(intent, module_embeddings, file_embeddings, tokenizer_codebert, codebert)
    print(f"Best Matching Module: {best_module}")
    print(f"Best Matching File: {best_file}")

    # Decide which agents to awaken
    print("Deciding which agents to awaken...")
    agents_to_awaken = decide_which_agents_to_awaken(intent)
    print(f"Agents to Awaken: {agents_to_awaken}")
