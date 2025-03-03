import logging
import os
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

# Configure logging at the module level
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)  # Create a logger for this module

class LLMHandler:
    def __init__(self, codebert_model_name="microsoft/codebert-base", classifier_model_path="/mnt/d/wsl_workspace/fine_tuned_bert_v4"):
        """
        Initializes the LLM handler with CodeBERT and a fine-tuned classification model.
        """
        logger.info("Initializing LLMHandler with models.")
        
        # Initialize CodeBERT
        logger.info(f"Loading CodeBERT model: {codebert_model_name}")
        self.tokenizer_codebert = AutoTokenizer.from_pretrained(codebert_model_name)
        self.codebert = AutoModel.from_pretrained(codebert_model_name)

        # Initialize Fine-Tuned Classification Model
        logger.info(f"Loading fine-tuned classification model from: {classifier_model_path}")
        self.tokenizer_classifier = AutoTokenizer.from_pretrained(classifier_model_path)
        self.classifier = AutoModelForSequenceClassification.from_pretrained(classifier_model_path)

    def generate_hierarchical_embeddings(self, repo_data):
        """
        Generates embeddings for each file and aggregates them by directory.
        """
        logger.info("Generating hierarchical embeddings for repository data.")
        embeddings = {}
        for file_path, code in repo_data.items():
            logger.debug(f"Processing file: {file_path}")
            inputs = self.tokenizer_codebert(code, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = self.codebert(**inputs)
            file_embedding = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
            embeddings[file_path] = file_embedding
            logger.debug(f"Generated embedding for {file_path}")

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
            logger.debug(f"Aggregated embeddings for directory: {directory}")

        logger.info("Completed hierarchical embedding generation.")
        return embeddings, directory_embeddings

    def process_query(self, query):
        """
        Classifies the user query into one or more predefined intents using the fine-tuned model.
        """
        logger.info(f"Processing query: {query}")
        
        predefined_intents = [
            "debug_code", 
            "add_feature", 
            "write_tests", 
            "optimize_performance", 
            "document_code"
        ]

        # Create a custom prompt for the model
        prompt = (
            f"Classify the following query into predefined intents:\n"
            f"Predefined intents: {', '.join(predefined_intents)}\n"
            f"Query: {query}\n"
            f"Output format: Intent: <intent_1>; Intent: <intent_2>; ..."
        )

        # Tokenize the prompt for the fine-tuned model
        inputs = self.tokenizer_classifier(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.classifier(**inputs)

        logger.info(f"Model outputs: {outputs}")
        # Apply sigmoid activation for multi-label classification
        scores = torch.sigmoid(outputs.logits).squeeze().tolist()
        logger.info(f"Intent scores: {scores}")
        # Map scores to intents
        detected_intents = [
            intent for intent, score in zip(predefined_intents, scores) if score > 0.5  # Threshold for intent detection
        ]

        if not detected_intents:
            logger.warning("No predefined intents detected with sufficient confidence.")
            detected_intents = ["unknown_intent"]

        logger.info(f"Extracted intents: {detected_intents}")
        return detected_intents
