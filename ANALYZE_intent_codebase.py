import logging
import os
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, pipeline

# Configure logging at the module level
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)  # Create a logger for this module

class LLMHandler:
    def __init__(self, codebert_model_name="microsoft/codebert-base", classifier_model_name="facebook/bart-large-mnli"):
        """
        Initializes the LLM handler with CodeBERT and a classification model.
        """
        logger.info("Initializing LLMHandler with models.")
        
        # Initialize CodeBERT
        logger.info(f"Loading CodeBERT model: {codebert_model_name}")
        self.tokenizer_codebert = AutoTokenizer.from_pretrained(codebert_model_name)
        self.codebert = AutoModel.from_pretrained(codebert_model_name)

        # Initialize BERT-based Classifier
        logger.info(f"Loading classification model: {classifier_model_name}")
        # Initialize zero-shot classification pipeline using facebook/bart-large-mnli
        self.classifier = pipeline("text-classification", model="Falconsai/intent_classification")

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
        Classifies the user query into one or more predefined intents using zero-shot classification.
        """
        logger.info(f"Processing query: {query}")
        
        predefined_intents = [
            "debug_code", 
            "add_feature", 
            "write_tests", 
            "optimize_performance", 
            "document_code"
        ]

        # Use the zero-shot classification pipeline to classify the query
        result = self.classifier(query, candidate_labels=predefined_intents)

        # Log the raw results
        logger.info(f"Classification result: {result}")

        # Extract intents with the highest scores
        detected_intents = [
            intent for intent, score in zip(result["labels"], result["scores"])
            if score > 0.1  # Threshold to filter out low-confidence intents
        ]

        if not detected_intents:
            logger.warning("No predefined intents detected with sufficient confidence.")
            detected_intents = ["unknown_intent"]

        logger.info(f"Extracted intents: {detected_intents}")
        return detected_intents

