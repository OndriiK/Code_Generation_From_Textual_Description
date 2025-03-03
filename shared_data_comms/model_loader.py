import logging
from pathlib import Path
import torch
from typing import Protocol, Type
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class ModelI(Protocol):
    def __init__(self, *args, **kwargs) -> None:
        ...

    def generate(self, query: str) -> str:
        """Processes an input query and generates a response.

        Args:
            query (str): The input query to process.

        Returns:
            str: The processed output.
        """
        ...

class HuggingFaceModel:
    def __init__(self, model_path: str, device: str = "cpu"):
        """
        Initializes the Hugging Face model and tokenizer.

        Args:
            model_path (str): Path to the fine-tuned model directory.
            device (str): Device to load the model on ('cpu' or 'cuda').
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.device = device
        self.model.to(device)

    def generate(self, task: dict) -> str:
        """
        Processes the input task and generates a response using the model.

        Args:
            task (dict): A task dictionary containing a 'content' field.

        Returns:
            str: The model's response as a string.
        """
        query = task.get("content", "")  # Extract the 'content' field
        if not isinstance(query, str):
            raise ValueError("Task 'content' must be a string.")

        # Tokenize and generate output
        inputs = self.tokenizer(query, return_tensors="pt", truncation=True, padding=True)
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        outputs = self.model(**inputs)
        predicted_class = outputs.logits.argmax(dim=-1).item()
        scores = torch.sigmoid(outputs.logits).squeeze().tolist()
        return f"Predicted scores: {scores}"


def load_model(model_path: Path) -> Type[ModelI]:
    """
    Load a Hugging Face model from the specified path.

    Args:
        model_path (Path): The path to the model directory.

    Returns:
        ModelI: An instance of the loaded model.
    """
    try:
        logging.info(f"Loading model from '{model_path}'")
        return HuggingFaceModel(model_path=str(model_path))
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Failed to load model from '{model_path}'")
