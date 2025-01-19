import unittest
from unittest.mock import MagicMock, patch
import torch
from ANALYZE_intent_codebase import LLMHandler

class TestLLMHandler(unittest.TestCase):

    def setUp(self):
        self.handler = LLMHandler()

    @patch("ANALYZE_intent_codebase.AutoTokenizer.from_pretrained")
    @patch("ANALYZE_intent_codebase.AutoModel.from_pretrained")
    @patch("ANALYZE_intent_codebase.AutoModelForSeq2SeqLM.from_pretrained")
    def test_initialization(self, mock_flan_t5, mock_codebert, mock_tokenizer):
        mock_tokenizer.return_value = MagicMock()
        mock_codebert.return_value = MagicMock()
        mock_flan_t5.return_value = MagicMock()

        handler = LLMHandler()

        self.assertIsNotNone(handler.tokenizer_codebert)
        self.assertIsNotNone(handler.codebert)
        self.assertIsNotNone(handler.tokenizer_flan_t5)
        self.assertIsNotNone(handler.flan_t5)

    @patch("ANALYZE_intent_codebase.AutoModel.from_pretrained")
    @patch("ANALYZE_intent_codebase.AutoTokenizer.from_pretrained")
    def test_generate_hierarchical_embeddings(self, mock_tokenizer, mock_model):
        mock_model.return_value = MagicMock()
        mock_tokenizer.return_value = MagicMock()

        repo_data = {
            "file1.py": "print(\"Hello, World!\")",
            "dir1/file2.py": "def greet(): return \"Hi!\""
        }

        mock_output = MagicMock()
        mock_output.last_hidden_state.mean.return_value = torch.tensor([1.0, 2.0, 3.0])
        mock_model.return_value.__call__.return_value = mock_output

        embeddings, dir_embeddings = self.handler.generate_hierarchical_embeddings(repo_data)

        self.assertIn("file1.py", embeddings)
        self.assertIn("dir1", dir_embeddings)
        self.assertTrue(torch.equal(dir_embeddings["dir1"], torch.tensor([1.0, 2.0, 3.0])))

    @patch("ANALYZE_intent_codebase.AutoModelForSeq2SeqLM.from_pretrained")
    @patch("ANALYZE_intent_codebase.AutoTokenizer.from_pretrained")
    def test_process_query(self, mock_tokenizer, mock_flan_t5):
        mock_tokenizer.return_value = MagicMock()
        mock_flan_t5.return_value = MagicMock()

        mock_flan_t5.return_value.generate.return_value = [torch.tensor([0, 1, 2, 3])]
        mock_tokenizer.return_value.decode.return_value = "Intent: debug_code; Intent: optimize_performance"

        query = "Optimize the code and debug issues"
        intents = self.handler.process_query(query)

        self.assertIn("debug_code", intents)
        self.assertIn("optimize_performance", intents)

if __name__ == "__main__":
    unittest.main()
