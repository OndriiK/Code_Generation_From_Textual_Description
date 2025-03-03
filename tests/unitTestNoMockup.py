import unittest
import sys
import os
from classify_intent.ANALYZE_intent_codebase import LLMHandler

class TestLLMHandlerRealWorkflow(unittest.TestCase):
    
    def setUp(self):
        # Use real models without mocking
        self.handler = LLMHandler(
            codebert_model_name="microsoft/codebert-base",
            classifier_model_path="/mnt/d/wsl_workspace/fine_tuned_bert_v3"#"google/flan-t5-base"
        )
    
    def test_process_query_real(self):
        query = "Please add unit tests for my code."
        intents = self.handler.process_query(query)
        print(f"Detected intents: {intents}")
        # Check if expected intents are extracted
        # self.assertIn("debug_code", intents)
        self.assertIn("write_tests", intents)
    
    def test_generate_embeddings_real(self):
        # Small example repository data
        repo_data = {
            "file1.py": "print('Hello, World!')",
            "dir1/file2.py": "def greet(): return 'Hi!'"
        }
        
        embeddings, dir_embeddings = self.handler.generate_hierarchical_embeddings(repo_data)
        
        # Assert embeddings are created for each file
        self.assertIn("file1.py", embeddings)
        self.assertIn("dir1/file2.py", embeddings)
        
        # Assert directory embeddings are calculated
        self.assertIn("dir1", dir_embeddings)

if __name__ == "__main__":
    unittest.main()
