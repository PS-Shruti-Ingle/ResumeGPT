import sys
import unittest
from pathlib import Path
from langchain_core.documents import Document

# Add project root to python path to resolve local modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import importlib
from config import Config

# Dynamically import modules with leading numbers in their filenames
chunking_mod = importlib.import_module("pipeline.2_chunking")
chunk_documents = chunking_mod.chunk_documents

prompt_mod = importlib.import_module("pipeline.6_prompt")
get_rag_prompt = prompt_mod.get_rag_prompt

class TestRAGPipeline(unittest.TestCase):
    """Unit tests for validating the modules of the Resume Reader RAG chatbot."""

    def test_config_paths(self):
        """Verifies configuration absolute path conversions function properly."""
        rel_path = "test_dir"
        abs_path = Config.get_absolute_path(rel_path)
        self.assertTrue(abs_path.is_absolute())
        self.assertEqual(abs_path.name, "test_dir")

    def test_chunking_logic(self):
        """Validates that character text splitter segments docs as expected."""
        mock_docs = [
            Document(
                page_content=(
                    "Senior Software Engineer with 8 years of experience. "
                    "Specialized in Python, Machine Learning, and Cloud Systems. "
                    "Led high-performing teams to deploy large language models."
                ),
                metadata={"source": "test_resume.pdf", "page": 0}
            )
        ]
        # Run chunking with small size to guarantee split
        chunks = chunk_documents(mock_docs, chunk_size=50, chunk_overlap=10)
        
        self.assertGreater(len(chunks), 1)
        # Check metadata conservation
        for chunk in chunks:
            self.assertEqual(chunk.metadata["source"], "test_resume.pdf")
            self.assertEqual(chunk.metadata["page"], 0)
            self.assertIn("start_index", chunk.metadata)

    def test_prompt_template(self):
        """Validates that the custom prompt includes required input placeholders."""
        prompt = get_rag_prompt()
        self.assertIsNotNone(prompt)
        
        # Ensure our specific keys are expected input parameters
        input_vars = prompt.input_variables
        self.assertIn("context", input_vars)
        self.assertIn("question", input_vars)
        self.assertIn("chat_history", input_vars)
        
        # Verify strict instructions exist in the system prompt
        system_msg = prompt.messages[0].prompt.template
        self.assertIn("I couldn't find this information in the uploaded resume.", system_msg)
        self.assertIn("ONLY the provided resume context", system_msg)

if __name__ == "__main__":
    print("Running RAG Pipeline Verification Tests...")
    unittest.main()
