# tests/test_data_processing.py

import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
from rag_project.data_processing import get_pdf_text, get_text_chunks

class TestDataProcessing(unittest.TestCase):

    @patch('rag_project.data_processing.PdfReader')
    def test_get_pdf_text(self, mock_pdf_reader):
        # Arrange
        mock_pdf_file = MagicMock()
        mock_pdf_file.name = "test.pdf"
        
        mock_pdf_reader.return_value.pages = [MagicMock(extract_text=lambda: "page 1"), MagicMock(extract_text=lambda: "page 2")]
        
        with patch('pathlib.Path.glob') as mock_glob:
            mock_glob.return_value = [mock_pdf_file]
            
            # Act
            documents = get_pdf_text(Path("dummy_path"))
            
            # Assert
            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0][0], "test.pdf")
            self.assertEqual(documents[0][1], "page 1page 2")

    def test_get_text_chunks(self):
        # Arrange
        documents = [("test.pdf", "This is a test document.")]
        
        # Act
        chunks = get_text_chunks(documents)
        
        # Assert
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]['source'], "test.pdf")
        self.assertEqual(chunks[0]['content'], "This is a test document.")
        self.assertEqual(chunks[0]['id'], "test.pdf_chunk_0")

if __name__ == '__main__':
    unittest.main()
