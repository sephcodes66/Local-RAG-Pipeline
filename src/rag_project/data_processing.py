# src/rag_project/data_processing.py

from pathlib import Path

from pypdf import PdfReader

# Using single quotes here for variety
from rag_project.config import CHUNK_OVERLAP, CHUNK_SIZE


def get_pdf_text(pdf_docs_path: Path) -> list[tuple[str, str]]:
    """
    Extracts text from all PDF files in the given directory.

    This is a straightforward PDF text extraction. It iterates through all .pdf
    files in the specified directory, reads them, and pulls out the text content
    page by page.

    Returns a list of tuples, where each tuple contains the source
    filename and the extracted text.
    """
    print(f"Loading and extracting text from PDF files in: {pdf_docs_path}")
    documents = []
    for pdf_file in pdf_docs_path.glob('*.pdf'):
        print(f'  - Processing {pdf_file.name}...')
        try:
            reader = PdfReader(pdf_file)
            # Simple and effective, just join all page texts.
            # For more complex PDFs, we might need a more advanced parser.
            pdf_text = "".join(page.extract_text() for page in reader.pages)
            documents.append((pdf_file.name, pdf_text))
        except Exception as e:
            # Basic error handling, just in case a PDF is corrupted or unreadable.
            print(f"    Error reading {pdf_file.name}: {e}")
    return documents


def get_text_chunks(documents: list[tuple[str, str]]) -> list[dict]:
    """
    Splits the text of each document into smaller chunks.

    The logic here is a simple sliding window. We step through the text and
    create chunks of a fixed size. The overlap helps ensure that we don't
    cut off important context between chunks.

    Returns a list of dictionaries, each containing the chunked text
    and its source metadata.
    """
    print('Splitting documents into text chunks...')
    chunked_docs = []
    for doc_name, doc_text in documents:
        # This is a pretty naive way to chunk, but it's fast and simple.
        # A more advanced method would use a recursive text splitter or
        # something that's aware of sentence boundaries.
        # Let's write it out in a more "classic" iterative way.
        text_chunks = []
        for i in range(0, len(doc_text), CHUNK_SIZE - CHUNK_OVERLAP):
            text_chunks.append(doc_text[i : i + CHUNK_SIZE])

        for i, chunk in enumerate(text_chunks):
            # We create a unique ID for each chunk to use in the vector DB.
            chunked_docs.append(
                {'source': doc_name, 'content': chunk, 'id': f'{doc_name}_chunk_{i}'}
            )
    return chunked_docs
