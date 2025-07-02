# src/rag_project/data_processing.py

from pathlib import Path

from pypdf import PdfReader

from rag_project.config import CHUNK_OVERLAP, CHUNK_SIZE


def get_pdf_text(pdf_docs_path: Path) -> list[tuple[str, str]]:
    """
    Extracts text from all PDF files in the given directory.
    Returns a list of tuples, where each tuple contains the source
    filename and the extracted text.
    """
    print(f'Loading and extracting text from PDF files in: {pdf_docs_path}')
    documents = []
    for pdf_file in pdf_docs_path.glob('*.pdf'):
        print(f'  - Processing {pdf_file.name}...')
        try:
            reader = PdfReader(pdf_file)
            pdf_text = ''.join(page.extract_text() for page in reader.pages)
            documents.append((pdf_file.name, pdf_text))
        except Exception as e:
            print(f'    Error reading {pdf_file.name}: {e}')
    return documents


def get_text_chunks(documents: list[tuple[str, str]]) -> list[dict]:
    """
    Splits the text of each document into smaller chunks.
    Returns a list of dictionaries, each containing the chunked text
    and its source metadata.
    """
    print('Splitting documents into text chunks...')
    chunked_docs = []
    for doc_name, doc_text in documents:
        text_chunks = [
            doc_text[i : i + CHUNK_SIZE]
            for i in range(0, len(doc_text), CHUNK_SIZE - CHUNK_OVERLAP)
        ]
        for i, chunk in enumerate(text_chunks):
            chunked_docs.append(
                {'source': doc_name, 'content': chunk, 'id': f'{doc_name}_chunk_{i}'}
            )
    return chunked_docs
