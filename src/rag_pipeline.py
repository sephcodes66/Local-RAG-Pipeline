# src/rag_pipeline.py

import argparse
import ollama
import chromadb
from pathlib import Path
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# --- CONFIGURATION ---
# Paths
PDF_SOURCE_DIR = Path("./docs")
CHROMA_PERSIST_DIR = Path("./chroma_db")
# Models
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
LLM_MODEL = 'phi3:mini' 
# ChromaDB
COLLECTION_NAME = "rag_project_docs"
# Text Splitting
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def get_pdf_text(pdf_docs_path: Path) -> list[tuple[str, str]]:
    """
    Extracts text from all PDF files in the given directory.
    Returns a list of tuples, where each tuple contains the source filename and the extracted text.
    """
    print(f"Loading and extracting text from PDF files in: {pdf_docs_path}")
    documents = []
    for pdf_file in pdf_docs_path.glob("*.pdf"):
        print(f"  - Processing {pdf_file.name}...")
        try:
            reader = PdfReader(pdf_file)
            pdf_text = "".join(page.extract_text() for page in reader.pages)
            documents.append((pdf_file.name, pdf_text))
        except Exception as e:
            print(f"    Error reading {pdf_file.name}: {e}")
    return documents

def get_text_chunks(documents: list[tuple[str, str]]) -> list[dict]:
    """
    Splits the text of each document into smaller chunks.
    Returns a list of dictionaries, each containing the chunked text and its source metadata.
    """
    print("Splitting documents into text chunks...")
    chunked_docs = []
    for doc_name, doc_text in documents:
        text_chunks = [
            doc_text[i:i + CHUNK_SIZE]
            for i in range(0, len(doc_text), CHUNK_SIZE - CHUNK_OVERLAP)
        ]
        for i, chunk in enumerate(text_chunks):
            chunked_docs.append({
                "source": doc_name,
                "content": chunk,
                "id": f"{doc_name}_chunk_{i}"
            })
    return chunked_docs

def index_documents():
    """
    The main indexing pipeline. It loads PDFs, chunks the text, generates embeddings,
    and stores them in a persistent ChromaDB vector database.
    """
    print("--- Starting Indexing Pipeline ---")

    # 1. Load and Extract Text from PDFs
    documents = get_pdf_text(PDF_SOURCE_DIR)
    if not documents:
        print("No PDF documents found. Aborting.")
        return

    # 2. Split Text into Chunks
    chunks = get_text_chunks(documents)
    print(f"Created {len(chunks)} text chunks.")

    # 3. Initialize ChromaDB and Embedding Model
    print("Initializing ChromaDB and loading embedding model...")
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    embedding_model = SentenceTransformer(EMBEDDING_MODEL, device='cpu') # Explicitly use CPU

    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    # 4. Generate Embeddings and Store in ChromaDB
    print(f"Generating embeddings for {len(chunks)} chunks and adding to ChromaDB...")
    # Using a tqdm progress bar for better user experience
    for chunk in tqdm(chunks, desc="Embedding Chunks"):
        embedding = embedding_model.encode(chunk["content"], convert_to_tensor=False).tolist()
        collection.add(
            ids=[chunk["id"]],
            embeddings=[embedding],
            documents=[chunk["content"]],
            metadatas=[{"source": chunk["source"]}]
        )

    print("--- Indexing Pipeline Complete ---")

def run_query():
    """
    The main query pipeline. It takes a user's question, finds relevant document
    chunks from the vector database, and uses an LLM to generate an answer.
    """
    print("--- Starting Query Session ---")
    print("Initializing ChromaDB and loading models...")

    # 1. Initialize ChromaDB and Models
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    collection = chroma_client.get_collection(name=COLLECTION_NAME)
    embedding_model = SentenceTransformer(EMBEDDING_MODEL, device='cpu')

    while True:
        # 2. Get User Query
        query_text = input("\nEnter your query (or type 'exit' to quit): ").strip()
        if query_text.lower() == 'exit':
            break
        if not query_text:
            continue

        # 3. Find Relevant Context from Vector DB
        print("Finding relevant context...")
        query_embedding = embedding_model.encode(query_text, convert_to_tensor=False).tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5, # Retrieve top 5 most relevant chunks
            include=['documents', 'metadatas']
        )

        context_str = "\n---\n".join(
            f"Source: {meta['source']}\nContent: {doc}"
            for doc, meta in zip(results['documents'][0], results['metadatas'][0])
        )
        #print("--- Print Context for debugging ---")
        #print(context_str)
        #   print("--- End of Context ---")
        # 4. Generate Answer with LLM
        prompt = f"""
        You are an expert Q&A assistant. Your task is to answer the user's question based *only* on the provided context.
        If the context does not contain the information needed to answer the question, you must state: "The provided context does not contain enough information to answer this question."

        CONTEXT:
        {context_str}

        QUESTION:
        {query_text}

        ANSWER:
        """

        print("Generating answer...")
        response_stream = ollama.chat(
            model=LLM_MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            stream=True
        )

        # 5. Stream the response to the console
        full_response = ""
        for chunk in response_stream:
            content = chunk['message']['content']
            full_response += content
            print(content, end='', flush=True)
        print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A local RAG pipeline for Q&A with your documents.")
    parser.add_argument("--index", action="store_true", help="Run the indexing pipeline to process and store documents.")
    parser.add_argument("--query", action="store_true", help="Run the interactive query session.")
    args = parser.parse_args()

    if args.index:
        index_documents()
    elif args.query:
        run_query()
    else:
        print("Please specify a mode to run: --index or --query")
        parser.print_help()