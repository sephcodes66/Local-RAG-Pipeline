# src/rag_project/main.py

import argparse

import ollama

from rag_project.config import LLM_MODEL, PDF_SOURCE_DIR
from rag_project.data_processing import get_pdf_text, get_text_chunks
from rag_project.vector_db import (
    embed_and_store_chunks,
    get_chroma_collection,
    get_embedding_model,
)


def index_documents():
    """
    The main indexing pipeline. It loads PDFs, chunks the text, generates embeddings,
    and stores them in a persistent ChromaDB vector database.
    """
    print('--- Starting Indexing Pipeline ---')

    # 1. Load and Extract Text from PDFs
    documents = get_pdf_text(PDF_SOURCE_DIR)
    if not documents:
        print('No PDF documents found. Aborting.')
        return

    # 2. Split Text into Chunks
    chunks = get_text_chunks(documents)
    print(f'Created {len(chunks)} text chunks.')

    # 3. Initialize ChromaDB and Embedding Model
    print('Initializing ChromaDB and loading embedding model...')
    collection = get_chroma_collection()
    embedding_model = get_embedding_model()

    # 4. Generate Embeddings and Store in ChromaDB
    embed_and_store_chunks(collection, chunks, embedding_model)

    print('--- Indexing Pipeline Complete ---')


def run_query():
    """
    The main query pipeline. It takes a user's question, finds relevant document
    chunks from the vector database, and uses an LLM to generate an answer.
    """
    print('--- Starting Query Session ---')
    print('Initializing ChromaDB and loading models...')

    # 1. Initialize ChromaDB and Models
    collection = get_chroma_collection()
    embedding_model = get_embedding_model()

    while True:
        # 2. Get User Query
        query_text = input("\nEnter your query (or type 'exit' to quit): ").strip()
        if query_text.lower() == 'exit':
            break
        if not query_text:
            continue

        # 3. Find Relevant Context from Vector DB
        print('Finding relevant context...')
        query_embedding = embedding_model.encode(
            query_text, convert_to_tensor=False
        ).tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5,  # Retrieve top 5 most relevant chunks
            include=['documents', 'metadatas'],
        )

        context_str = '\n---\n'.join(
            f'Source: {meta["source"]}\nContent: {doc}'
            for doc, meta in zip(results['documents'][0], results['metadatas'][0])
        )

        # 4. Generate Answer with LLM
        prompt = f"""
        You are an expert Q&A assistant. Your task is to answer the user's
        question based *only* on the provided context. If the context does not
        contain the information needed to answer the question, you must state:
        "The provided context does not contain enough information to answer this
        question."

        CONTEXT:
        {context_str}

        QUESTION:
        {query_text}

        ANSWER:
        """

        print('Generating answer...')
        response_stream = ollama.chat(
            model=LLM_MODEL, messages=[{'role': 'user', 'content': prompt}], stream=True
        )

        # 5. Stream the response to the console
        full_response = ''
        for chunk in response_stream:
            content = chunk['message']['content']
            full_response += content
            print(content, end='', flush=True)
        print('\n')


def main():
    parser = argparse.ArgumentParser(
        description='A local RAG pipeline for Q&A with your documents.'
    )
    parser.add_argument(
        '--index',
        action='store_true',
        help='Run the indexing pipeline to process and store documents.',
    )
    parser.add_argument(
        '--query', action='store_true', help='Run the interactive query session.'
    )
    args = parser.parse_args()

    if args.index:
        index_documents()
    elif args.query:
        run_query()
    else:
        print('Please specify a mode to run: --index or --query')
        parser.print_help()


if __name__ == '__main__':
    main()
