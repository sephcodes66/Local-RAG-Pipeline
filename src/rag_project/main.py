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
    The main indexing pipeline.
    It loads PDFs, chunks the text, generates embeddings, and stores them
    in a persistent ChromaDB vector database. This is the "setup" step.
    """
    print('--- Starting Indexing Pipeline ---')

    # Step 1: Load and parse PDFs from the source directory.
    documents = get_pdf_text(PDF_SOURCE_DIR)
    if not documents:
        print('No PDF documents found. Nothing to index. Aborting.')
        return

    # Step 2: Split the extracted text into manageable chunks.
    chunks = get_text_chunks(documents)
    print(f'Created {len(chunks)} text chunks.')

    # Step 3: Initialize our tools - the vector DB and the embedding model.
    print('Initializing ChromaDB and loading embedding model...')
    collection = get_chroma_collection()
    embedding_model = get_embedding_model()

    # Step 4: Generate embeddings for each chunk and store them.
    embed_and_store_chunks(collection, chunks, embedding_model)

    print('--- Indexing Pipeline Complete ---')


def run_query():
    """
    The main query pipeline.
    It takes a user's question, finds relevant document chunks from the
    vector database, and uses an LLM to generate a final answer.
    """
    print('--- Starting Query Session ---')
    print('Initializing ChromaDB and loading models...')

    # Initialize the tools we need for querying.
    collection = get_chroma_collection()
    embedding_model = get_embedding_model()

    # This loop runs the interactive Q&A session.
    while True:
        query_text = input("\nEnter your query (or type 'exit' to quit): ").strip()
        if query_text.lower() == 'exit':
            break
        if not query_text:
            continue

        # Step 1: Find the most relevant document chunks for the query.
        print('Finding relevant context...')
        query_embedding = embedding_model.encode(
            query_text, convert_to_tensor=False
        ).tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5,  # Retrieve the top 5 most relevant chunks
            include=['documents', 'metadatas'],
        )

        # Step 2: Stitch the retrieved chunks together into a single context string.
        context_str = "\n---\n".join(
            f"Source: {meta['source']}\nContent: {doc}"
            for doc, meta in zip(results['documents'][0], results['metadatas'][0])
        )

        # Step 3: Build the prompt for the LLM.
        # This is a critical step. The prompt guides the LLM to answer based
        # *only* on the context we provide. This is the core of RAG.
        prompt = f"""
        You are an expert Q&A assistant. Your task is to answer the user's
        question based *only* on the provided context.

        If the context does not contain the information needed to answer the
        question, you must state: "The provided context does not contain
        enough information to answer this question."

        CONTEXT:
        ---
        {context_str}
        ---

        QUESTION:
        {query_text}

        ANSWER:
        """

        # Step 4: Send the prompt to the LLM and stream the response.
        print('Generating answer...')
        response_stream = ollama.chat(
            model=LLM_MODEL, messages=[{'role': 'user', 'content': prompt}], stream=True
        )

        # Streaming the response gives a better user experience.
        full_response = ''
        for chunk in response_stream:
            content = chunk['message']['content']
            full_response += content
            print(content, end='', flush=True)
        print('\n')


def main():
    # A simple command-line interface to choose between indexing and querying.
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
        # Default behavior if no arguments are given.
        print('Please specify a mode to run: --index or --query')
        parser.print_help()


if __name__ == '__main__':
    main()
