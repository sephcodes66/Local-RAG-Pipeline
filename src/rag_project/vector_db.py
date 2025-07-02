# src/rag_project/vector_db.py

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from rag_project.config import CHROMA_PERSIST_DIR, COLLECTION_NAME, EMBEDDING_MODEL


def get_chroma_collection():
    """Initializes ChromaDB and returns the collection."""
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    return chroma_client.get_or_create_collection(name=COLLECTION_NAME)


def get_embedding_model():
    """Initializes and returns the embedding model."""
    return SentenceTransformer(EMBEDDING_MODEL, device='cpu')


def embed_and_store_chunks(collection, chunks, embedding_model):
    """
    Generates embeddings for text chunks and stores them in ChromaDB.
    """
    print(f'Generating embeddings for {len(chunks)} chunks and adding to ChromaDB...')
    for chunk in tqdm(chunks, desc='Embedding Chunks'):
        embedding = embedding_model.encode(
            chunk['content'], convert_to_tensor=False
        ).tolist()
        collection.add(
            ids=[chunk['id']],
            embeddings=[embedding],
            documents=[chunk['content']],
            metadatas=[{'source': chunk['source']}],
        )
