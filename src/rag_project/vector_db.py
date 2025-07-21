# src/rag_project/vector_db.py

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from rag_project.config import CHROMA_PERSIST_DIR, COLLECTION_NAME, EMBEDDING_MODEL


def get_chroma_collection():
    """
    Initializes ChromaDB client and returns the specified collection.
    This will create the collection if it doesn't exist yet.
    """
    # Using a persistent client means the data will be saved to disk.
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    return chroma_client.get_or_create_collection(name=COLLECTION_NAME)


def get_embedding_model():
    """
    Initializes and returns the sentence-transformer embedding model.
    """
    # Forcing CPU for now. If you have a good GPU, you might want to change this.
    # e.g., device='cuda' if you have a compatible NVIDIA GPU.
    return SentenceTransformer(EMBEDDING_MODEL, device='cpu')


def embed_and_store_chunks(collection, chunks, embedding_model):
    """
    Generates embeddings for text chunks and stores them in ChromaDB.

    This function iterates through all the text chunks, encodes them into
    vector embeddings, and then adds them to the ChromaDB collection.
    The `tqdm` library gives us a nice progress bar.
    """
    print(f'Generating embeddings for {len(chunks)} chunks and adding to ChromaDB...')

    # Looping through chunks one by one. For a very large dataset, we might
    # want to batch this process for better efficiency.
    for chunk in tqdm(chunks, desc='Embedding Chunks'):
        # The actual embedding generation.
        embedding = embedding_model.encode(
            chunk['content'], convert_to_tensor=False
        ).tolist()

        # Adding the data to the collection. We use the chunk ID we created
        # earlier to uniquely identify each entry.
        collection.add(
            ids=[chunk['id']],
            embeddings=[embedding],
            documents=[chunk['content']],
            metadatas=[{'source': chunk['source']}],
        )
