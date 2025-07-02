# src/rag_project/config.py

from pathlib import Path

# --- CONFIGURATION ---
# Paths
PDF_SOURCE_DIR = Path('./docs')
CHROMA_PERSIST_DIR = Path('./chroma_db')
# Models
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
LLM_MODEL = 'phi3:mini'
# ChromaDB
COLLECTION_NAME = 'rag_project_docs'
# Text Splitting
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
