# src/rag_project/config.py

from pathlib import Path

# --- CONFIGURATION ---
# We're using Path for OS-agnostic paths. It's just better.
# All paths are relative to the project root.

# Source directory for the PDF documents we want to process.
PDF_SOURCE_DIR = Path('./docs')
# Where we'll store the ChromaDB vector database.
CHROMA_PERSIST_DIR = Path('./chroma_db')


# --- Models ---
# Using a smaller, faster model for local embedding generation.
# 'all-MiniLM-L6-v2' is a good balance of speed and quality.
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
# The LLM for the generation step. 'phi3:mini' is a powerful small model.
# Make sure you have it installed with `ollama pull phi3:mini`
LLM_MODEL = 'phi3:mini'


# --- ChromaDB Settings ---
# Just a name for our collection inside the database.
COLLECTION_NAME = 'rag_project_docs'


# --- Text Splitting Parameters ---
# These values seem to work well, but they're worth tuning if results are poor.
# A larger chunk size can give more context, but might dilute the key info.
CHUNK_SIZE = 1000
# A bit of overlap helps to not split sentences right at the edge of a chunk.
CHUNK_OVERLAP = 200
