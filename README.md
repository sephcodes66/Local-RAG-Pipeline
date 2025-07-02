# Local RAG Pipeline with Phi-3 and ChromaDB

This project provides a complete, local-first Retrieval-Augmented Generation (RAG) pipeline for question-answering over a collection of PDF documents. The entire workflow runs on your local machine, ensuring data privacy and eliminating the need for cloud services or API keys.

The pipeline demonstrates a modern data engineering approach to LLM application development, including data extraction, chunking, embedding, and vector storage.

## Core Technologies

| Category          | Technology         | Purpose                               |
| ----------------- | ------------------ | ------------------------------------- |
| Language          | Python 3           | Core programming language             |
| LLM Engine        | Ollama             | Serving the local LLM                 |
| Language Model    | `phi3:mini`        | Answer generation                     |
| Vector Database   | ChromaDB           | Storing and querying document chunks  |
| Embedding Model   | `all-MiniLM-L6-v2` | Generating vector embeddings          |
| Core Libraries    | `ollama`, `chromadb`, `pypdf` | Interacting with the tech stack       |
| Automation        | Makefile           | Streamlining development tasks        |
| Code Quality      | `ruff`, `pytest`   | Linting, formatting, and testing      |

## Project Structure

```
.
├── Makefile
├── README.md
├── docs
│   ├── 1706.03762v7.pdf
│   ├── 2019-duckdbdemo.pdf
│   └── 2307.06435v10.pdf
├── pyproject.toml
├── requirements.txt
├── ruff.toml
├── src
│   ├── __init__.py
│   └── rag_project
│       ├── __init__.py
│       ├── config.py
│       ├── data_processing.py
│       ├── main.py
│       └── vector_db.py
└── tests
    ├── __init__.py
    └── test_data_processing.py
```

## Getting Started

### Prerequisites

- **Ollama:** This project requires Ollama to run the local LLM.
  1. Download and install [Ollama](https://ollama.com/).
  2. Launch the Ollama application.
  3. Pull the `phi3:mini` model:
     ```bash
     ollama pull phi3:mini
     ```
  > **Note:** The Ollama application must be running in the background for the project to work.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/sephcodes66/Smart_Dita_Factory
    cd RAG_Project_Local
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    make setup
    ```

## Usage

The `Makefile` provides a set of commands to streamline the development workflow.

-   **Index Documents:**
    Process all PDFs in the `./docs` folder and store them in the local vector database. Run this command once, or whenever you add, remove, or change the source documents.
    ```bash
    make index
    ```

-   **Ask Questions:**
    Start the interactive query session to ask questions about your indexed documents.
    ```bash
    make query
    ```
    To exit the session, type `exit` and press Enter.

-   **Run Linter:**
    Check the code quality and formatting using `ruff`.
    ```bash
    make lint
    ```

-   **Run Tests:**
    Run the unit tests using `pytest`.
    ```bash
    make test
    ```

## Troubleshooting

-   **Error: Connection refused:**
    This error indicates that the Ollama application is not running. Launch the Ollama application and ensure it is running in the background.

-   **Stale or incorrect sources in results:**
    The ChromaDB database is persistent. If you have deleted or modified your source documents, you may need to delete the `chroma_db` directory and re-index your documents.
    ```bash
    rm -rf chroma_db
    make index
    ```
    > **Warning:** This will permanently delete your existing index.