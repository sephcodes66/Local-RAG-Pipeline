
> **Note:** This project was a fun experiment built by a human developer with the help of an AI assistant. The goal was to explore local-first RAG pipelines.

# My Local RAG Project

This is a little project I put together to play around with Retrieval-Augmented Generation (RAG). It's a simple pipeline that lets you ask questions about a collection of PDF documents, and it all runs on your local machine. No data ever leaves your computer.

I wanted to build something that was easy to set up and use, so I tried to keep the dependencies to a minimum. See the `DESIGN_CHOICES.md` for more on why I picked the tools I did.

## What's in the Box?

-   **Language:** Python 3
-   **LLM Engine:** Ollama (because it's awesome for running models locally)
-   **Language Model:** `phi3:mini` (a great little model)
-   **Vector Database:** ChromaDB (super easy to get started with)
-   **Embedding Model:** `all-MiniLM-L6-v2` (fast and effective)
-   **Core Libraries:** `ollama`, `chromadb`, `pypdf`
-   **Automation:** A simple `Makefile` to make life easier
-   **Code Quality:** `ruff` for linting and `pytest` for testing (though I need to add more tests!)

## How to Get it Running

### Stuff you need first

-   **Ollama:** You'll need to have [Ollama](https://ollama.com/) installed.
    1.  Once it's installed, pull the model: `ollama pull phi3:mini`.
    2.  Make sure the Ollama application is running before you start.

### Installation

1.  **Clone the repo:**
    ```bash
    git clone https://github.com/sephcodes66/Smart_Dita_Factory
    cd RAG_Project_Local
    ```

2.  **Set up your environment and install the goods:**
    ```bash
    make setup
    ```

## How to Use It

1.  **Drop your PDFs** into the `docs` folder.
2.  **Index your documents:** This will process the PDFs and load them into the vector database.
    ```bash
    make index
    ```
3.  **Ask away!** This starts the interactive Q&A session.
    ```bash
    make query
    ```
    You can type `exit` to end the session.

### Other useful commands

-   **Linting:** `make lint`
-   **Testing:** `make test`

## A Few Notes...

-   If you get a "connection refused" error, it probably means the Ollama application isn't running.
-   If you change the documents in the `docs` folder, you'll need to re-index them. The easiest way is to just delete the `chroma_db` directory and run `make index` again.
    ```bash
    rm -rf chroma_db
    make index
    ```
