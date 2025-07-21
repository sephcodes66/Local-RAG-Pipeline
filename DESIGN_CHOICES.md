# Design Choices & Philosophy

This document explains some of the "why" behind the tools and structure of this project. I wanted to keep things simple, local, and easy to understand.

## Core Philosophy: Local-First and Simple

The main goal was to build a RAG pipeline that runs entirely on my own machine. I didn't want to rely on any cloud services or APIs that would send my data somewhere else. This was both for privacy reasons and just as a fun challenge.

I also tried to keep the code as straightforward as possible. This isn't meant to be a production-ready, enterprise-grade system. It's a learning project, so readability and simplicity were more important than having all the bells and whistles.

## Technology Choices

-   **Python:** It's the go-to language for data science and AI work, so it was the obvious choice.

-   **Ollama:** This was a key discovery for me. It makes it incredibly easy to download and run powerful open-source LLMs locally. It completely abstracts away the complexity of setting up and managing the models.

-   **ChromaDB:** I needed a vector database to store the document embeddings. I looked at a few options, but ChromaDB stood out because it's so easy to get started with. It can run in-memory or be persisted to disk, which was perfect for this project. I didn't need a full-blown database server.

-   **`all-MiniLM-L6-v2`:** For the embedding model, I needed something that was fast and effective, but small enough to run comfortably on my laptop's CPU. This model is a great balance of performance and size.

-   **`pypdf`:** A simple and effective library for extracting text from PDF files. It just works.

-   **`Makefile`:** I'm a fan of `make` for simple project automation. It's a classic tool, and it's perfect for defining a few simple commands like `setup`, `index`, and `query`. It's simpler than a more complex build system like `tox` or `nox` for a project of this size.

## What I Might Do Differently Next Time

-   **More advanced chunking:** The current text chunking is very basic. It just splits the text by a fixed character count. A better approach would be to use a recursive text splitter that's aware of sentence and paragraph boundaries.
-   **More robust error handling:** The error handling is pretty minimal right now. A real-world application would need to be much more robust.
-   **More tests!** I've set up the testing framework, but I need to add more comprehensive tests.
-   **Configuration management:** The configuration is just a simple `config.py` file. For a larger project, I'd probably use a library like `pydantic` for settings management.
