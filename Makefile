# Makefile

GREEN=\033[0;32m
NC=\033[0m

.PHONY: help setup index query lint test all

help:
	@echo "Commands:"
	@echo "  setup   : Creates a virtual environment and installs dependencies."
	@echo "  index   : Runs the indexing pipeline to process PDFs."
	@echo "  query   : Starts the interactive Q&A session."
	@echo "  lint    : Runs the linter to check code quality."
	@echo "  test    : Runs the unit tests."
	@echo "  all     : Runs the full setup, index, and query flow."

setup:
	@echo "${GREEN}>>> Creating Python virtual environment...${NC}"
	python3 -m venv venv
	@echo "${GREEN}>>> Activating virtual environment and installing dependencies...${NC}"
	@. venv/bin/activate && pip install -q -r requirements.txt
	@echo "${GREEN}>>> Setup complete!${NC}"

index:
	@echo "${GREEN}>>> Running the indexing pipeline...${NC}"
	@. venv/bin/activate && python -m rag_project.main --index

query:
	@echo "${GREEN}>>> Starting the interactive query session...${NC}"
	@. venv/bin/activate && python -m rag_project.main --query

lint:
	@echo "${GREEN}>>> Running linter...${NC}"
	@. venv/bin/activate && ruff check .

test:
	@echo "${GREEN}>>> Running unit tests...${NC}"
	@. venv/bin/activate && pytest

all: setup index query
