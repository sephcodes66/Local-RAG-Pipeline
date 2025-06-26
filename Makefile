# Makefile

GREEN=\033[0;32m
NC=\033[0m

.PHONY: help setup index query all

help:
	@echo "Commands:"
	@echo "  setup   : Creates a virtual environment and installs dependencies."
	@echo "  index   : Runs the indexing pipeline to process PDFs."
	@echo "  query   : Starts the interactive Q&A session."
	@echo "  all     : Runs the full setup, index, and query flow."

setup:
	@echo "${GREEN}>>> Creating Python virtual environment...${NC}"
	python3 -m venv venv
	@echo "${GREEN}>>> Activating virtual environment and installing dependencies...${NC}"
	@. venv/bin/activate && pip install -q -r requirements.txt
	@echo "${GREEN}>>> Setup complete!${NC}"

index:
	@echo "${GREEN}>>> Running the indexing pipeline...${NC}"
	@. venv/bin/activate && python src/rag_pipeline.py --index

query:
	@echo "${GREEN}>>> Starting the interactive query session...${NC}"
	@. venv/bin/activate && python src/rag_pipeline.py --query

all: setup index query