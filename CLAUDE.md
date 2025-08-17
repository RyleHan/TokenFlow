# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TokenFlow is an intelligent API gateway that routes user requests using a small LLM for intent recognition, then dynamically retrieves relevant MCP (Multi-Component Program) documentation to provide context for a larger LLM. The goal is to significantly reduce token usage by providing only relevant context instead of all available documentation.

## Project Status

This is currently a planning/documentation phase project. The only existing file is `todolist.md` which contains a detailed Chinese-language project specification and development roadmap.

## Architecture Components

Based on the project specification, the system will consist of:

1. **FastAPI Gateway** (`main.py`) - Main API endpoint `/route` that accepts user prompts
2. **Document Retriever** (`retriever.py`) - Uses Faiss/ChromaDB for semantic search of MCP documentation
3. **Small LLM Intent Classifier** - Fine-tuned Qwen-0.5B-Chat model for intent recognition
4. **MCP Documentation Store** - Markdown files describing various API services
5. **Large LLM Interface** - Integration point for final decision making

## Technology Stack

- **Language**: Python 3.9+
- **Web Framework**: FastAPI
- **Small LLM**: Qwen-0.5B-Chat
- **LLM Framework**: Hugging Face transformers
- **Fine-tuning**: LoRA (PEFT)
- **Vector Search**: Faiss or ChromaDB
- **Containerization**: Docker
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)

## Development Phases

The project is organized into 5 development phases as outlined in `todolist.md`:

1. **Environment Setup** - FastAPI skeleton, Docker, basic routing
2. **Document Store & Retrieval** - Mock MCP docs, vector search implementation
3. **Small LLM Integration** - Fine-tuning dataset, model training, intent prediction
4. **Large LLM Integration** - Context assembly, final response generation
5. **Documentation & Optimization** - README, performance benchmarking

## Key Design Principles

- **Token Efficiency**: Primary goal is reducing tokens sent to large LLMs
- **Intent-Driven Routing**: Small LLM identifies user intent to filter relevant docs
- **Modular Architecture**: Separate concerns for retrieval, intent classification, and response generation
- **Containerized Deployment**: Docker-based deployment for consistency

## Development Notes

- The project emphasizes Chinese language support based on the documentation
- Uses 4-bit quantization for memory efficiency with small LLM
- Includes benchmarking scripts to demonstrate token savings
- Mock/simulation approach for testing without actual large LLM API calls