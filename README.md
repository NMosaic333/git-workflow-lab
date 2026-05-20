# Research Paper RAG Assistant

A polished, production-style AI-powered web application focused on helping students and researchers interact with academic papers intelligently.

This project implements a Retrieval-Augmented Generation (RAG) system with a FastAPI backend and a Streamlit frontend, allowing users to upload research papers and ask contextual questions using hybrid search capabilities (Vector Search + Keyword Search).

## Core Features

*   **PDF Upload & Parsing**: Robust extraction of text and metadata using PyMuPDF.
*   **Intelligent Text Chunking**: Semantic-aware chunking preserving continuity.
*   **Hybrid Retrieval**: Combines semantic search (ChromaDB + SentenceTransformers) and keyword search (BM25) via Reciprocal Rank Fusion.
*   **Citation-Aware Responses**: Every answer highlights the source chunk and specific page number of the referenced paper.
*   **Specialized Research Modes**: Switch between General, Beginner, Methodology, Limitations, and Comparison modes to tailor the AI's focus.
*   **Evaluation Dashboard**: Built-in tracking of system metrics including token usage and latency metrics stored via SQLite.

## System Architecture

The application is built using a decoupled architecture:

*   **Backend Framework**: FastAPI
*   **Frontend Framework**: Streamlit
*   **LLM Orchestration**: LangChain + Google Gemini API (`gemini-1.5-pro-latest`)
*   **Embeddings**: HuggingFace `sentence-transformers` (`all-MiniLM-L6-v2`)
*   **Vector Database**: ChromaDB (Local persistent)
*   **Keyword Retrieval**: rank_bm25

## Setup and Installation

### Prerequisites

*   Docker and Docker Compose installed.
*   A Google Gemini API key.

### Instructions

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/research-assistant.git
    cd research-assistant
    ```

2.  **Environment Variables:**
    Copy the `.env.example` file to `.env` and configure your API key:
    ```bash
    cp .env.example .env
    ```
    Add your `GOOGLE_API_KEY` to the `.env` file.

3.  **Run with Docker Compose:**
    ```bash
    docker-compose up --build
    ```

4.  **Access the Application:**
    *   **Streamlit Frontend**: http://localhost:8501
    *   **FastAPI Backend (Swagger UI)**: http://localhost:8000/docs

## Directory Structure

*   `backend/`: Contains the FastAPI application, RAG pipeline, and retrieval logic.
*   `frontend/`: Contains the Streamlit application and UI components.
*   `data/`: Persistent storage directory (mounted via Docker) for uploaded papers, the ChromaDB vector store, BM25 corpus, and SQLite evaluation metrics.

## Engineering Highlights

*   **Modular Design**: Clean separation of concerns between API routing, RAG orchestration, and retrieval mechanisms.
*   **Hybrid Search**: Implementing Reciprocal Rank Fusion significantly reduces hallucinations by ensuring exact keyword matching alongside semantic understanding.
*   **Dockerized Deployment**: Fully containerized environment ensuring consistent setup and data persistence.
