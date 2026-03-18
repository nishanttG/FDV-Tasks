### Week-5 — Graph ML and RAG

Summary
 - This repo contains a small project combining a FastAPI backend (`api.py`) and a Streamlit frontend (`app.py`).
 - The backend implements a RAG-style retrieval pipeline (`src/rag_engine.py`) that queries a Neo4j graph (`src/graph_db.py`) and returns relevant constitutional articles.
 - Core source is under `src/`; exploratory analysis is in `notebooks/` and helper scripts in `scripts/`.

Project structure (top-level)
 - `api.py` — FastAPI app exposing `/query` (used by Streamlit UI).
 - `app.py` — Streamlit frontend that posts user prompts to the API.
 - `src/` — library code: `rag_engine.py`, `graph_db.py`, `parser.py`, `ml_pipeline.py`, etc.
 - `data/` — local datasets (ignored from VCS).
 - `models/` — trained models or large artifacts (ignored).
 - `notebooks/` — exploratory analysis notebooks.
 - `scripts/` — ingestion and utility scripts.
 - `requirements.in` / `requirements.txt` — pinned dependencies (generated with Python 3.10).
 - `Dockerfile`, `.dockerignore`, `docker-compose.yml` — dockerization for development and local testing.

Key details
 - Python compatibility: `requirements.txt` was generated using Python 3.10; the Docker image uses `python:3.10-slim` to match that environment.
 - External services: `src/rag_engine.py` expects access to a Neo4j instance (see `src/graph_db.py` for connection configuration). The SentenceTransformer model (`all-MiniLM-L6-v2`) downloads its weights at runtime unless cached locally.

Local development (recommended)
1. Create a Conda environment and install dependencies (recommended):
```powershell
# Create and activate a Conda env with Python 3.10
conda create -n week5 python=3.10 -y
conda activate week5

# Use pip-tools to produce a pinned `requirements.txt` from `requirements.in`
pip install pip-tools
pip-compile requirements.in --output-file=requirements.txt

# Install pinned requirements
pip install -r requirements.txt
```
2. Start a Neo4j instance and set connection env vars in `.env` (example):
```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=yourpassword
```
3. Run the API (development server):
```powershell
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```
4. In another shell, run the Streamlit UI:
```powershell
streamlit run app.py
```

Running tests
 - Install dev deps and run `pytest -q` from the `Week-5` folder.

Dockerization (local dev)
 - A `Dockerfile` and `docker-compose.yml` are provided to make running both the API and the Streamlit UI easy during development.

Build the image (single-image flow)
```powershell
# from Week-5
docker build -t week5-app:py3.10 .
```

Run API-only (maps port 8000)
```powershell
docker run --rm -p 8000:8000 --env-file .env --name week5-api week5-app:py3.10
```

Run API + Streamlit together (recommended for dev)
```powershell
# from Week-5
docker-compose up --build
# API -> http://localhost:8000
# Streamlit -> http://localhost:8501
```


Implementation notes & where to look
 - RAG search: `src/rag_engine.py` — uses `sentence-transformers` for embeddings, `src/graph_db.py` for Neo4j queries, and returns top-k relevant articles.
 - Graph DB: `src/graph_db.py` — reads connection config (check environment variables in `src/config.py`).
 - Ingestion: `scripts/ingest.py` — example data ingestion into Neo4j (check this script before running on production data).
