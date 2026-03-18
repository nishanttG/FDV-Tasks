FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install build deps (adjust if you need system libs for specific packages)
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python deps (copy requirements first for caching)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Create non-root user
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Expose API (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Default: run the API with Uvicorn. For Streamlit or other commands, override in docker-compose or docker run.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
