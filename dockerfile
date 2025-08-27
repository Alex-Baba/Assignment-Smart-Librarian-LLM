FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CHROMA_PERSIST_DIR=/app/db_store

WORKDIR /app

# System deps (audio + build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ffmpeg libsndfile1 git && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps first for better caching
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# App code
COPY . .

# Ensure writable cache & Chroma store
RUN mkdir -p /app/.cache /app/db_store && chmod -R 777 /app/.cache /app/db_store

EXPOSE 8501

# Launch Streamlit
CMD ["streamlit","run","ui/app_streamlit.py","--server.port=8501","--server.address=0.0.0.0"]