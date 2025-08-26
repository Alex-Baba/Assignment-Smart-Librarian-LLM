# Use a small multi-arch Python image
FROM python:3.11-slim

# System deps (chromadb / hnswlib sometimes need libgomp)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy only dependency files first (better layer caching)
COPY requirements.txt ./requirements.txt

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . /app

# Environment defaults (safe to override at runtime)
ENV PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501

# Expose Streamlit port
EXPOSE 8501

# Create a non-root user (optional but recommended)
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Persist Chroma DB inside /app/db_store by default
VOLUME ["/app/db_store"]

CMD ["streamlit", "run", "ui/app_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]