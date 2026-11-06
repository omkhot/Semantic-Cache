# OPTIONAL: you can run the project without Docker (see README).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/models

WORKDIR /app

# Install CPU-only PyTorch first (much smaller than the default CUDA build),
# then the remaining requirements and FAISS.
COPY requirements.txt requirements-faiss.txt ./
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt -r requirements-faiss.txt

# Bake the embedding model into the image so the container starts offline-fast.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY semcache ./semcache
COPY evaluation ./evaluation
COPY scripts ./scripts

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "semcache.app:app", "--host", "0.0.0.0", "--port", "8000"]
