FROM python:3.11-slim

WORKDIR /app

# Install CPU-only torch first for fast and clean builds
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY . .

ENV DATA_DIR=/app/data/store

CMD sh -c "\
    if [ ! -f \"$DATA_DIR/metadata_aware.faiss\" ]; then \
        python data/prepare_dataset.py --n-passages 3000; \
    fi && \
    uvicorn app.server:app --host 0.0.0.0 --port ${PORT:-8080}"
