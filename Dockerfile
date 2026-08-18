FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build the index at image build time is possible but slow; safer to run it
# once at container start if the store is missing (see entrypoint below).
ENV DATA_DIR=/app/data/store
ENV PORT=8000

CMD sh -c "\
    if [ ! -f \"$DATA_DIR/semantic.faiss\" ]; then \
        python data/prepare_dataset.py --n-passages 3000; \
    fi && \
    uvicorn app.server:app --host 0.0.0.0 --port ${PORT}"
