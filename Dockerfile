FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV DATA_DIR=/app/data/store
ENV PORT=8000

EXPOSE 8000

CMD sh -c "\
    if [ ! -f \"$DATA_DIR/semantic.faiss\" ]; then \
        python data/prepare_dataset.py --n-passages 3000; \
    fi && \
    uvicorn app.server:app --host 0.0.0.0 --port ${PORT:-8000}"
