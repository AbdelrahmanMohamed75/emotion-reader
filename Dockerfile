FROM python:3.11-slim

WORKDIR /app
ENV PRELOAD_MODEL=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY . .

# Download the model once at build time, so the site does not download it on every wake-up
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('abdelrahmanemam10/emotion-reader-model', local_dir='Artifacts')"

CMD gunicorn main:app --bind 0.0.0.0:${PORT:-10000} --workers 1 --threads 2 --timeout 180
