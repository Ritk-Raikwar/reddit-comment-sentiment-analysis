FROM python:3.11-slim

WORKDIR /app

# 1. Install System Dependencies (libgomp1 for LightGBM, build-essential for Wordcloud)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 2. Copy and Install Python Dependencies
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# 3. Download EXACT NLTK Data required by your preprocessing script
RUN python -m nltk.downloader wordnet stopwords omw-1.4

# 4. Copy Application Code and Models
COPY app.py .
COPY src/ ./src/
COPY models/ ./models/

EXPOSE 5000

# 5. Run with Gunicorn (Production WSGI Server)
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "app:app"]
