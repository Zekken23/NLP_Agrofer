FROM python:3.11-slim

WORKDIR /app

# Dependency sistem minimal yang sering dibutuhkan scikit-learn/numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download data NLTK yang dipakai (stopwords, punkt) saat build,
# supaya container tidak perlu akses internet lagi saat runtime start
RUN python -m nltk.downloader stopwords punkt punkt_tab -d /usr/share/nltk_data
ENV NLTK_DATA=/usr/share/nltk_data

COPY . .

# HF Spaces default expose port 7860
ENV PORT=7860
EXPOSE 7860

CMD ["python", "app.py"]
