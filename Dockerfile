FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TFHUB_CACHE_DIR=/tmp/tfhub_cache \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

WORKDIR /app

# Dependencias del sistema para librosa/soundfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

# Instalar requirements desde src
COPY src/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# Copiar todo el código
COPY . /app

EXPOSE 10000
CMD ["gunicorn","--chdir","src","prototype:app","--bind","0.0.0.0:10000","--timeout","600","--workers","1","--threads","1","--capture-output","--log-level","info","--max-requests","200","--max-requests-jitter","50"]

