# Dockerfile
FROM python:3.10-slim

# avoid Python writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1  
ENV PYTHONUNBUFFERED=1

# install system deps (PyMuPDF, Pillow, pytesseract need tesseract and libs)
RUN apt-get update && apt-get install -y \
    build-essential \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy your code
COPY . .

# entrypoint runs your pipeline, which expects /app/input and /app/output
ENTRYPOINT ["python", "run_pipeline.py"]
