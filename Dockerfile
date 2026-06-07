# Start from an official Python image that includes apt-get
FROM python:3.11-slim

# Install FFmpeg and libmediainfo in one clean step
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libmediainfo0v5 \
    libmediainfo-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Start the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libmediainfo0v5 \
    libmediainfo-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
