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
