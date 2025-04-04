# Use official Python base image
FROM python:3.10-slim

# Install system dependencies for opencv & YOLO
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy app files
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Expose port
EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
