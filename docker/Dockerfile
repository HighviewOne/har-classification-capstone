# Human Activity Recognition API - Docker Image
# 
# Build:
#   docker build -t har-classifier -f docker/Dockerfile .
#
# Run:
#   docker run -it -p 9696:9696 har-classifier

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY models/ ./models/

# Expose port
EXPOSE 9696

# Set environment variables
ENV PORT=9696

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:9696/health || exit 1

# Run the prediction service
CMD ["python", "src/predict.py"]
