FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    cmake \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY config/ ./config/
COPY Makefile .
COPY setup.py .

# Build C++ module
RUN make build-cpp

# Build Python package
RUN make build-python

# Create necessary directories
RUN mkdir -p logs results

# Set environment variables
ENV PYTHONPATH=/app/src/python
ENV OMP_NUM_THREADS=4

# Default command
CMD ["python", "src/python/main_engine.py"]
