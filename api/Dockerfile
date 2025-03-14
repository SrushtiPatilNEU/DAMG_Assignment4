FROM --platform=linux/amd64 python:3.9-slim-buster

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    make \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app
COPY . .

# Create necessary directories
RUN mkdir -p /app/uploads /app/markdowns

# Expose port 8080 for Cloud Run (Cloud Run expects your app to listen on 8080)
EXPOSE 8080

# Start the Uvicorn server on port 8080 (Cloud Run default)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
