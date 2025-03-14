# Use a lightweight Python base image compatible with Cloud Run
FROM --platform=linux/amd64 python:3.9-slim-buster

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .

# Install necessary dependencies
RUN pip install --no-cache-dir -r requirements.txt fastapi uvicorn redis litellm python-dotenv

# Copy application code
COPY . .

# Create directory for credentials
RUN mkdir -p /app/credentials

# Expose port 8080
EXPOSE 8080

# Start the application with Uvicorn and run the worker process in the background
CMD ["uvicorn", "worker:app", "--host", "0.0.0.0", "--port", "8080"]