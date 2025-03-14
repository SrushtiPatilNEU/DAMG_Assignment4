FROM --platform=linux/amd64 python:3.9-slim-buster

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose the correct port for Google Cloud Run
ENV STREAMLIT_SERVER_PORT=8080

EXPOSE 8080

CMD ["streamlit", "run", "app.py", "--server.port", "8080"]
