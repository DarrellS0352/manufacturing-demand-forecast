# Start from official Python 3.11 slim image
FROM python:3.11-slim

# Install system dependencies required by LightGBM
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# Set working directory inside container
WORKDIR /app

# Copy requirements first (Docker caches this layer — faster rebuilds)
COPY requirements.txt .

# Install only the packages needed to run the API
RUN pip install --no-cache-dir fastapi uvicorn lightgbm pandas numpy scikit-learn intermittent-forecast

# Copy the rest of the project
COPY . .

# Expose port 8080 (Cloud Run default)
EXPOSE 8080

# Start the FastAPI app
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]