# Stage 1: Build stage using Python slim image
FROM python:3.11-slim AS builder

# Set the working directory inside the container
WORKDIR /app

# Install required system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements file to the container
COPY requirements.txt .

# Install the dependencies without caching to minimize size
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Final image using Python slim base
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install minimal system dependencies required for the final image
RUN apt-get update && apt-get install -y --no-install-recommends \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy installed dependencies from the builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the entire application code to the container
COPY . .

# Expose the port on which FastAPI will run (default is 8000)
EXPOSE 8000

# Define the command to run your FastAPI app using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
