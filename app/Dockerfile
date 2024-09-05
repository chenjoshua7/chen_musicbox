# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install system dependencies required for PyTorch and ODBC drivers
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    unixodbc \
    unixodbc-dev \
    gcc \
    curl \
    g++ \
    && apt-get clean

# Install the ODBC Driver for SQL Server
RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - \
    && curl https://packages.microsoft.com/config/debian/10/prod.list > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y msodbcsql17 \
    && apt-get clean

# Install Python dependencies including pyodbc
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir "fastapi==0.112.1" "uvicorn==0.30.1" "torch==2.2.0" "transformers==4.37.2" "pydantic==2.6.1" "numpy<2.0.0" "pyodbc"

# Expose port 80 to the outside world
EXPOSE 80

# Define environment variable
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
