# Use an official Python runtime as a parent image
# We use a slim-bullseye image to keep the size of the final image small.
# Python 3.10 is a good choice as it is a widely supported version.
FROM python:3.10-slim-bullseye

# Set the working directory in the container
# This is where all the application code will be stored.
WORKDIR /app

# Install system dependencies needed by some Python packages.
# The 'build-essential' and 'git' packages are often required for
# certain libraries to compile correctly.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install the required Python packages
# The '--no-cache-dir' flag saves space by not storing cache files.
# The '--upgrade pip' command ensures pip is up-to-date.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all the application files into the container at /app
# This includes all Python scripts and the data directory.
COPY ./app /app/app
COPY ./data /app/data

# Expose port 8000 to the outside world
# This is the default port for the FastAPI server.
EXPOSE 8000

# Define the command to run the application
# This is the entry point for the container. It starts the Uvicorn server.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
