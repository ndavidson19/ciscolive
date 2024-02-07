# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY . /app


# Expose the port the app runs on
EXPOSE 4141

# Use the entrypoint script to boot the application, ensure it's executable
ENTRYPOINT ["bash", "-c", "chmod +x /app/entrypoint.sh && /app/entrypoint.sh"]

