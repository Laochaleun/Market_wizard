# Use Python 3.11 as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies if needed (e.g. for potential audio/image processing later)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire project
COPY . /app

# Install backend package and dependencies
RUN cd backend && pip install --no-cache-dir .

# Expose the port Gradio will run on (7860 is standard for HF Spaces)
EXPOSE 7860

# Set environment variables
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT=7860
ENV PYTHONPATH="${PYTHONPATH}:/app/backend"

# Command to run the application
# We run the frontend directly, which imports the backend
CMD ["python", "frontend/main.py"]
