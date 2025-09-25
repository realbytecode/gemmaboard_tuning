# RunPod Ollama Image with Gemma3n pre-installed
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Pull the model during build
# We need to start ollama serve temporarily to pull the model
RUN ollama serve & \
    sleep 5 && \
    ollama pull gemma3n:e4b && \
    pkill ollama

# Copy entrypoint script (relative to build context)
COPY ./docker-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Expose Ollama port
EXPOSE 11434

# Set entrypoint
ENTRYPOINT ["/entrypoint.sh"]