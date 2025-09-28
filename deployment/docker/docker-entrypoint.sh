#!/bin/bash
# Docker entrypoint with RunPod model download support

echo "=== Starting Ollama Server ==="
echo "Time: $(date)"

# Start Ollama server
export OLLAMA_HOST=0.0.0.0
echo "Starting Ollama server on 0.0.0.0:11434..."
ollama serve &
OLLAMA_PID=$!

# Wait for server to be ready
echo "Waiting for Ollama server to start..."
for i in {1..30}; do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "✓ Ollama server is running!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "✗ Failed to start Ollama server"
        exit 1
    fi
    sleep 1
done

echo ""
echo "=== Ollama Ready ==="
echo "API endpoint: http://0.0.0.0:11434"

# Check for models to download (passed via environment variable)
if [ -n "$OLLAMA_MODELS" ]; then
    echo ""
    echo "=== Model Download Starting ==="
    echo "Models to download: $OLLAMA_MODELS"
    echo ""

    # Split models by comma and download each
    IFS=',' read -ra MODELS <<< "$OLLAMA_MODELS"
    for model in "${MODELS[@]}"; do
        model=$(echo "$model" | xargs)  # Trim whitespace
        echo "Downloading: $model"
        ollama pull "$model"
        if [ $? -eq 0 ]; then
            echo "✓ Successfully downloaded: $model"
        else
            echo "✗ Failed to download: $model"
        fi
        echo ""
    done

    echo "=== Model Download Complete ==="
    ollama list
fi

echo ""

# Keep container running
wait $OLLAMA_PID