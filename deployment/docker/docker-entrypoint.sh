#!/bin/bash
# Docker entrypoint for Ollama server

echo "=== Starting Ollama Server ==="
echo "Time: $(date)"

# Start Ollama server
export OLLAMA_HOST=0.0.0.0
echo "Starting Ollama server on 0.0.0.0:11434..."
ollama serve &

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

# List available models
echo ""
echo "Available models:"
ollama list

echo ""
echo "=== Ollama Ready ==="
echo "API endpoint: http://0.0.0.0:11434"
echo ""

# Keep container running and follow logs
tail -f /dev/null