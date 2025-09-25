#!/bin/bash
# Build and push Docker image for RunPod

# Load environment variables from .env if it exists
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Configuration - get from environment or use default
DOCKER_USERNAME="${DOCKER_USERNAME}"
IMAGE_NAME="ollama-gemma3n"
TAG="latest"

# Check if Docker username is set
if [ -z "$DOCKER_USERNAME" ]; then
    echo "Error: DOCKER_USERNAME environment variable is not set"
    echo ""
    echo "Please set it using one of these methods:"
    echo "  1. Export: export DOCKER_USERNAME=yourusername"
    echo "  2. Inline: DOCKER_USERNAME=yourusername ./docker-build.sh"
    echo "  3. Add to .env file: DOCKER_USERNAME=yourusername"
    exit 1
fi

FULL_IMAGE="${DOCKER_USERNAME}/${IMAGE_NAME}:${TAG}"

echo "Building Docker image: ${FULL_IMAGE}"
echo "======================================="

# Build for linux/amd64 platform (required for RunPod)
docker build --platform linux/amd64 -t ${FULL_IMAGE} .

if [ $? -eq 0 ]; then
    echo ""
    echo "Build successful!"
    echo ""
    echo "To push to Docker Hub:"
    echo "  1. Login: docker login"
    echo "  2. Push:  docker push ${FULL_IMAGE}"
    echo ""
    echo "To test locally:"
    echo "  docker run -p 11434:11434 ${FULL_IMAGE}"
    echo ""
    echo "To use with RunPod:"
    echo "  python scripts/setup_runpod.py --docker-image ${FULL_IMAGE}"
else
    echo "Build failed!"
    exit 1
fi