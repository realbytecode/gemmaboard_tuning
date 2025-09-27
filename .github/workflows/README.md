# GitHub Actions CI/CD Setup

## Overview

This workflow automatically builds and deploys Docker images to Docker Hub whenever code is pushed to the master branch.

## Setup Instructions

### 1. Create Docker Hub Access Token

1. Log in to [Docker Hub](https://hub.docker.com/)
2. Go to Account Settings → Personal Access Tokens
3. Click "New Access Token"
4. Give it a name (e.g., "github-actions")
5. Copy the generated token

### 2. Add GitHub Secrets

In your GitHub repository:

1. Go to Settings → Secrets and variables → Actions
2. Add the following secrets:

| Secret Name | Description | Example |
|------------|-------------|---------|
| `DOCKER_USERNAME` | Your Docker Hub username | `yourusername` |
| `DOCKER_TOKEN` | Docker Hub access token (not password!) | `dckr_pat_...` |

### 3. How It Works

When manually triggered:
1. Builds the Docker image from `deployment/docker/`
2. Tags it with:
   - `latest` - Always points to newest build
   - `YYYYMMDD-HHMMSS` - Timestamp for specific versions
3. Pushes both tags to Docker Hub

### 4. Using the Images

After a successful build, the images are available at:
- `yourusername/ollama-runtime:latest` - Latest version
- `yourusername/ollama-runtime:20240125-143022` - Specific version

Use in RunPod:
```bash
python deployment/runpod/setup_runpod.py \
  --docker-image yourusername/ollama-runtime:latest
```

### 5. Manual Trigger

You can also manually trigger the workflow:
1. Go to Actions tab in GitHub
2. Select "Docker Build and Deploy"
3. Click "Run workflow"

### 6. Build Status

The build status is shown in the Actions tab. Failed builds will show error details.

## Troubleshooting

### Build Fails with Authentication Error
- Verify `DOCKER_USERNAME` and `DOCKER_TOKEN` secrets are set correctly
- Ensure the token has push permissions

### Image Not Found on Docker Hub
- Check if the build completed successfully in Actions tab
- Verify the Docker Hub repository is public or you're logged in

### RunPod Can't Pull Image
- Ensure the Docker Hub repository is public
- Or configure RunPod with Docker Hub credentials