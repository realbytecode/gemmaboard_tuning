#!/usr/bin/env python3
"""Simple RunPod setup script"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
import requests

# Load environment variables from .env file if it exists
env_file = Path(__file__).parent.parent.parent / '.env'
if env_file.exists():
    with open(env_file, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value

RUNPOD_API_URL = "https://api.runpod.io/graphql"

# Use user's Docker image if DOCKER_USERNAME is set, otherwise use default
docker_username = os.getenv('DOCKER_USERNAME')
if docker_username:
    DEFAULT_DOCKER_IMAGE = f"{docker_username}/ollama-gemma3n:latest"
else:
    DEFAULT_DOCKER_IMAGE = "realbytecode/ollama-gemma3n:latest"


def test_api_key(api_key):
    """Test if API key is valid"""
    headers = {
        "Authorization": f"{api_key}",
        "Content-Type": "application/json"
    }

    query = """
    query {
        myself {
            id
            email
        }
    }
    """

    try:
        response = requests.post(
            RUNPOD_API_URL,
            json={"query": query},
            headers=headers,
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            if "data" in data and "myself" in data["data"]:
                return True
    except Exception as e:
        print(f"API test error: {e}")

    return False


def get_pod_info(api_key, pod_id):
    """Get pod information"""
    headers = {
        "Authorization": f"{api_key}",
        "Content-Type": "application/json"
    }

    query = """
    query {
        pod(input: {podId: "%s"}) {
            id
            name
            runtime {
                uptimeInSeconds
                ports {
                    ip
                    publicPort
                    privatePort
                }
            }
        }
    }
    """ % pod_id

    try:
        response = requests.post(
            RUNPOD_API_URL,
            json={"query": query},
            headers=headers,
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            if "data" in data and "pod" in data["data"]:
                return data["data"]["pod"]
    except Exception as e:
        print(f"Error getting pod info: {e}")

    return None


def get_existing_pod(api_key, pod_name="ollama-server"):
    """Check if a pod with the given name already exists"""
    headers = {
        "Authorization": f"{api_key}",
        "Content-Type": "application/json"
    }

    query = """
    query {
        myself {
            pods {
                id
                name
                desiredStatus
            }
        }
    }
    """

    try:
        response = requests.post(
            RUNPOD_API_URL,
            json={"query": query},
            headers=headers,
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            if "data" in data and "myself" in data["data"]:
                if "pods" in data["data"]["myself"]:
                    for pod in data["data"]["myself"]["pods"]:
                        if pod.get("name") == pod_name:
                            return pod.get("id")
    except Exception as e:
        print(f"Error checking existing pods: {e}")

    return None


def create_pod(api_key, gpu_type="NVIDIA GeForce RTX 3090",
               docker_image=None):
    """Create RunPod pod with Ollama"""

    # Check for existing pod with same name
    existing_pod_id = get_existing_pod(api_key, "ollama-server")
    if existing_pod_id:
        print(f"Found existing pod with same name (ID: {existing_pod_id})")
        print("Deleting existing pod...")
        if stop_pod(api_key, existing_pod_id):
            print("Existing pod deleted successfully")
            time.sleep(5)
        else:
            print("Warning: Could not delete existing pod")

    # Use provided image or default
    image_name = docker_image if docker_image else DEFAULT_DOCKER_IMAGE
    print(f"Creating pod with {gpu_type}...")
    print(f"Using Docker image: {image_name}")

    # Map common names to RunPod GPU IDs
    gpu_map = {
        "RTX 3090": "NVIDIA GeForce RTX 3090",
        "RTX 4090": "NVIDIA GeForce RTX 4090",
        "A40": "NVIDIA A40",
        "A5000": "NVIDIA RTX A5000",
        "A4000": "NVIDIA RTX A4000"
    }

    # Use mapped name if available
    for short_name, full_name in gpu_map.items():
        if short_name in gpu_type:
            gpu_type = full_name
            print(f"Using GPU: {gpu_type}")
            break

    headers = {
        "Authorization": f"{api_key}",
        "Content-Type": "application/json"
    }

    mutation = """
    mutation {
        podFindAndDeployOnDemand(
            input: {
                cloudType: COMMUNITY
                gpuCount: 1
                volumeInGb: 40
                containerDiskInGb: 40
                minVcpuCount: 4
                minMemoryInGb: 8
                gpuTypeId: "%s"
                name: "ollama-server"
                imageName: "%s"
                dockerArgs: ""
                ports: "11434/http"
                volumeMountPath: "/workspace"
            }
        ) {
            id
            name
            runtime {
                ports {
                    ip
                    publicPort
                    privatePort
                }
            }
        }
    }
    """ % (gpu_type, image_name)

    try:
        response = requests.post(
            RUNPOD_API_URL,
            json={"query": mutation},
            headers=headers,
            timeout=30
        )

        if response.status_code != 200:
            print(f"Error: API returned status {response.status_code}")
            print(f"Response: {response.text}")
            return None

        data = response.json()

        if "errors" in data:
            print(f"Error: {data['errors']}")
            return None

        if "data" in data and data["data"]:
            if "podFindAndDeployOnDemand" in data["data"]:
                pod = data["data"]["podFindAndDeployOnDemand"]
                pod_id = pod["id"]
                print(f"Created pod: {pod_id}")

                # Wait and get pod info
                print("Waiting for pod to start...")
                print("(this may take a few minutes)")
                time.sleep(60)

                # Get pod details to find the URL
                pod_info = get_pod_info(api_key, pod_id)
                if pod_info:
                    print("\nPod is ready!")
                    print(f"SSH Command: ssh root@{pod_id}.runpod.io")
                    print(f"\nOllama URL (after pod starts):")
                    print(f"  http://{pod_id}.runpod.io:11434")
                    print("\nExport for local use:")
                    print(f"  export OLLAMA_HOST=http://{pod_id}.runpod.io:11434")
                else:
                    print("\nPod created but URL not yet available.")
                    print(f"Predicted URL: http://{pod_id}.runpod.io:11434")
                    print("\nTest with:")
                    print(f"  python deployment/runpod/health_check.py \\")
                    print(f"    http://{pod_id}.runpod.io:11434")

                return pod_id

        print("Error: Unexpected response format")
        print(json.dumps(data, indent=2))
        return None

    except Exception as e:
        print(f"Error creating pod: {e}")
        return None


def stop_pod(api_key, pod_id):
    """Stop a pod"""
    headers = {
        "Authorization": f"{api_key}",
        "Content-Type": "application/json"
    }

    mutation = """
    mutation {
        podStop(input: {podId: "%s"}) {
            id
        }
    }
    """ % pod_id

    try:
        response = requests.post(
            RUNPOD_API_URL,
            json={"query": mutation},
            headers=headers,
            timeout=10
        )

        if response.status_code == 200:
            print(f"Stopped pod {pod_id}")
            return True
    except Exception as e:
        print(f"Error stopping pod: {e}")

    return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Setup RunPod with Ollama')
    parser.add_argument('--api-key', help='RunPod API key (overrides env)')
    parser.add_argument('--gpu-type', default='RTX 3090',
                        help='GPU type: RTX 3090, RTX 4090, A40, A5000')
    parser.add_argument('--pod-id', help='Pod ID to stop')
    parser.add_argument('--stop', action='store_true', help='Stop the pod')
    parser.add_argument('--replace', action='store_true',
                        help='Delete existing pod before creating new one')
    parser.add_argument('--docker-image',
                        help='Custom Docker image (default: realbytecode/ollama-gemma3n)')
    args = parser.parse_args()

    # Use API key from args or environment
    api_key = args.api_key or os.getenv('RUNPOD_API_KEY')

    if not api_key:
        print("Error: Need RunPod API key")
        print("  --api-key or RUNPOD_API_KEY in .env")
        sys.exit(1)

    # Validate API key
    if not api_key.startswith('rpa_'):
        print("Error: Invalid API key format. Should start with 'rpa_'")
        sys.exit(1)

    print("Testing API key...")
    if not test_api_key(api_key):
        print("Error: Invalid API key or unable to connect to RunPod API")
        print("Please check your API key in .env file")
        sys.exit(1)
    print("API key validated")

    if args.pod_id and args.stop:
        stop_pod(api_key, args.pod_id)
    else:
        # Check for existing pod from saved file if --replace flag is used
        if args.replace and os.path.exists('.runpod_pod_id'):
            with open('.runpod_pod_id', 'r', encoding='utf-8') as f:
                saved_pod_id = f.read().strip()
            if saved_pod_id:
                print(f"Found saved pod ID: {saved_pod_id}")
                print("Deleting existing pod...")
                if stop_pod(api_key, saved_pod_id):
                    print("Existing pod deleted")
                    time.sleep(5)

        pod_id = create_pod(api_key, args.gpu_type, args.docker_image)
        if pod_id:
            with open('.runpod_pod_id', 'w', encoding='utf-8') as f:
                f.write(pod_id)
            print("\nPod ID saved to .runpod_pod_id")


if __name__ == "__main__":
    main()
