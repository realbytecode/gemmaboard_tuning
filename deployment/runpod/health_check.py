#!/usr/bin/env python3
"""Health check for Ollama server with pod status"""

import sys
import os
import json
import requests
from pathlib import Path

# Load environment variables from .env file if it exists
env_file = Path(__file__).parent.parent / '.env'
if env_file.exists():
    with open(env_file, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value

RUNPOD_API_URL = "https://api.runpod.io/graphql"

def get_pod_status(api_key, pod_id):
    """Get RunPod pod status"""
    headers = {
        "Authorization": f"{api_key}",
        "Content-Type": "application/json"
    }

    query = """
    query {
        pod(input: {podId: "%s"}) {
            id
            desiredStatus
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
                pod = data["data"]["pod"]
                status = pod.get('desiredStatus', 'Unknown')
                uptime = pod.get('runtime', {}).get('uptimeInSeconds', 0) if pod.get('runtime') else 0

                # Try to find correct URL from pod info
                if pod.get('runtime', {}).get('ports'):
                    for port in pod['runtime']['ports']:
                        if port.get('privatePort') == 11434 and port.get('ip') and port.get('publicPort'):
                            correct_url = f"http://{port['ip']}:{port['publicPort']}"
                            return status, uptime, correct_url

                return status, uptime, None
    except Exception:
        pass

    return None, None, None

def check_ollama(host=None):
    """Check if Ollama is running"""
    if not host:
        host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')

    try:
        response = requests.get(f"{host}/api/tags", timeout=5)
        if response.status_code == 200:
            print(f"✓ Ollama is running at {host}")
            return True
        else:
            print(f"✗ Ollama returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Cannot connect to Ollama at {host}")

        # If it's a RunPod URL, try to check pod status
        if 'runpod' in host:
            # Extract pod ID from URL
            pod_id = None
            if '.runpod.io' in host:
                pod_id = host.split('//')[1].split('.')[0]
            elif 'proxy.runpod.net' in host:
                pod_id = host.split('//')[1].split('-')[0]

            if pod_id:
                api_key = os.getenv('RUNPOD_API_KEY')
                if api_key:
                    print(f"\nChecking pod status for {pod_id}...")
                    status, uptime, correct_url = get_pod_status(api_key, pod_id)

                    if status:
                        print(f"  Pod Status: {status}")
                        print(f"  Uptime: {uptime} seconds")

                        if status != "RUNNING":
                            print(f"  ⚠ Pod is not running yet, please wait...")
                        elif uptime < 120:
                            print(f"  ⚠ Pod just started, Ollama may still be initializing...")

                        if correct_url and correct_url != host:
                            print(f"\n  Try this URL instead: {correct_url}")
                            print(f"  Export: export OLLAMA_HOST={correct_url}")
                    else:
                        print(f"  Could not get pod status")

                    # Suggest alternative URLs
                    print(f"\nAlternative URLs to try:")
                    print(f"  https://{pod_id}-11434.proxy.runpod.net")
                    print(f"  http://{pod_id}-11434.proxy.runpod.net")

        return False

if __name__ == "__main__":
    host = sys.argv[1] if len(sys.argv) > 1 else None
    if not check_ollama(host):
        sys.exit(1)