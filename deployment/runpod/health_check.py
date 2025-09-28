#!/usr/bin/env python3
"""Health check for Ollama server with pod status"""

import sys
import os
import json
import requests
from pathlib import Path

# Load environment variables from .env file if it exists
# Try multiple locations for .env file
possible_env_files = [
    Path(__file__).parent.parent.parent / '.env',  # Project root
    Path(__file__).parent.parent / '.env',  # deployment/
    Path.cwd() / '.env'  # Current directory
]

for env_file in possible_env_files:
    if env_file.exists():
        with open(env_file, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key] = value
        break

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
            name
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
                name = pod.get('name', 'Unknown')

                # Try to find correct URL from pod info
                if pod.get('runtime', {}).get('ports'):
                    for port in pod['runtime']['ports']:
                        if port.get('privatePort') == 11434 and port.get('ip') and port.get('publicPort'):
                            correct_url = f"http://{port['ip']}:{port['publicPort']}"
                            return status, uptime, correct_url, name

                return status, uptime, None, name
    except Exception as e:
        print(f"  Error getting pod status: {e}")

    return None, None, None, None

def check_ollama(host=None):
    """Check if Ollama is running"""
    if not host:
        host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')

    print(f"Checking Ollama at: {host}")

    try:
        # Try different endpoints
        endpoints = ['/api/tags', '/api/version', '']

        for endpoint in endpoints:
            try:
                url = f"{host}{endpoint}"
                print(f"  Trying: {url}")
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    print(f"✓ Ollama is running at {host}")

                    # Try to get model list
                    try:
                        models_response = requests.get(f"{host}/api/tags", timeout=5)
                        if models_response.status_code == 200:
                            models_data = models_response.json()
                            if 'models' in models_data and models_data['models']:
                                print(f"  Available models: {', '.join([m['name'] for m in models_data['models']])}")
                            else:
                                print("  ⚠ No models loaded yet. Models may still be downloading...")
                    except:
                        pass

                    return True
                else:
                    print(f"    Response: {response.status_code}")
            except requests.exceptions.ConnectTimeout:
                print(f"    Timeout")
            except requests.exceptions.ConnectionError as e:
                print(f"    Connection error")
            except Exception as e:
                print(f"    Error: {e}")

        print(f"✗ Cannot connect to Ollama at {host}")

        # If it's a RunPod URL, try to check pod status
        if 'runpod' in host or 'proxy' in host:
            # Extract pod ID from URL
            pod_id = None
            if '.runpod.io' in host:
                # Format: http://pod-id.runpod.io:11434
                pod_id = host.split('//')[1].split('.')[0]
            elif 'proxy.runpod.net' in host:
                # Format: https://pod-id-11434.proxy.runpod.net
                pod_id = host.split('//')[1].split('-')[0]

            if pod_id:
                api_key = os.getenv('RUNPOD_API_KEY')
                if api_key:
                    print(f"\nChecking pod status for {pod_id}...")
                    status, uptime, correct_url, name = get_pod_status(api_key, pod_id)

                    if status:
                        print(f"  Pod Name: {name}")
                        print(f"  Pod Status: {status}")
                        print(f"  Uptime: {uptime} seconds ({uptime//60} minutes)")

                        if status != "RUNNING":
                            print(f"  ⚠ Pod is not running yet, please wait...")
                        elif uptime < 120:
                            print(f"  ⚠ Pod just started, Ollama may still be initializing...")
                            print(f"     Models may still be downloading. This can take 5-10 minutes for large models.")

                        if correct_url and correct_url != host:
                            print(f"\n  Try this URL instead: {correct_url}")
                            print(f"  Export: export OLLAMA_HOST={correct_url}")
                    else:
                        print(f"  Could not get pod status (check RUNPOD_API_KEY in .env)")

                    # Suggest alternative URLs
                    print(f"\nAlternative URLs to try:")
                    print(f"  1. https://{pod_id}-11434.proxy.runpod.net")
                    print(f"  2. http://{pod_id}.runpod.io:11434")
                    print(f"  3. SSH into pod: ssh root@{pod_id}.runpod.io")
                    print(f"     Then check: curl http://localhost:11434/api/tags")
                else:
                    print("\n  ⚠ RUNPOD_API_KEY not found in environment")
                    print("    Add it to .env file to check pod status")

        return False

    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        host = sys.argv[1]
    else:
        # Check if there's a saved pod ID
        pod_id_file = Path('.runpod_pod_id')
        if pod_id_file.exists():
            with open(pod_id_file) as f:
                pod_id = f.read().strip()
            print(f"Found saved pod ID: {pod_id}")
            host = f"http://{pod_id}.runpod.io:11434"
        else:
            host = None

    if not check_ollama(host):
        print("\nTroubleshooting tips:")
        print("1. Ensure pod is fully started (wait 2-3 minutes after creation)")
        print("2. Models download on first startup (can take 5-10 minutes)")
        print("3. Check pod logs via SSH or RunPod dashboard")
        print("4. Verify port 11434 is exposed in pod settings")
        sys.exit(1)