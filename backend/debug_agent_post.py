
import requests
import base64
import json
from pathlib import Path

BASE_URL = "http://localhost:3001"

def test_agent_post():
    print("\n🔍 Debugging Agent Chat (POST)...")
    
    image_path = Path("backend/data/images/20251216110324_89_50.png")
    if not image_path.exists():
         # Try finding any png
        images = list(Path("backend/data/images").glob("*.png"))
        if images:
            image_path = images[0]
    
    if image_path:
        with open(image_path, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode()
    else:
        b64_data = None

    payload = {
        "message": "帮我检索一下跟这张架构图类似的图片",
        "image": b64_data
    }
    
    url = f"{BASE_URL}/api/agent/chat"
    
    print(f"Sending POST request to {url}...")
    try:
        with requests.post(url, json=payload, stream=True, timeout=120) as resp:
            print(f"Status Code: {resp.status_code}")
            if resp.status_code == 200:
                print("Stream started... Listening for events:")
                for line in resp.iter_lines():
                    if line:
                        print(line.decode('utf-8'))
            else:
                print(f"Response Error: {resp.text}")
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_agent_post()
