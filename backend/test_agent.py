
import requests
import json
import base64
from pathlib import Path

BASE_URL = "http://localhost:3001"

def test_agent_chat_stream():
    print("Testing Agent Chat Stream...")
    
    query = "找一张架构图"
    url = f"{BASE_URL}/api/agent/chat"
    params = {"message": query}
    
    print(f"Sending query: {query}")
    
    print(f"Sending query: {query}")
    
    try:
        with requests.get(url, params=params, stream=True, timeout=180) as resp:
            print(f"Status Code: {resp.status_code}")
            if resp.status_code == 200:
                print("Stream started...")
                for line in resp.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        if decoded_line.startswith("data: "):
                            data_str = decoded_line[6:] # Removing "data: " prefix
                            print(f"Received: {data_str}")
            else:
                 print(f"Error: {resp.text}")
    except Exception as e:
        print(f"Stream failed: {e}")

def test_agent_image_chat():
    print("\nTesting Agent Image Chat...")
    # Try to find a local image
    backend_dir = Path("backend")
    image_dir = backend_dir / "data" / "images"
    test_img = None
    if image_dir.exists():
        images = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))
        if images:
            test_img = images[0]
    
    if test_img:
        print(f"Using image: {test_img}")
        with open(test_img, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode()
            
        params = {
            "message": "这张图是什么？",
            "image": b64_data
        }
        url = f"{BASE_URL}/api/agent/chat"
        
        try:
            with requests.get(url, params=params, stream=True) as resp:
                if resp.status_code == 200:
                    for line in resp.iter_lines():
                        if line:
                             print(f"Received: {line.decode('utf-8')}")
        except Exception as e:
            print(f"Image chat failed: {e}")

if __name__ == "__main__":
    test_agent_chat_stream()
    test_agent_image_chat()
