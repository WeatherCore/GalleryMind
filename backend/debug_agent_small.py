
import requests
import json

BASE_URL = "http://localhost:3001"

def test_agent_small_image():
    print("\n🔍 Debugging Agent Chat with Small Image...")
    
    # Tiny 1x1 GIF Base64
    b64_data = "R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"

    params = {
        "message": "Test tiny image",
        "image": b64_data 
    }
    
    url = f"{BASE_URL}/api/agent/chat"
    
    print(f"Sending request to {url}...")
    try:
        with requests.get(url, params=params, stream=True, timeout=120) as resp:
            print(f"Status Code: {resp.status_code}")
            if resp.status_code == 200:
                 print("✅ Stream started successfully with small image.")
            else:
                print(f"❌ Response Error: {resp.status_code} {resp.text}")
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_agent_small_image()
