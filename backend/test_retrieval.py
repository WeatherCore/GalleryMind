
import requests
import base64
import os
from pathlib import Path

BASE_URL = "http://localhost:3001"

def test_root():
    print("Testing Root Endpoint...")
    try:
        resp = requests.get(f"{BASE_URL}/", timeout=5)
        print(f"Root: {resp.status_code}, {resp.json()}")
    except Exception as e:
        print(f"Root failed: {e}")

def test_text_search():
    print("\nTesting Text Search...")
    payload = {
        "textQuery": "架构图",
        "searchMode": "文搜图",
        "recallTopK": 10,
        "rerankTopK": 3
    }
    try:
        resp = requests.post(f"{BASE_URL}/api/search", json=payload, timeout=60)
        if resp.status_code == 200:
            data = resp.json()
            results = data.get("results", [])
            print(f"Standard Search Results Count: {len(results)}")
            for r in results:
                print(f"  - {r.get('metadata', {}).get('file_name')} (Score: {r.get('score')})")
            
            # Verify Top-K
            if len(results) <= 3:
                 print("✅ Top-K check passed (<= 3)")
            else:
                 print(f"❌ Top-K check failed: Expected <= 3, got {len(results)}")
        else:
            print(f"Error: {resp.text}")
    except Exception as e:
        print(f"Text search failed: {e}")

def test_top_k_parameter():
    print("\nTesting Top-K Parameter Control...")
    k_values = [1, 5]
    for k in k_values:
        payload = {
            "textQuery": "系统设计",
            "searchMode": "文搜图",
            "recallTopK": 10,
            "rerankTopK": k
        }
        try:
            resp = requests.post(f"{BASE_URL}/api/search", json=payload, timeout=60)
            if resp.status_code == 200:
                count = len(resp.json().get("results", []))
                print(f"Requested k={k}, Got {count} results")
                if count <= k: 
                     print(f"  ✅ k={k} PASSED")
                else:
                     print(f"  ❌ k={k} FAILED")
            else:
                print(f"Error {resp.status_code}: {resp.text}")
        except Exception as e:
            print(f"Top-K test failed: {e}")

def test_image_search_mock():
    print("\nTesting Image Search (Image-to-Image)...")
    
    # Try to find a local image to use
    backend_dir = Path("backend")
    image_dir = backend_dir / "data" / "images"
    test_img = None
    if image_dir.exists():
        images = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))
        if images:
            test_img = images[0]
            print(f"Using test image: {test_img}")
    
    if test_img:
        with open(test_img, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode()
            
        payload = {
            "searchMode": "图搜图",
            "uploadedImage": b64_data, # Schema field name
            "rerankTopK": 3
        }
        try:
            resp = requests.post(f"{BASE_URL}/api/search", json=payload, timeout=60)
            print(f"Image Search Status: {resp.status_code}")
            if resp.status_code == 200:
                print(f"Results: {len(resp.json().get('results', []))}")
            else:
                print(f"Error: {resp.text}")
        except Exception as e:
            print(f"Image search failed: {e}")
    else:
        print("⚠️ No test image found to run image search test.")

if __name__ == "__main__":
    test_root()
    test_text_search()
    test_top_k_parameter()
    test_image_search_mock()
