
import requests
import json

BASE_URL = "http://localhost:3001"

def test_text_search_debug():
    print("\n🔍 Debugging Text Search (Threshold=0.5)...")
    payload = {
        "textQuery": "架构图",
        "searchMode": "文搜图",
        "recallTopK": 30,
        "rerankTopK": 6,
        "threshold": 0.4  # Adjusted based on observed scores (~0.43)
    }
    try:
        resp = requests.post(f"{BASE_URL}/api/search", json=payload, timeout=60)
        print(f"Status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            results = data.get("results", [])
            print(f"Results Count: {len(results)}")
            for r in results:
                print(f"  - {r.get('title')} (Score: {r.get('score')})")
        else:
            print(f"Error: {resp.text}")
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_text_search_debug()
