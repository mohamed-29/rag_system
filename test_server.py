import requests
import time
import sys

def test_chat():
    print("Testing /chat endpoint...")
    url = "http://127.0.0.1:5000/chat"
    
    # Wait for server to start
    for i in range(10):
        try:
            response = requests.get("http://127.0.0.1:5000/")
            if response.status_code == 200:
                print("Server is up!")
                break
        except requests.exceptions.ConnectionError:
            print(f"Waiting for server... ({i+1}/10)")
            time.sleep(2)
    else:
        print("Server failed to start in time.")
        sys.exit(1)

    # Test Chat
    payload = {
        "message": "Hello, what is this document about?",
        "session_id": "test_session_1"
    }
    
    try:
        print(f"Sending payload: {payload}")
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        
        print("Response received:")
        print(data)
        
        if "response" in data and "session_id" in data:
            print("TEST PASSED: Valid response format.")
        else:
            print("TEST FAILED: Invalid response format.")
            sys.exit(1)

    except Exception as e:
        print(f"TEST FAILED: {e}")
        if response:
            print(f"Status Code: {response.status_code}")
            print(f"Content: {response.text}")
        sys.exit(1)

if __name__ == "__main__":
    test_chat()
