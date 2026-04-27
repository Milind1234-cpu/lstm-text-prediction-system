"""Quick test script for the running API server."""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint."""
    print("\n🔍 Testing Health Endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_root():
    """Test root endpoint."""
    print("\n🔍 Testing Root Endpoint...")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_predict():
    """Test predict endpoint."""
    print("\n🔍 Testing Predict Endpoint...")
    data = {
        "text": "machine learning is",
        "temperature": 1.0
    }
    response = requests.post(f"{BASE_URL}/predict", json=data, timeout=30)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_model_info():
    """Test model info endpoint."""
    print("\n🔍 Testing Model Info Endpoint...")
    response = requests.get(f"{BASE_URL}/model/info")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Architecture: {json.dumps(data.get('architecture', {}), indent=2)}")
    print(f"Parameters: {json.dumps(data.get('parameters', {}), indent=2)}")
    return response.status_code == 200

def test_metrics():
    """Test metrics endpoint."""
    print("\n🔍 Testing Metrics Endpoint...")
    response = requests.get(f"{BASE_URL}/metrics")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 LSTM Text Prediction API - Live Test")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Health", test_health()))
    results.append(("Root", test_root()))
    results.append(("Model Info", test_model_info()))
    results.append(("Metrics", test_metrics()))
    results.append(("Predict", test_predict()))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name:20s} {status}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\nTotal: {passed}/{total} tests passed")
    print("=" * 60)
