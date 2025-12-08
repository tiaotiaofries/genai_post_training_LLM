"""
Test script for the RL-trained GPT2 API
Tests all endpoints and validates responses
"""
import requests
import json

BASE_URL = "http://localhost:8000"


def test_root():
    """Test root endpoint"""
    print("\n" + "="*60)
    print("Testing GET /")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 200
    print("✅ Root endpoint OK")


def test_health():
    """Test health endpoint"""
    print("\n" + "="*60)
    print("Testing GET /health")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 200
    print("✅ Health endpoint OK")


def test_generate_qa():
    """Test Q&A generation endpoint"""
    print("\n" + "="*60)
    print("Testing POST /generate_qa")
    print("="*60)
    
    test_questions = [
        "What is machine learning?",
        "How does reinforcement learning work?",
        "What are neural networks?",
    ]
    
    for question in test_questions:
        print(f"\n📝 Question: {question}")
        
        payload = {
            "question": question,
            "max_length": 80,
            "temperature": 0.7
        }
        
        response = requests.post(f"{BASE_URL}/generate_qa", json=payload)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Formatted Response:\n{result['formatted_response']}")
            print(f"\nAnswer Only:\n{result['answer_only']}")
            print("✅ Generation successful")
        else:
            print(f"❌ Error: {response.text}")
        
        print("-" * 60)


def test_format_compliance():
    """Test that responses follow the required format"""
    print("\n" + "="*60)
    print("Testing Format Compliance")
    print("="*60)
    
    payload = {
        "question": "What is AI?",
        "max_length": 60,
        "temperature": 0.7
    }
    
    response = requests.post(f"{BASE_URL}/generate_qa", json=payload)
    
    if response.status_code == 200:
        result = response.json()
        formatted_text = result['formatted_response']
        
        # Check format
        has_question = "Question:" in formatted_text
        has_answer = "Answer:" in formatted_text
        
        print(f"Generated: {formatted_text}")
        print(f"\nFormat Check:")
        print(f"  Has 'Question:': {has_question} {'✅' if has_question else '❌'}")
        print(f"  Has 'Answer:': {has_answer} {'✅' if has_answer else '❌'}")
        
        if has_question and has_answer:
            print("\n✅ Format compliance PASSED")
        else:
            print("\n❌ Format compliance FAILED")
    else:
        print(f"❌ Request failed: {response.text}")


if __name__ == "__main__":
    print("="*60)
    print("RL-TRAINED GPT2 API TESTS")
    print("="*60)
    
    try:
        test_root()
        test_health()
        test_generate_qa()
        test_format_compliance()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED")
        print("="*60)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Cannot connect to API")
        print("Make sure the API is running:")
        print("  python main.py")
        print("  OR")
        print("  docker-compose up")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
