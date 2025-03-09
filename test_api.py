import requests
import json
import time

# Base URL of your API
BASE_URL = 'http://localhost:5000'
API_PREFIX = '/api'  # Your routes use /api prefix according to docs

def test_api_root():
    """Test if API is online and accessible at root endpoint"""
    try:
        response = requests.get(f"{BASE_URL}")
        print(f"API Root Status: {response.status_code}")
        print(response.json())
        return response.status_code == 200
    except Exception as e:
        print(f"Error connecting to API root: {e}")
        return False

def test_api_docs():
    """Test the API documentation endpoint"""
    try:
        response = requests.get(f"{BASE_URL}{API_PREFIX}/docs")
        print(f"API Docs Status: {response.status_code}")
        print("Available endpoints:")
        for endpoint in response.json().get('endpoints', []):
            print(f"  {endpoint['method']} {endpoint['path']}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error accessing API docs: {e}")
        return False

def test_vectorization(text_inputs):
    """Test the vectorization endpoint"""
    try:
        response = requests.post(
            f"{BASE_URL}{API_PREFIX}/vectorize",
            json={"texts": text_inputs},
            headers={"Content-Type": "application/json"}
        )
        print(f"Vectorization Status: {response.status_code}")
        if response.status_code == 200:
            print("Successfully vectorized text")
            # Print first few dimensions of first vector to verify
            result = response.json()
            if "vectors" in result and len(result["vectors"]) > 0:
                print(f"First vector (truncated): {result['vectors'][0][:5]}...")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Error with vectorization endpoint: {e}")
        return False

def test_build_graph(event_id):
    """Test the graph building endpoint"""
    try:
        response = requests.post(
            f"{BASE_URL}{API_PREFIX}/events/{event_id}/build_graph",
            headers={"Content-Type": "application/json"}
        )
        print(f"Build Graph Status: {response.status_code}")
        if response.status_code == 200:
            print("Successfully built graph")
            print(response.json())
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Error with build graph endpoint: {e}")
        return False

def test_optimization(event_id, config):
    """Test the optimization endpoint"""
    try:
        response = requests.post(
            f"{BASE_URL}{API_PREFIX}/events/{event_id}/optimize",
            json={"config": config},
            headers={"Content-Type": "application/json"}
        )
        print(f"Optimization Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            job_id = result.get("job_id")
            print(f"Optimization job started with ID: {job_id}")
            return job_id
        else:
            print(f"Error: {response.text}")
            return None
    except Exception as e:
        print(f"Error with optimization endpoint: {e}")
        return None

def test_get_results(event_id, job_id):
    """Test getting optimization results"""
    try:
        response = requests.get(
            f"{BASE_URL}{API_PREFIX}/events/{event_id}/optimize/{job_id}/results"
        )
        print(f"Results Status: {response.status_code}")
        if response.status_code == 200:
            print("Successfully retrieved results")
            print(json.dumps(response.json(), indent=2))
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Error with results endpoint: {e}")
        return False

# Run tests
if __name__ == "__main__":
    print("Testing API Endpoints...")
    
    # Test API root to confirm server is running
    if not test_api_root():
        print("API not accessible at root URL. Please ensure it's running.")
        exit(1)
    
    # Test API docs
    test_api_docs()
    
    # Sample data for testing
    event_id = "test123"  # Replace with an actual event ID
    sample_texts = [
        "I want to sit with my family",
        "Please don't seat me near my ex",
        "I'm friends with the bride's side"
    ]
    
    optimization_config = {
        "num_tables": 5,
        "seats_per_table": 8,
        "weight_relationships": 0.7,
        "weight_preferences": 0.3
    }
    
    # Run the vectorization test
    test_vectorization(sample_texts)
    
    # Run the graph building test
    test_build_graph(event_id)
    
    # Run the optimization test and get results
    job_id = test_optimization(event_id, optimization_config)
    if job_id:
        print("Waiting a moment for optimization to complete...")
        time.sleep(5)  # Give it some time to process
        test_get_results(event_id, job_id)
    
    print("API testing complete!")