"""
Tests for API endpoints.
"""
import pytest
from fastapi.testclient import TestClient
import numpy as np

from api.main import app


client = TestClient(app)


class TestAPIEndpoints:
    """Test API endpoints."""
    
    def test_root_endpoint(self):
        """Test root endpoint."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "message" in data
        assert "predictor_loaded" in data
    
    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert data["status"] in ["healthy", "degraded"]
    
    def test_predict_endpoint_structure(self):
        """Test predict endpoint request structure."""
        # Valid request
        request_data = {
            "trajectory": [[0.1, 0.2], [0.15, 0.25], [0.2, 0.3]],
            "timestamps": [0.0, 0.5, 1.0],
            "top_k": 5
        }
        
        response = client.post("/predict", json=request_data)
        
        # Response could be 200 (if predictor loaded) or 503 (if not loaded)
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "predictions" in data
            assert "trajectory_length" in data
    
    def test_predict_endpoint_invalid_trajectory(self):
        """Test predict endpoint with invalid trajectory."""
        # Invalid trajectory (wrong shape)
        request_data = {
            "trajectory": [[0.1], [0.15], [0.2]],  # Should be (N, 2)
            "timestamps": [0.0, 0.5, 1.0],
            "top_k": 5
        }
        
        response = client.post("/predict", json=request_data)
        
        # Should return 400 or 503 (if predictor not loaded)
        assert response.status_code in [400, 503]
    
    def test_predict_endpoint_mismatched_lengths(self):
        """Test predict endpoint with mismatched trajectory and timestamps."""
        request_data = {
            "trajectory": [[0.1, 0.2], [0.15, 0.25]],
            "timestamps": [0.0, 0.5, 1.0],  # Length mismatch
            "top_k": 5
        }
        
        response = client.post("/predict", json=request_data)
        
        # Should return 400 or 503 (if predictor not loaded)
        assert response.status_code in [400, 503]
    
    def test_predict_batch_endpoint(self):
        """Test batch prediction endpoint."""
        requests_data = [
            {
                "trajectory": [[0.1, 0.2], [0.15, 0.25], [0.2, 0.3]],
                "timestamps": [0.0, 0.5, 1.0],
                "top_k": 3
            },
            {
                "trajectory": [[0.2, 0.3], [0.25, 0.35], [0.3, 0.4]],
                "timestamps": [0.0, 0.5, 1.0],
                "top_k": 3
            }
        ]
        
        response = client.post("/predict_batch", json=requests_data)
        
        # Response could be 200 (if predictor loaded) or 503 (if not loaded)
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list)
            assert len(data) == len(requests_data)


class TestAPISchemas:
    """Test API schemas."""
    
    def test_gesture_request_validation(self):
        """Test GestureRequest validation."""
        # Valid request
        valid_request = {
            "trajectory": [[0.1, 0.2], [0.15, 0.25]],
            "timestamps": [0.0, 1.0],
            "top_k": 5
        }
        
        response = client.post("/predict", json=valid_request)
        assert response.status_code in [200, 503]  # Valid request structure
    
    def test_gesture_request_defaults(self):
        """Test GestureRequest default values."""
        # Request without optional fields
        minimal_request = {
            "trajectory": [[0.1, 0.2], [0.15, 0.25]],
            "timestamps": [0.0, 1.0]
        }
        
        response = client.post("/predict", json=minimal_request)
        assert response.status_code in [200, 503]  # Should use defaults
    
    def test_top_k_validation(self):
        """Test top_k parameter validation."""
        # Invalid top_k (too large)
        request_data = {
            "trajectory": [[0.1, 0.2], [0.15, 0.25]],
            "timestamps": [0.0, 1.0],
            "top_k": 100  # Should be <= 20
        }
        
        response = client.post("/predict", json=request_data)
        
        # Should return validation error (422) or 503 if predictor not loaded
        assert response.status_code in [422, 503]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
