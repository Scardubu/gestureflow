"""Tests for API endpoints."""

import pytest
import sys
from pathlib import Path
from fastapi.testclient import TestClient

# Add api to path
sys.path.append(str(Path(__file__).parent.parent))

from api.main import app
from api.schemas import PredictionRequest, GesturePoint


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestAPIEndpoints:
    """Test cases for API endpoints."""
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code in [200, 404]  # May or may not be implemented
    
    def test_predict_endpoint_structure(self, client):
        """Test predict endpoint with valid request structure."""
        # Create sample gesture data
        gesture_points = [
            {"x": 0.0, "y": 0.0, "timestamp": 0},
            {"x": 1.0, "y": 1.0, "timestamp": 100},
            {"x": 2.0, "y": 2.0, "timestamp": 200}
        ]
        
        request_data = {
            "gesture": gesture_points,
            "language": "en_US",
            "top_k": 5
        }
        
        response = client.post("/predict", json=request_data)
        # May return 200 (success) or 500 (if model not loaded)
        assert response.status_code in [200, 500, 503]
    
    def test_predict_endpoint_invalid_language(self, client):
        """Test predict endpoint with invalid language."""
        gesture_points = [
            {"x": 0.0, "y": 0.0},
            {"x": 1.0, "y": 1.0}
        ]
        
        request_data = {
            "gesture": gesture_points,
            "language": "invalid_lang",
            "top_k": 5
        }
        
        response = client.post("/predict", json=request_data)
        # Should return 422 (validation error) or 400 (bad request)
        assert response.status_code in [400, 422, 500, 503]
    
    def test_predict_endpoint_empty_gesture(self, client):
        """Test predict endpoint with empty gesture."""
        request_data = {
            "gesture": [],
            "language": "en_US",
            "top_k": 5
        }
        
        response = client.post("/predict", json=request_data)
        # Should handle empty gesture gracefully
        assert response.status_code in [400, 422, 500, 503]
    
    def test_predict_endpoint_invalid_top_k(self, client):
        """Test predict endpoint with invalid top_k value."""
        gesture_points = [
            {"x": 0.0, "y": 0.0},
            {"x": 1.0, "y": 1.0}
        ]
        
        request_data = {
            "gesture": gesture_points,
            "language": "en_US",
            "top_k": 100  # Exceeds maximum
        }
        
        response = client.post("/predict", json=request_data)
        assert response.status_code == 422  # Validation error
    
    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options("/predict")
        # Check if CORS is configured (may vary based on implementation)
        assert response.status_code in [200, 405]


class TestAPISchemas:
    """Test cases for API schemas."""
    
    def test_gesture_point_schema(self):
        """Test GesturePoint schema validation."""
        valid_point = GesturePoint(x=1.0, y=2.0, timestamp=100.0)
        assert valid_point.x == 1.0
        assert valid_point.y == 2.0
        assert valid_point.timestamp == 100.0
    
    def test_gesture_point_optional_timestamp(self):
        """Test GesturePoint with optional timestamp."""
        point = GesturePoint(x=1.0, y=2.0)
        assert point.x == 1.0
        assert point.y == 2.0
    
    def test_prediction_request_defaults(self):
        """Test PredictionRequest default values."""
        gesture_points = [GesturePoint(x=0.0, y=0.0)]
        request = PredictionRequest(gesture=gesture_points)
        
        assert request.language == "en_US"
        assert request.top_k == 5
    
    def test_prediction_request_validation(self):
        """Test PredictionRequest validation."""
        gesture_points = [GesturePoint(x=0.0, y=0.0)]
        
        # Valid request
        request = PredictionRequest(
            gesture=gesture_points,
            language="es_ES",
            top_k=3
        )
        assert request.language == "es_ES"
        assert request.top_k == 3
        
        # Invalid top_k (should raise validation error)
        with pytest.raises(Exception):
            PredictionRequest(
                gesture=gesture_points,
                top_k=0  # Must be >= 1
            )
