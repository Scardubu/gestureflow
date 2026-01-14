"""Pydantic schemas for GestureFlow API."""
from pydantic import BaseModel, Field
from typing import List


class GesturePoint(BaseModel):
    """Single point in gesture trajectory."""
    x: float = Field(..., ge=0, le=1, description="Normalized x coordinate")
    y: float = Field(..., ge=0, le=1, description="Normalized y coordinate")
    timestamp: float = Field(..., ge=0, description="Timestamp in milliseconds")


class PredictionRequest(BaseModel):
    """Request for swipe prediction."""
    trajectory: List[GesturePoint] = Field(
        ..., 
        min_items=3,
        description="List of gesture points"
    )
    language: str = Field(
        default="en",
        description="Language code (en, es, fr)"
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of predictions to return"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "trajectory": [
                    {"x": 0.18, "y": 0.33, "timestamp": 0},
                    {"x": 0.28, "y": 0.33, "timestamp": 50},
                    {"x": 0.48, "y": 0.33, "timestamp": 100},
                ],
                "language": "en",
                "top_k": 5
            }
        }


class Prediction(BaseModel):
    """Single prediction result."""
    word: str
    confidence: float = Field(..., ge=0, le=1)
    rank: int = Field(..., ge=1)


class PredictionResponse(BaseModel):
    """Response with predictions."""
    predictions: List[Prediction]
    inference_time_ms: float
    num_points: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "predictions": [
                    {"word": "sad", "confidence": 0.234, "rank": 1},
                    {"word": "as", "confidence": 0.189, "rank": 2},
                ],
                "inference_time_ms": 42.3,
                "num_points": 5
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    loaded_models: int


class LanguagesResponse(BaseModel):
    """Available languages response."""
    available: List[str]
    loaded: List[str]


class ModelInfoResponse(BaseModel):
    """Model information response."""
    language: str
    vocab_size: int
    model_params: int
    model_size_mb: float
    architecture: str