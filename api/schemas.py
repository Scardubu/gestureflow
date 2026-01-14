"""
Pydantic schemas for API requests and responses.
"""
from pydantic import BaseModel, Field
from typing import List, Optional


class GestureRequest(BaseModel):
    """Request model for gesture prediction."""
    
    trajectory: List[List[float]] = Field(
        ...,
        description="List of [x, y] coordinates",
        example=[[0.1, 0.2], [0.15, 0.25], [0.2, 0.3]]
    )
    
    timestamps: List[float] = Field(
        ...,
        description="List of timestamps",
        example=[0.0, 0.5, 1.0]
    )
    
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of top predictions to return"
    )
    
    language: Optional[str] = Field(
        default="en",
        description="Language code (en, es, fr)"
    )


class Prediction(BaseModel):
    """Single prediction result."""
    
    word: str = Field(..., description="Predicted word")
    confidence: float = Field(..., description="Confidence score")
    index: int = Field(..., description="Word index in vocabulary")


class PredictionResponse(BaseModel):
    """Response model for gesture prediction."""
    
    predictions: List[Prediction] = Field(
        ...,
        description="List of top-k predictions"
    )
    
    trajectory_length: int = Field(
        ...,
        description="Number of points in trajectory"
    )


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Service status")
    message: str = Field(..., description="Status message")
    predictor_loaded: bool = Field(..., description="Whether predictor is loaded")


class ErrorResponse(BaseModel):
    """Error response."""
    
    detail: str = Field(..., description="Error message")
