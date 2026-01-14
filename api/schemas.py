"""API request and response schemas."""

from pydantic import BaseModel, Field
from typing import List, Tuple, Optional


class GesturePoint(BaseModel):
    """A single point in a gesture sequence."""
    x: float = Field(..., description="X coordinate")
    y: float = Field(..., description="Y coordinate")
    timestamp: Optional[float] = Field(None, description="Timestamp in milliseconds")


class PredictionRequest(BaseModel):
    """Request schema for gesture prediction."""
    gesture: List[GesturePoint] = Field(..., description="List of gesture points")
    language: str = Field("en_US", description="Language code (en_US, es_ES, fr_FR)")
    top_k: int = Field(5, ge=1, le=10, description="Number of predictions to return")


class WordPrediction(BaseModel):
    """A single word prediction with confidence."""
    word: str = Field(..., description="Predicted word")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")


class PredictionResponse(BaseModel):
    """Response schema for gesture prediction."""
    predictions: List[WordPrediction] = Field(..., description="List of predicted words")
    language: str = Field(..., description="Language used for prediction")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Whether model is loaded")


class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
