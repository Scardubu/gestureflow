"""FastAPI backend for GestureFlow inference."""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict
import torch
import json
from pathlib import Path
import numpy as np
import sys
import time

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))  # FIXED: **file** -> __file__

from src.models.lstm_model import create_model
from src.config import CHECKPOINTS_DIR, MODEL_CONFIG, INFERENCE_CONFIG

# Initialize FastAPI app
app = FastAPI(
    title="SwipePredict API",
    description="LSTM-based swipe typing prediction API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # FIXED: Restricted from "*" for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for loaded models
loaded_models: Dict[str, Dict] = {}

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

class Prediction(BaseModel):
    """Single prediction result."""
    word: str
    confidence: float
    rank: int

class PredictionResponse(BaseModel):
    """Response with predictions."""
    predictions: List[Prediction]
    inference_time_ms: float
    num_points: int

def load_model(language: str = "en") -> Dict:
    """
    Load model and vocabulary for a language.

    Returns:
        Dictionary with model, vocabulary, and metadata
    """
    if language in loaded_models:
        return loaded_models[language]

    model_dir = CHECKPOINTS_DIR / language

    # Check if model exists
    model_path = model_dir / "best_model.pt"
    vocab_path = model_dir / "vocabulary.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found for language '{language}'. "
            f"Train it first: python scripts/train_model.py --language {language}"
        )

    # Load vocabulary
    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)

    word_to_idx = vocab_data['word_to_idx']
    idx_to_word = {int(k): v for k, v in vocab_data['idx_to_word'].items()}
    vocab_size = vocab_data['vocab_size']

    # Create and load model
    model = create_model(vocab_size=vocab_size, model_type="lstm")

    # Load weights
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Store in cache
    loaded_models[language] = {
        'model': model,
        'word_to_idx': word_to_idx,
        'idx_to_word': idx_to_word,
        'vocab_size': vocab_size
    }

    print(f"Loaded {language} model: {vocab_size} words")

    return loaded_models[language]

def preprocess_trajectory(
    trajectory: List[GesturePoint],
    max_length: int = 50
) -> torch.Tensor:
    """
    Preprocess gesture trajectory for model input.

    Args:
        trajectory: List of gesture points
        max_length: Maximum sequence length
        
    Returns:
        Tensor of shape (1, max_length, 3)
    """
    # Extract coordinates and timestamps
    points = []
    for point in trajectory:
        points.append([point.x, point.y, point.timestamp])

    features = np.array(points)

    # Normalize timestamps
    if len(features) > 1:
        timestamps = features[:, 2]
        timestamps = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min() + 1e-8)
        features[:, 2] = timestamps

    # Pad or truncate
    if len(features) < max_length:
        padding = np.zeros((max_length - len(features), 3))
        features = np.vstack([features, padding])
    else:
        features = features[:max_length]

    # Convert to tensor
    tensor = torch.FloatTensor(features).unsqueeze(0)  # Add batch dimension

    return tensor

@app.on_event("startup")
async def startup_event():
    """Load default model on startup."""
    try:
        load_model("en")
        print("Default English model loaded successfully")
    except Exception as e:
        print(f"Warning: Could not load default model: {e}")

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "SwipePredict API",
        "version": "1.0.0",
        "status": "online",
        "loaded_models": list(loaded_models.keys())
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "loaded_models": len(loaded_models)
    }

@app.get("/languages")
async def get_languages():
    """Get list of available languages."""
    available = []
    for lang_code in ["en", "es", "fr"]:
        model_path = CHECKPOINTS_DIR / lang_code / "best_model.pt"
        if model_path.exists():
            available.append(lang_code)

    return {
        "available": available,
        "loaded": list(loaded_models.keys())
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict word from swipe gesture.

    Args:
        request: Prediction request with trajectory
        
    Returns:
        Top-k word predictions with confidence scores
    """
    start_time = time.time()

    try:
        # Load model if not already loaded
        model_data = load_model(request.language)
        model = model_data['model']
        idx_to_word = model_data['idx_to_word']
        
        # Preprocess trajectory
        input_tensor = preprocess_trajectory(
            request.trajectory,
            max_length=MODEL_CONFIG["sequence_length"]
        )
        
        # Make prediction
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, k=request.top_k, dim=-1)
        
        # Format predictions
        predictions = []
        for rank, (idx, prob) in enumerate(zip(
            top_indices[0].tolist(),
            top_probs[0].tolist()
        ), 1):
            word = idx_to_word.get(idx, "<UNK>")  # COMPLETED: Added closing quote and <UNK> for unknown
            predictions.append(Prediction(word=word, confidence=prob, rank=rank))
        
        inference_time_ms = (time.time() - start_time) * 1000
        
        return PredictionResponse(
            predictions=predictions,
            inference_time_ms=inference_time_ms,
            num_points=len(request.trajectory)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))