"""
FastAPI main application.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import numpy as np
from typing import List

from .schemas import GestureRequest, PredictionResponse, HealthResponse
from src.inference.predictor import GesturePredictor
from src.config import CHECKPOINT_DIR, PROCESSED_DATA_DIR

# Initialize FastAPI app
app = FastAPI(
    title="GestureFlow API",
    description="API for gesture-based text prediction",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance
predictor = None


@app.on_event("startup")
async def startup_event():
    """Initialize predictor on startup."""
    global predictor
    
    model_path = CHECKPOINT_DIR / "best_model.h5"
    vocab_path = PROCESSED_DATA_DIR / "vocab.json"
    
    if model_path.exists() and vocab_path.exists():
        try:
            predictor = GesturePredictor(
                model_path=model_path,
                vocab_path=vocab_path
            )
            print(f"Predictor loaded successfully!")
        except Exception as e:
            print(f"Warning: Failed to load predictor: {e}")
            predictor = None
    else:
        print(f"Warning: Model or vocabulary not found. API will run in limited mode.")


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint."""
    return {
        "status": "healthy",
        "message": "GestureFlow API is running",
        "predictor_loaded": predictor is not None
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy" if predictor is not None else "degraded",
        "message": "Predictor loaded" if predictor is not None else "Predictor not loaded",
        "predictor_loaded": predictor is not None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: GestureRequest):
    """
    Predict word from gesture trajectory.
    
    Args:
        request: Gesture request with trajectory and timestamps
        
    Returns:
        Prediction response with top-k predictions
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Predictor not loaded. Please ensure model and vocabulary files exist."
        )
    
    try:
        # Convert input to numpy arrays
        trajectory = np.array(request.trajectory)
        timestamps = np.array(request.timestamps)
        
        # Validate input shapes
        if trajectory.shape[1] != 2:
            raise HTTPException(
                status_code=400,
                detail="Trajectory must have shape (N, 2) for x, y coordinates"
            )
        
        if len(timestamps) != len(trajectory):
            raise HTTPException(
                status_code=400,
                detail="Timestamps must have same length as trajectory"
            )
        
        # Predict
        predictions = predictor.predict(
            trajectory=trajectory,
            timestamps=timestamps,
            top_k=request.top_k
        )
        
        return {
            "predictions": predictions,
            "trajectory_length": len(trajectory)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict_batch")
async def predict_batch(requests: List[GestureRequest]):
    """
    Predict words from multiple gesture trajectories.
    
    Args:
        requests: List of gesture requests
        
    Returns:
        List of prediction responses
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Predictor not loaded"
        )
    
    try:
        results = []
        for request in requests:
            trajectory = np.array(request.trajectory)
            timestamps = np.array(request.timestamps)
            
            predictions = predictor.predict(
                trajectory=trajectory,
                timestamps=timestamps,
                top_k=request.top_k
            )
            
            results.append({
                "predictions": predictions,
                "trajectory_length": len(trajectory)
            })
        
        return results
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
