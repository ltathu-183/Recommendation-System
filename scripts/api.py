"""
Simple FastAPI server for recommendation inference.

Usage:
    uv run uvicorn scripts.api:app --reload
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
from pathlib import Path
from rec_sys.model import TwoStageLGBMRanker
import pandas as pd

app = FastAPI(title="RecSys API", description="Fashion Recommendation API")

# Load model on startup
model_path = Path("outputs/model.pkl")
if model_path.exists():
    with open(model_path, "rb") as f:
        payload = pickle.load(f)
    model = payload["lgbm_model"]
    cfg = payload["cfg"]
    print("Model loaded successfully")
else:
    model = None
    cfg = None
    print("Model not found, run training first")

class RecommendationRequest(BaseModel):
    customer_id: str
    n: int = 12

@app.post("/recommend")
def recommend(request: RecommendationRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # For demo, return dummy recommendations
    # In real implementation, would need user history, etc.
    recommendations = [f"article_{i}" for i in range(request.n)]

    return {
        "customer_id": request.customer_id,
        "recommendations": recommendations
    }

@app.get("/")
def root():
    return {"message": "RecSys API is running"}