from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import random
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests # <-- NEW IMPORT
import os # <-- We need this to read the API token securely

app = FastAPI(title="Voyage AI Recommendation Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ... (Keep your UserPreferences and get_recommendations code exactly the same) ...

# ==========================================
# 🕸️ MOCK WEB SCRAPER
# ==========================================
def fetch_recent_reviews(place_name: str):
    mock_review_database = [
        f"Absolutely loved {place_name}! The views were breathtaking.",
        f"It was okay, but way too crowded.",
        f"Terrible experience at {place_name}. Overpriced and dirty.",
        f"A must-visit! Best part of our trip.",
        f"Not worth the hype. Skip it.",
        f"Beautiful architecture and great history.",
        f"Staff was rude, but the place itself is nice."
    ]
    return random.sample(mock_review_database, k=random.randint(3, 5))

# ==========================================
# 🚀 THE NEW VIBE CHECK API ENDPOINT (Using HF API)
# ==========================================
class VibeRequest(BaseModel):
    place_name: str

@app.post("/vibe-check")
def vibe_check(req: VibeRequest):
    reviews = fetch_recent_reviews(req.place_name)
    
    # 1. Prepare the API request to Hugging Face
    API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
    
    # Read the token from Render's environment variables
    hf_token = os.environ.get("HUGGINGFACE_TOKEN")
    headers = {"Authorization": f"Bearer {hf_token}"}
    
    # 2. Call the Hugging Face Inference API
    try:
        response = requests.post(API_URL, headers=headers, json={"inputs": reviews})
        nlp_results = response.json()
        
        # If the API is still warming up, it might return an error, handle it gracefully
        if "error" in nlp_results:
            return {"place": req.place_name, "vibe_score": 50, "vibe_label": "⏳ Model Warming Up", "reviews_analyzed": len(reviews)}

    except Exception as e:
        return {"place": req.place_name, "vibe_score": 50, "vibe_label": "⚠️ API Error", "reviews_analyzed": len(reviews)}

    # 3. Calculate the Vibe Score based on API response
    positive_count = 0
    
    # Hugging Face API returns a list of lists: [[{'label': 'POSITIVE', 'score': 0.99}, ...]]
    for result_list in nlp_results:
        # Get the top prediction for this specific review
        top_prediction = result_list[0] if isinstance(result_list, list) else result_list
        if top_prediction['label'] == 'POSITIVE':
            positive_count += 1
            
    vibe_percentage = (positive_count / len(reviews)) * 100

    if vibe_percentage >= 75:
        vibe_label = "🔥 Immaculate Vibe"
    elif vibe_percentage >= 40:
        vibe_label = "🤔 Mixed Vibe"
    else:
        vibe_label = "🚩 Bad Vibe"

    return {
        "place": req.place_name,
        "vibe_score": round(vibe_percentage),
        "vibe_label": vibe_label,
        "reviews_analyzed": len(reviews)
    }
