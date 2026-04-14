import os
# Force Hugging Face to use our local folder BEFORE it even boots up
os.environ["HF_HOME"] = "./hf_cache"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
import random
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="Voyage AI Recommendation Engine")

# CRITICAL: Allow your React app to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # We will restrict this to your Vercel link later for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 🧠 LOAD THE ML MODEL (Do this OUTSIDE the route)
# ==========================================
print("Loading NLP Model... (This takes a few seconds)")
# We use distilbert for fast sentiment analysis
sentiment_analyzer = pipeline(
    "sentiment-analysis", 
    model="distilbert-base-uncased-finetuned-sst-2-english",
    cache_dir="./hf_cache" 
)
print("Model Loaded!")

# ==========================================
# 🕸️ MOCK WEB SCRAPER
# ==========================================
def fetch_recent_reviews(place_name: str):
    """
    In a production app, you would replace this function with a call 
    to the Google Places API to fetch real reviews.
    """
    # A mix of generic good, bad, and okay reviews to test our model
    mock_review_database = [
        f"Absolutely loved {place_name}! The views were breathtaking.",
        f"It was okay, but way too crowded.",
        f"Terrible experience at {place_name}. Overpriced and dirty.",
        f"A must-visit! Best part of our trip.",
        f"Not worth the hype. Skip it.",
        f"Beautiful architecture and great history.",
        f"Staff was rude, but the place itself is nice."
    ]
    # Randomly select 3 to 5 reviews to simulate a scrape
    return random.sample(mock_review_database, k=random.randint(3, 5))


# ==========================================
# 🚀 THE VIBE CHECK API ENDPOINT
# ==========================================
class VibeRequest(BaseModel):
    place_name: str

@app.post("/vibe-check")
def vibe_check(req: VibeRequest):
    # 1. "Scrape" the reviews
    reviews = fetch_recent_reviews(req.place_name)
    
    # 2. Pass reviews through the HuggingFace BERT model
    nlp_results = sentiment_analyzer(reviews)
    
    # 3. Calculate the "Vibe Score" (% of positive reviews)
    positive_count = 0
    total_score_confidence = 0
    
    for result in nlp_results:
        # result looks like: {'label': 'POSITIVE', 'score': 0.99}
        if result['label'] == 'POSITIVE':
            positive_count += 1
        total_score_confidence += result['score']
    
    # Math to get a clean percentage (e.g., 85%)
    vibe_percentage = (positive_count / len(reviews)) * 100
    avg_confidence = total_score_confidence / len(reviews)

    # Determine a vibe label
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

# 1. Define the Input Data Structure from React
class UserPreferences(BaseModel):
    nature: float      # e.g., 1 to 5 scale
    history: float
    nightlife: float
    relaxation: float
    adventure: float

# 2. Our Mock Destination Database (You can move this to a CSV later!)
# Scores are on a scale of 1 to 5.
destinations_data = {
    "city": ["Kyoto, Japan", "Ibiza, Spain", "Swiss Alps", "Rome, Italy", "Bali, Indonesia", "Las Vegas, USA"],
    "nature":     [4.0, 2.0, 5.0, 1.0, 4.5, 1.0],
    "history":    [5.0, 1.0, 1.0, 5.0, 2.0, 1.0],
    "nightlife":  [1.0, 5.0, 1.0, 3.0, 4.0, 5.0],
    "relaxation": [4.0, 2.0, 3.0, 2.0, 5.0, 2.0],
    "adventure":  [2.0, 3.0, 5.0, 1.0, 3.0, 4.0]
}
df = pd.DataFrame(destinations_data)
features = ["nature", "history", "nightlife", "relaxation", "adventure"]

# 3. The ML Recommendation Endpoint
@app.post("/recommend")
def get_recommendations(prefs: UserPreferences):
    # Convert user input into a NumPy array
    user_vector = np.array([[
        prefs.nature, 
        prefs.history, 
        prefs.nightlife, 
        prefs.relaxation, 
        prefs.adventure
    ]])
    
    # Extract destination vectors
    city_vectors = df[features].values
    
    # Calculate Cosine Similarity
    similarities = cosine_similarity(user_vector, city_vectors)[0]
    
    # Add scores to the dataframe and sort by best match
    df['match_score'] = similarities
    top_matches = df.sort_values(by="match_score", ascending=False).head(3)
    
    # Format the output to send back to React
    results = []
    for _, row in top_matches.iterrows():
        results.append({
            "destination": row["city"],
            "match_percentage": round(row["match_score"] * 100, 1)
        })
        
    return {"recommendations": results}
