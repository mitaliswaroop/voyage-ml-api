from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import random
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests 
import os 

app = FastAPI(title="Voyage AI ML Backend")

# CRITICAL: Allow your React app to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 🌍 PART 1: DESTINATION RECOMMENDER (Cosine Similarity)
# ==========================================

# 1. Define the Input Data Structure from React
class UserPreferences(BaseModel):
    nature: float      
    history: float
    nightlife: float
    relaxation: float
    adventure: float

# 2. Mock Destination Database (Scores on a scale of 1 to 5)
destinations_data = { "city": ["Tokyo", "Paris", "New York", "London", "Bangkok", "Dubai", "Singapore", "Istanbul", "Seoul", "Barcelona",
                 "Prague", "Vienna", "Amsterdam", "Milan", "Taipei", "Berlin", "Madrid", "Athens", "Kyoto", "Rome",
                 "Bali", "Phuket", "Swiss Alps", "Patagonia", "Queenstown", "Banff", "Iceland", "Santorini", "Maui", "Fiji"],
        "nature": [2, 2, 1, 2, 3, 1, 3, 2, 2, 3, 2, 2, 3, 1, 4, 3, 2, 2, 4, 1, 5, 5, 5, 5, 5, 5, 5, 4, 5, 5],
        "history": [4, 5, 3, 5, 4, 3, 2, 5, 4, 4, 5, 5, 4, 4, 3, 4, 4, 5, 5, 5, 2, 1, 1, 1, 1, 1, 1, 4, 1, 1],
        "nightlife": [5, 4, 5, 5, 5, 4, 4, 3, 5, 5, 4, 3, 5, 4, 4, 5, 5, 3, 1, 3, 4, 5, 1, 1, 3, 1, 1, 3, 2, 2],
        "relaxation": [3, 3, 1, 2, 4, 4, 3, 3, 2, 4, 3, 4, 3, 3, 3, 2, 3, 3, 5, 2, 5, 5, 4, 3, 2, 3, 4, 5, 5, 5],
        "adventure": [2, 1, 2, 2, 4, 3, 2, 2, 3, 2, 1, 1, 2, 1, 4, 2, 2, 2, 2, 1, 4, 4, 5, 5, 5, 5, 5, 2, 4, 4]
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


# ==========================================
# 🔮 PART 2: NLP VIBE CHECKER (Hugging Face API)
# ==========================================

# 1. Mock Web Scraper
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

# 2. Define Input Structure
class VibeRequest(BaseModel):
    place_name: str

# 3. The Vibe Check Endpoint
@app.post("/vibe-check")
def vibe_check(req: VibeRequest):
    reviews = fetch_recent_reviews(req.place_name)
    
    # Prepare the API request to Hugging Face
    API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
    
    # Read the token from Render's environment variables
    hf_token = os.environ.get("HUGGINGFACE_TOKEN")
    headers = {"Authorization": f"Bearer {hf_token}"}
    
    # Call the Hugging Face Inference API
    try:
        response = requests.post(API_URL, headers=headers, json={"inputs": reviews})
        nlp_results = response.json()
        
        # If the API is still warming up, it might return an error, handle it gracefully
        if isinstance(nlp_results, dict) and "error" in nlp_results:
            return {"place": req.place_name, "vibe_score": 50, "vibe_label": "⏳ Model Warming Up", "reviews_analyzed": len(reviews)}

    except Exception as e:
        return {"place": req.place_name, "vibe_score": 50, "vibe_label": "⚠️ API Error", "reviews_analyzed": len(reviews)}

    # Calculate the Vibe Score based on API response
    positive_count = 0
    
    # Hugging Face API returns a list of lists: [[{'label': 'POSITIVE', 'score': 0.99}, ...]]
    if isinstance(nlp_results, list):
        for result_list in nlp_results:
            # Get the top prediction for this specific review
            top_prediction = result_list[0] if isinstance(result_list, list) else result_list
            if top_prediction.get('label') == 'POSITIVE':
                positive_count += 1
                
    vibe_percentage = (positive_count / len(reviews)) * 100 if reviews else 0

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
