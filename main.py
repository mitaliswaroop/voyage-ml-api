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

# ==========================================
# 🕸️ MOCK WEB SCRAPER (Upgraded Dataset)
# ==========================================
def fetch_recent_reviews(place_name: str):
    mock_review_database = [
        # --- Positives (🔥) ---
        f"Absolutely loved {place_name}! The views were breathtaking.",
        f"A must-visit! Best part of our entire trip.",
        f"Beautiful architecture and incredible history. Highly recommend.",
        f"The food around {place_name} was amazing. 10/10 experience.",
        f"I could spend all day here. So peaceful and picturesque.",
        f"{place_name} exceeded all our expectations. Truly magical.",
        f"Very clean, safe, and the locals were incredibly welcoming.",
        f"Perfect for photos! We went at sunset and it was gorgeous.",
        f"Got there early to beat the crowds, and it was absolutely worth it.",
        f"Such a vibrant atmosphere! Will definitely be coming back.",
        f"A hidden gem. I'm so glad we added {place_name} to our itinerary.",
        f"Great value for money. We spent hours exploring.",
        f"The tour guides at {place_name} were so knowledgeable and friendly.",
        f"Incredible energy. You can feel the history of the place.",
        f"One of the most beautiful places I have ever seen in my life.",

        # --- Mixed / Neutral (🤔) ---
        f"It was okay, but way too crowded for my liking.",
        f"Staff was a bit rude, but the place itself is nice.",
        f"Beautiful to look at, but everything nearby is a tourist trap.",
        f"Glad I saw it, but I probably wouldn't go out of my way to visit again.",
        f"The weather was terrible when we went, which ruined the vibe of {place_name}.",
        f"It's exactly what you expect. Nothing more, nothing less.",
        f"Nice place, but the entrance fee is a bit steep for what you get.",
        f"Good for a quick 30-minute stop, but don't plan your whole day around it.",
        f"Finding parking near {place_name} was a nightmare, but the visit was decent.",
        f"Beautiful, but undergoing a lot of construction right now.",
        f"A bit overhyped on social media, but still a pleasant afternoon.",
        f"Food nearby was average, but the scenery made up for it.",

        # --- Negatives (🚩) ---
        f"Terrible experience at {place_name}. Overpriced and extremely dirty.",
        f"Not worth the hype. Skip it and go somewhere else.",
        f"An absolute tourist trap. Everything is a scam here.",
        f"Way too loud and crowded. We left after 10 minutes.",
        f"Very disappointing. Looks nothing like the pictures online.",
        f"Felt unsafe walking around {place_name} at night.",
        f"The wait times were ridiculous. Not worth standing in line for 2 hours.",
        f"Rude locals and terrible service everywhere we went.",
        f"Save your money. Total waste of time.",
        f"{place_name} is completely ruined by mass tourism.",
        f"It smelled awful and there was trash everywhere.",
        f"Customer service was non-existent. Completely disorganized.",
        f"We got food poisoning from a cafe right next to {place_name}.",
        f"Boring. There is literally nothing to do here.",
        f"I regret adding {place_name} to our trip. Huge letdown."
    ]
    
    # Analyze a larger batch (10 to 15 reviews) to get a more accurate percentage
    # (If the database happens to be smaller than the request, we cap it safely)
    sample_size = min(len(mock_review_database), random.randint(10, 15))
    return random.sample(mock_review_database, k=sample_size)

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
        
        # Check if Hugging Face sent back an error
        if isinstance(nlp_results, dict) and "error" in nlp_results:
            actual_error = nlp_results["error"]
            
            # If it's just sleeping, tell the user to wait
            if "loading" in actual_error.lower() or "estimated_time" in nlp_results:
                return {"place": req.place_name, "vibe_score": 50, "vibe_label": "⏳ Waking up AI... Click again in 20s", "reviews_analyzed": len(reviews)}
            
            # If it's a token issue, print the REAL error to the screen!
            return {"place": req.place_name, "vibe_score": 50, "vibe_label": f"⚠️ Error: {actual_error}", "reviews_analyzed": len(reviews)}

    except Exception as e:
        return {"place": req.place_name, "vibe_score": 50, "vibe_label": "⚠️ Server Error", "reviews_analyzed": len(reviews)}

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
