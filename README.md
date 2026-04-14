# voyage-ml-api
# 🌍 Voyage AI - Full-Stack Smart Travel Planner

**Voyage AI** is a highly advanced, full-stack travel planning application. It leverages Large Language Models (LLMs) for generative planning, traditional Machine Learning for personalized recommendations, and Natural Language Processing (NLP) for real-time sentiment analysis of tourist destinations.

Check out the live web app here: **[Insert Your Vercel Link Here]** *Backend ML API Repository:* **[Insert Your Backend GitHub Repo Link Here]**

---

## ✨ Core AI Features

### 🤖 Generative AI Itinerary Planning (Google Gemini)
* Dynamically prompts the `gemini-2.5-flash` model to generate highly customized, day-by-day itineraries based on user constraints (destination, duration, total group budget, and interests).
* Enforces strict JSON schema validation for reliable, automated data hydration into the React UI.

### 🧠 Content-Based Recommendation Engine (Scikit-Learn)
* Solves the "Cold Start" problem by using **Cosine Similarity** and matrix math. 
* Users rate their preferred travel vibes (Nature, History, Nightlife, etc.) on a 1-5 scale. The backend compares these preferences against a multi-dimensional database of global destinations to calculate a precise "Match Percentage."

### 🔮 Real-Time NLP "Vibe Check" (Hugging Face & PyTorch)
* Integrates a pre-trained Deep Learning model (`distilbert-base-uncased-finetuned-sst-2-english`) directly into the Python backend.
* Performs live **Sentiment Analysis** on destination reviews to give users a data-driven "Vibe Score" (e.g., 95% Positive) before they visit a location.

---

## 🛠️ Tech Stack & Architecture

This project utilizes a decoupled microservice architecture:

**Frontend (Client UI):**
* **Framework:** React.js + Vite
* **Styling:** Tailwind CSS / Custom UI Components
* **Document Generation:** `jspdf` for one-click itinerary downloads
* **Hosting:** Vercel

**Backend (Machine Learning API):**
* **Framework:** Python + FastAPI
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn
* **Deep Learning/NLP:** PyTorch, Hugging Face Transformers
* **Hosting:** Render

---

## 🚀 Getting Started (Local Development)

To run this full-stack application on your local machine, you will need to run both the frontend and backend servers simultaneously.

### 1. Setup the Python ML Backend
Clone the backend repository and install the dependencies:
```bash
git clone [Insert Your Backend Repo Link]
cd voyage-ml-api
pip install -r requirements.txt
