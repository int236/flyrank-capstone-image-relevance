# Image Relevance & Auto-Tagging
A backend AI application that automatically classifies images, generates structured tags, and matches them to the most relevant blog posts using semantic similarity. The system also includes a mismatch guard to prevent incorrect image–post pairings and a review interface for approving or rejecting suggestions.

## Features
* Image classification with a vision model (pluggable Claude / Gemini providers, plus an offline mock provider for testing)
* Structured image tags (subject, category, attributes, caption, confidence)
* Semantic matching between images and blog posts
* Mismatch guard using similarity thresholds and tag validation
* Batch classification pipeline with retries, rate limiting, and per-image failure isolation
* Protected review interface (Approve/Reject)
* Cost tracking for vision model calls

## Tech Stack
* Python
* FastAPI
* SQLModel + SQLite
* Supabase Authentication
* Claude / Gemini Vision API (pluggable providers, with a mock offline provider for testing)
* NumPy & scikit-learn 

## Setup
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file using `.env.example`.
4. Run the application:
   ```bash
   uvicorn app:app --reload
   ```

## API Documentation
Swagger UI:
```
http://127.0.0.1:8000/docs
```

