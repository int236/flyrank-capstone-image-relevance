# Image Relevance & Auto-Tagging
A backend AI application that automatically classifies images, generates structured tags, and matches them to the most relevant blog posts using semantic similarity. The system also includes a mismatch guard to prevent incorrect image–post pairings and a review interface for approving or rejecting suggestions.

## Features
* Image classification with a vision model (pluggable Claude / Gemini providers, plus an offline mock provider for testing)
* Structured image tags (subject, category, attributes, caption, confidence)
* Semantic matching between images and blog posts (TF-IDF + SVD by default, with a pluggable sentence-transformer provider for dense semantic embeddings)
* Mismatch guard: similarity threshold + explicit subject-agreement check (extracts the post's target subject and verifies image tags actually agree with it, not just generic keyword overlap)
* Batch classification pipeline with retries, rate limiting, and per-image failure isolation
* Real multipart image upload endpoint plus path-based registration for local seeding
* Review interface (Approve/Reject) — ⚠️ verify Supabase auth is actually applied to these routes before calling this "protected"
* Cost tracking for vision and embedding calls

## Tech Stack
* Python
* FastAPI
* SQLModel + SQLite
* Supabase Authentication
* Claude / Gemini Vision API (pluggable providers, with a mock offline provider for testing)
* NumPy & scikit-learn (TF-IDF + TruncatedSVD embeddings, with a pluggable `sentence-transformers` provider for dense semantic embeddings)

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

