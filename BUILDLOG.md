# BUILDLOG.md

## AI Usage Log

### Where AI helped
- Explained FastAPI, Pydantic, SQLModel, and Supabase Auth concepts.
- Helped clarify TF-IDF embeddings, cosine similarity, and semantic matching.
- Suggested improvements for project structure and README formatting.
- Reviewed code for potential bugs and edge cases.
- Helped write documentation (`README.md`, `EVIDENCE.md`, `.env.example`, `.gitignore`).

### Where AI was incorrect or incomplete
- Initially assumed the project used embedding API calls for cost tracking, while the implementation used local TF-IDF embeddings.
- Suggested a `seed` command before verifying that `seed.py` was executable.
- Some suggestions required adapting to the actual project structure (e.g., `api.py` instead of `main.py`).

### Changes I made
- Verified every AI suggestion before applying it.
- Adjusted commands and paths to match my project structure.
- Kept TF-IDF as the embedding implementation instead of switching to an API-based embedding model.
- Added project-specific documentation and evidence rather than using generic templates.

### What I implemented independently
- Image classification pipeline.
- Semantic image matching and relevance scoring.
- Mismatch guard and rejection logic.
- Cost tracking integration.
- Evaluation pipeline and automated tests.
- Overall debugging and final project integration.