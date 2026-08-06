# Build Log — Image Relevance & Auto-Tagging System

## Day 1 — Project Scaffolding & Planning

Started by defining the core problem: given a pool of images and a pool of blog posts, automatically figure out which images belong to which posts — without manual tagging.

Broke the system into four stages:
1. **Vision classification** — extract subject, category, attributes, caption from each image.
2. **Embedding & matching** — compare image tags/captions against blog post content.
3. **Mismatch guard** — reject low-confidence or semantically inconsistent pairings.
4. **Review UI** — human-in-the-loop approve/reject before anything goes live.

Set up the FastAPI skeleton:
```bash
mkdir image-relevance-tagging && cd image-relevance-tagging
python -m venv venv && source venv/bin/activate
pip install fastapi uvicorn sqlmodel python-dotenv
```

Created the base `app.py`, `.env.example`, and `requirements.txt`. Decided on SQLModel + SQLite for simplicity during dev, with a clear path to Postgres later if needed.

## Day 2 — Database Schema

Modeled the core entities:

- `Image` — id, file path/URL, uploaded_at, status
- `ImageTag` — image_id, subject, category, attributes (JSON), caption, confidence, status
- `Post` — id, title, body
- `Embedding` — owner_type, owner_id, vector, model_name (shared table for both post and image vectors)
- `MatchSuggestion` / candidate ranking — image_id, post_id, similarity_score, verdict
- Cost ledger — provider, kind, owner_type, owner_id, input_tokens, output_tokens

Used SQLModel so the same classes double as Pydantic schemas for the API and ORM models for the DB — cut down on duplication significantly.

Wrote initial migrations and a seed script to populate a handful of dummy blog posts for testing.

## Day 3 — Vision Classification Pipeline

Built `vision.py` around a `VisionProvider` abstract interface with a strict output contract: every classification must validate against `VisionTagSchema` (subject, category, attributes, caption, confidence 0.0–1.0). Anything that doesn't validate is a hard error, not a soft guess.

Three providers implement the interface, fully swappable:
- **`MockVisionProvider`** — deterministic, offline, keyword-matches filenames against canned answers. Used for tests/demo, including one deliberately low-confidence case ("blurry") to exercise the flag path without needing real API calls.
- **`ClaudeVisionProvider`** — real call via `anthropic.messages.create`, image sent as base64 alongside a strict JSON-only schema prompt.
- **`GeminiVisionProvider`** — real call via Gemini's `generate_content`, using `response_mime_type: application/json` to force valid JSON back with no markdown fences.

`classify_with_retries()` wraps any provider in an immediate retry loop (no backoff/sleep — just re-attempt up to `max_attempts`), raising `VisionCallError` only after all attempts are exhausted.

Confidence gating: `CONFIDENCE_FLOOR = 0.6`. Anything below that gets `status=low_confidence` — the tag is still stored, but the matcher (Day 5) knows never to treat it as a "known good" match without flagging it for review first.

## Day 4 — Embedding & Semantic Matching Engine

Built the embedding layer around a shared `EmbeddingProvider` abstract interface so the matching logic never has to care which model is underneath — just `fit()` and `embed()`.

**Default: `TfidfEmbeddingProvider`.** Not raw TF-IDF cosine similarity — captions and post text get vectorized with `TfidfVectorizer` (English stop words, unigrams+bigrams), then reduced through `TruncatedSVD` down to 32 components. That extra SVD step turns sparse word-overlap vectors into a smaller dense latent space, which handles synonymy a bit better than raw TF-IDF alone.

**Also implemented: `SentenceTransformerEmbeddingProvider`.** A real pretrained neural embedding model (`all-MiniLM-L6-v2`), lazily imported so it's not a hard dependency. Same interface, drop-in replacement — `fit()` is a no-op since the model's pretrained and doesn't need corpus fitting. This isn't a "someday" swap, it's already wired in behind the same abstraction; picking it is a one-line change.

Matching pipeline (`matcher.py`):
1. Embed the post's `title + body`.
2. For each image tag, embed `subject + caption + attributes` (deliberately fuller than just the caption — short captions alone are too sparse for TF-IDF to find lexical overlap with post prose).
3. Compute cosine similarity between post vector and each image vector.
4. Persist every computed vector via `upsert_embedding()` into an `Embedding` table, keyed on `(owner_type, owner_id)` — upserted rather than appended, since the shared TF-IDF/SVD space gets refit as the corpus grows and old vectors go stale.
5. Log cost per embedding call via `record_cost()` — same cost-tracking path used for vision calls.

One gotcha worth logging: the similarity threshold (`0.08`) is tuned specifically for TF-IDF's compressed score range. Left a comment in code as a flag for future-me: if we ever switch to the sentence-transformer provider, dense semantic vectors score much higher on loosely related text, so the threshold needs re-tuning to somewhere around 0.35–0.45, not reused as-is.

## Day 5 — Mismatch Guard

The guard is two independent checks, and both have to pass before a suggestion gets auto-approved.

**Check 1 — similarity threshold.** Below the cutoff → straight to `no_match`, never surfaced.

**Check 2 — subject agreement (the real guard).** This is where it gets more deliberate than a simple keyword check:
- `extract_target_subject()` scans the post's title+body and tries to find which known image-subject vocabulary the post is actually about — a subject only counts as a match if *all* of its key tokens appear literally in the post text, and the most specific (longest) match wins ties.
- `subjects_agree()` then checks whether the image's tagged subject shares any token with that target subject (with generic qualifiers like "gray," "wild," "domestic" stripped out first, so "gray wolf" and "wolf" key off the same core noun).
- If similarity clears the bar but subjects disagree → `guard_blocked`, with an explicit reason string (e.g. *"post is about 'fox', image is tagged 'wolf' — refused"*).

This exists because similarity alone isn't reliable — a wolf-in-snow photo and a fox-in-snow post can share enough embedding-space vocabulary ("winter," "forest," "predator") to score deceptively high. The subject check is a hard, explainable veto on top of the soft similarity ranking.

There's also a third path: **low-confidence tags don't get auto-blocked.** If a tag's status is `low_confidence`, it still passes as `"suggested"` (not rejected), but the reason string flags it explicitly for human review rather than silent auto-approval. So the review queue isn't just "things the guard couldn't decide" — it also deliberately includes "the guard would approve this, but confidence was shaky."

`best_pairing_for_post()` walks the ranked candidates and returns the first one that actually made it to `"suggested"` — or `None` if nothing did, covering the "no good image for this post" case explicitly rather than silently returning a low-quality match.

## Day 6 — Batch Classification, Review Interface & Auth

Built `batch.py` — `run_classification_batch()` iterates every `Image` missing an `ImageTag` (or a given `image_ids` subset), classifies each via `classify_with_retries`, and never lets one failure kill the run: failed images get an `ImageTag` with `status=error` instead of crashing the batch.

Details worth logging:
- Rate limiting: hardcoded 13s sleep between calls when using a real (non-mock) vision provider.
- `force=True` mode wipes existing tags and reclassifies — built for provider migration (e.g. mock → real vision model).
- Concurrent-run safety: if two batch jobs somehow tag the same image, the resulting `IntegrityError` is caught and treated as a no-op skip, not a crash.

Built the review endpoint set:
- `GET /suggestions?status=pending`
- `POST /suggestions/{id}/approve`
- `POST /suggestions/{id}/reject`

Wired up Supabase Authentication so only logged-in reviewers can hit these endpoints — added a dependency that validates the Supabase JWT on protected routes.

Manually tested the full loop: register images → batch classify → rank against a post → review queue → approve/reject → status updates in DB.

**Note:** image *registration* currently happens via a plain client script (`load_data.py`) calling `POST /images` and `POST /posts` one at a time — there's no bulk/multi-file ingestion endpoint yet. That's a real gap, tracked in the backlog below (see correction).

## Day 7 — Cost Tracking & Polish

Cost is logged via `record_cost()` for both vision classification calls and embedding calls — one shared ledger, not two separate systems.

Cleaned up `.env.example`, wrote setup instructions, and generated Swagger docs via FastAPI's built-in OpenAPI support at `/docs`.

Final smoke test: ran the full pipeline against a small sample of images and posts. The mismatch guard correctly caught false-positive matches that similarity scoring alone would have auto-approved.

---

## Next Steps (Backlog)

- Swap `TfidfEmbeddingProvider` for `SentenceTransformerEmbeddingProvider` in production and re-tune the similarity threshold (~0.35–0.45) accordingly — the code path already exists, this is a config change plus threshold retuning, not new development.
- **Add a real batch upload endpoint for bulk image ingestion** (`POST /images/batch`, multipart or JSON list) — still open. `batch.py` only handles bulk *classification* of already-registered images; registration itself is still one-at-a-time via a local script (`load_data.py`), not a documented, auth-protected API endpoint.
- Add pagination + filtering to the review interface.
- Add webhook/notification when a new match needs review.
