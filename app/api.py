"""
Validated API + minimal review surface (M3).

Endpoints
---------
POST /images                 register an already-on-disk image path
POST /images/upload          real file upload (multipart/form-data)
POST /posts                  create a post
POST /batch/classify         run the vision batch job over unclassified images
GET  /posts/{id}/images      ranked candidates for a post (incl. blocked/no_match, for transparency)
POST /pairings/{id}/approve  human approves a suggested pairing
POST /pairings/{id}/reject   human rejects a suggested pairing
GET  /review                 minimal HTML table of all suggested pairings awaiting review
GET  /cost                   running cost total
"""
from __future__ import annotations

import os
import uuid

from dotenv import load_dotenv
load_dotenv()

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sqlmodel import Session, select

from .batch import run_classification_batch
from .cost import total_cost
from .db import get_session, init_db
from .embeddings import EmbeddingProvider
from .matcher import rank_images_for_post
from .models import Image, ImageTag, Pairing, PairingVerdict, Post
from .vision import ClaudeVisionProvider, GeminiVisionProvider, MockVisionProvider

app = FastAPI(title="Image Relevance & Auto-Tagging")


@app.on_event("startup")
def _startup():
    init_db()


# ---- request/response schemas (validated API surface) ----

class ImageIn(BaseModel):
    path: str


class PostIn(BaseModel):
    title: str
    body: str


class CandidateOut(BaseModel):
    image_id: int
    subject: str
    caption: str
    similarity: float
    verdict: str
    reason: str


_embedder_singleton: EmbeddingProvider | None = None


def _get_embedder_instance() -> EmbeddingProvider:
    """Loading a Sentence-Transformers model is relatively expensive (~seconds),
    so we build it once and reuse it across requests, not per-call."""
    global _embedder_singleton
    if _embedder_singleton is not None:
        return _embedder_singleton
    from .embeddings import SentenceTransformerEmbeddingProvider
    _embedder_singleton = SentenceTransformerEmbeddingProvider()
    return _embedder_singleton


def _build_embedder(session: Session) -> EmbeddingProvider:
    """Fit the shared embedding space fresh on current captions + post texts.
    For the Sentence-Transformers provider, fit() is a no-op since it's
    pretrained - kept for interface consistency."""
    embedder = _get_embedder_instance()
    image_texts = [f"{t.subject}. {t.caption} {' '.join(t.attributes)}"
                   for t in session.exec(select(ImageTag)).all() if t.caption]
    posts = [f"{p.title}. {p.body}" for p in session.exec(select(Post)).all()]
    embedder.fit(image_texts + posts)
    return embedder


@app.post("/images", response_model=Image)
def create_image(payload: ImageIn, session: Session = Depends(get_session)):
    """Register an image already sitting on disk at `path`. For actually
    uploading bytes from a client, use POST /images/upload instead."""
    img = Image(path=payload.path)
    session.add(img)
    session.commit()
    session.refresh(img)
    return img


UPLOAD_DIR = "images"


@app.post("/images/upload", response_model=Image)
async def upload_image(file: UploadFile = File(...), session: Session = Depends(get_session)):
    """Real file upload: accepts multipart/form-data, saves the bytes to
    disk under images/, and registers the resulting path."""
    allowed_ext = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in allowed_ext:
        raise HTTPException(400, f"unsupported file type: {ext or 'unknown'}")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    unique_name = f"{uuid.uuid4().hex}{ext}"
    dest_path = os.path.join(UPLOAD_DIR, unique_name)

    contents = await file.read()
    with open(dest_path, "wb") as f:
        f.write(contents)

    img = Image(path=dest_path)
    session.add(img)
    session.commit()
    session.refresh(img)
    return img

@app.post("/posts", response_model=Post)
def create_post(payload: PostIn, session: Session = Depends(get_session)):
    post = Post(title=payload.title, body=payload.body)
    session.add(post)
    session.commit()
    session.refresh(post)
    return post

def _select_vision_provider():
    """Picks the real provider if a key is present, else falls back to the
    mock. Priority: Gemini (free tier) > Claude > Mock. This means going
    live requires setting an env var - no code edits."""
    if os.environ.get("GEMINI_API_KEY"):
        return GeminiVisionProvider()
    if os.environ.get("ANTHROPIC_API_KEY"):
        return ClaudeVisionProvider()
    return MockVisionProvider()


@app.post("/batch/classify")
def batch_classify(force: bool = False, session: Session = Depends(get_session)):
    provider = _select_vision_provider()
    report = run_classification_batch(session, provider, force=force)
    return {
        "processed": report.processed, "ok": report.ok,
        "low_confidence": report.low_confidence, "failed": report.failed,
        "errors": report.errors,
        "provider": type(provider).__name__,
    }


@app.get("/posts/{post_id}/images", response_model=list[CandidateOut])
def images_for_post(post_id: int, session: Session = Depends(get_session)):
    post = session.get(Post, post_id)
    if post is None:
        raise HTTPException(404, "post not found")

    embedder = _build_embedder(session)
    ranked = rank_images_for_post(session, post, embedder)

    # persist suggestions so they show up in the review queue
    for c in ranked:
        existing = session.exec(
            select(Pairing).where(Pairing.post_id == post_id, Pairing.image_id == c.image_id)
        ).first()
        verdict_map = {
            "suggested": PairingVerdict.suggested,
            "guard_blocked": PairingVerdict.guard_blocked,
            "no_match": PairingVerdict.no_match,
        }
        if existing is None:
            session.add(Pairing(post_id=post_id, image_id=c.image_id, similarity=c.similarity,
                                 verdict=verdict_map[c.verdict], reason=c.reason))
        else:
            existing.similarity, existing.verdict, existing.reason = (
                c.similarity, verdict_map[c.verdict], c.reason)
            session.add(existing)
    session.commit()

    return [CandidateOut(image_id=c.image_id, subject=c.subject, caption=c.caption,
                          similarity=c.similarity, verdict=c.verdict, reason=c.reason)
            for c in ranked]


@app.post("/pairings/{pairing_id}/approve", response_model=Pairing)
def approve_pairing(pairing_id: int, session: Session = Depends(get_session)):
    p = session.get(Pairing, pairing_id)
    if p is None:
        raise HTTPException(404, "pairing not found")
    if p.reviewed:
      raise HTTPException(409, "pairing already reviewed")
    if p.verdict == PairingVerdict.guard_blocked:
        raise HTTPException(409, f"cannot approve a guard-blocked pairing: {p.reason}")
    p.verdict = PairingVerdict.approved
    p.reviewed = True
    session.add(p)
    session.commit()
    session.refresh(p)
    return p


@app.post("/pairings/{pairing_id}/reject", response_model=Pairing)
def reject_pairing(pairing_id: int, session: Session = Depends(get_session)):
    p = session.get(Pairing, pairing_id)
    if p is None:
        raise HTTPException(404, "pairing not found")
    if p.reviewed:
      raise HTTPException(409, "pairing already reviewed")
    p.verdict = PairingVerdict.rejected
    p.reviewed = True
    session.add(p)
    session.commit()
    session.refresh(p)
    return p


@app.get("/cost")
def cost(session: Session = Depends(get_session)):
    return {"total_estimated_usd": round(total_cost(session), 6)}


@app.get("/review", response_class=HTMLResponse)
def review_table(session: Session = Depends(get_session)):
    pairings = session.exec(select(Pairing).order_by(Pairing.created_at.desc())).all()
    rows = ""
    for p in pairings:
        post = session.get(Post, p.post_id)
        img = session.get(Image, p.image_id) if p.image_id else None
        color = {"suggested": "#e8f7ee", "guard_blocked": "#fdeaea",
                 "no_match": "#f2f2f2", "approved": "#d9f2e6", "rejected": "#f5d9d9"}.get(p.verdict.value, "#fff")
        rows += f"""<tr style="background:{color}">
          <td>{p.id}</td><td>{post.title if post else '?'}</td>
          <td>{img.path if img else '-'}</td>
          <td>{p.similarity:.3f}</td><td>{p.verdict.value}</td><td>{p.reason}</td>
          <td>
            <form style="display:inline" method="post" action="/pairings/{p.id}/approve"><button>Approve</button></form>
            <form style="display:inline" method="post" action="/pairings/{p.id}/reject"><button>Reject</button></form>
          </td>
        </tr>"""
    return f"""<html><head><title>Review Queue</title>
    <style>body{{font-family:sans-serif}} table{{border-collapse:collapse;width:100%}}
    td,th{{border:1px solid #ccc;padding:6px;text-align:left;font-size:14px}}</style></head>
    <body><h2>Pairing Review Queue</h2>
    <table><tr><th>ID</th><th>Post</th><th>Image</th><th>Similarity</th><th>Verdict</th><th>Reason</th><th>Action</th></tr>
    {rows}</table></body></html>"""
