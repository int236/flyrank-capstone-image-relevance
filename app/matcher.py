"""
Matching + mismatch guard (M8) - the decision core of the whole system.

Two independent signals feed the guard, exactly as specified:
  1. similarity below threshold  -> reject / no_match
  2. tags disagree with what the post is actually about -> guard_blocked

Signal 2 exists *because* signal 1 alone isn't enough: a wolf-in-snow photo
and a fox-in-snow post can share plenty of embedding-space vocabulary
("winter", "forest", "predator") and still score a deceptively high
similarity. The tag check is a hard, explainable veto on top of the soft
similarity ranking - it's what stops "close enough" from becoming "wrong".
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from sqlmodel import Session, select

from .cost import record_cost
from .embeddings import EmbeddingProvider, cosine_similarity, upsert_embedding
from .models import ClassificationStatus, Image, ImageTag, OwnerType, Post

SIMILARITY_THRESHOLD = 0.5

@dataclass
class Candidate:
    image_id: int
    subject: str
    caption: str
    similarity: float
    tag_status: ClassificationStatus
    verdict: str          # "suggested" | "guard_blocked" | "no_match"
    reason: str


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower()))

def _subject_tokens(subject: str) -> set[str]:
    # drop generic qualifiers so "gray wolf" and "wolf" and "domestic dog" and "dog" all key
    # off their most specific noun(s)
    stop = {"domestic", "gray", "grey", "wild", "young", "adult", "unknown"}
    return {t for t in _tokenize(subject) if t not in stop}

def extract_target_subject(post_text: str, known_subjects: list[str]) -> str | None:
    """Find which known tag vocabulary the post is actually about, by literal
    mention count. Longest/most-specific match wins on ties."""
    text_tokens = _tokenize(post_text)
    best: tuple[int, str] | None = None
    for subj in known_subjects:
        subj_tokens = _subject_tokens(subj)
        if not subj_tokens:
            continue
        overlap = len(subj_tokens & text_tokens)
        if overlap == len(subj_tokens) and overlap > 0:  # all of the subject's key tokens present
            score = len(subj_tokens)
            if best is None or score > best[0]:
                best = (score, subj)
    return best[1] if best else None


def subjects_agree(image_subject: str, target_subject: str) -> bool:
    a, b = _subject_tokens(image_subject), _subject_tokens(target_subject)
    return bool(a & b)


def rank_images_for_post(session: Session, post: Post, embedder: EmbeddingProvider,
                          threshold: float = SIMILARITY_THRESHOLD) -> list[Candidate]:
    tags = session.exec(
        select(ImageTag).where(ImageTag.status != ClassificationStatus.error)
    ).all()
    if not tags:
        return []

    known_subjects = sorted({t.subject for t in tags if t.status == ClassificationStatus.ok})
    target_subject = extract_target_subject(f"{post.title} {post.body}", known_subjects)

    post_text = f"{post.title}. {post.body}"
    post_vec = embedder.embed(post_text)
    upsert_embedding(session, OwnerType.post_text.value, post.id, post_vec)
    record_cost(session, "embedding", "post", post.id,
                input_tokens=len(post_text.split()), output_tokens=0)

    candidates: list[Candidate] = []
    for tag in tags:
        img = session.get(Image, tag.image_id)
        if img is None:
            continue
        # Embed the fuller structured description (subject + caption + attributes),
        # not just the caption sentence - short captions alone are too sparse for
        # TF-IDF to find lexical overlap with post prose (e.g. "domestic dog" needs
        # to appear, not just be implied by "porch").
        image_text = f"{tag.subject}. {tag.caption} {' '.join(tag.attributes)}"
        image_vec = embedder.embed(image_text)
        upsert_embedding(session, OwnerType.image_caption.value, img.id, image_vec)
        record_cost(session, "embedding", "image", img.id,
                    input_tokens=len(image_text.split()), output_tokens=0)
        sim = cosine_similarity(post_vec, image_vec)

        if sim < threshold:
            verdict, reason = "no_match", f"similarity {sim:.3f} below threshold {threshold}"
        elif tag.status == ClassificationStatus.low_confidence:
            verdict = "suggested"
            reason = (f"passes similarity ({sim:.3f}) but tag confidence is low "
                      f"({tag.confidence:.2f}) - flagged for human review, not auto-approved")
        elif target_subject is not None and not subjects_agree(tag.subject, target_subject):
            verdict = "guard_blocked"
            reason = (f"similarity {sim:.3f} clears the bar, but tags disagree: "
                      f"post is about '{target_subject}', image is tagged '{tag.subject}' - refused")
        else:
            verdict = "suggested"
            reason = f"similarity {sim:.3f}, tag '{tag.subject}' agrees with post subject"

        candidates.append(Candidate(
            image_id=img.id, subject=tag.subject, caption=tag.caption,
            similarity=sim, tag_status=tag.status, verdict=verdict, reason=reason,
        ))

    candidates.sort(key=lambda c: c.similarity, reverse=True)
    return candidates

def best_pairing_for_post(session: Session, post: Post, embedder: EmbeddingProvider,
                           threshold: float = SIMILARITY_THRESHOLD) -> Candidate | None:
    """Top candidate that actually cleared the guard, or None if nothing did
    (the 'no good image for this post' case)."""
    ranked = rank_images_for_post(session, post, embedder, threshold)
    for c in ranked:
        if c.verdict == "suggested":
            return c
    return None
