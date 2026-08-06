"""
Data model for the Image Relevance & Auto-Tagging system.

Tables
------
Image        one row per source image file
ImageTag     structured vision output for an image (1:1 with Image, but kept
             separate so we can re-run classification and keep history)
Post         one row per blog post
Embedding    a vector (stored as JSON floats) for either an image caption or
             a post body, keyed by (owner_type, owner_id) so both share one
             table and one similarity code path
Pairing      a suggested (or approved/rejected) post<->image match, with the
             guard's verdict and explanation attached

Indexes
-------
- ImageTag.image_id      (1:1 lookup, also unique)
- Embedding.owner_type/owner_id composite (lookup + uniqueness)
- Pairing.post_id, Pairing.image_id (ranking queries, dedupe)
"""
from __future__ import annotations

import datetime as dt
import enum
import json
from typing import Optional

from sqlmodel import SQLModel, Field, UniqueConstraint, Index


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class Image(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    path: str = Field(index=True, unique=True)
    created_at: dt.datetime = Field(default_factory=utcnow)


class ClassificationStatus(str, enum.Enum):
    pending = "pending"
    ok = "ok"
    low_confidence = "low_confidence"   # flagged, not guessed
    error = "error"


class ImageTag(SQLModel, table=True):
    """Structured vision output for one image. (M6 - validated schema)"""
    __table_args__ = (
        UniqueConstraint("image_id", name="uq_imagetag_image_id"),
        Index("ix_imagetag_image_id", "image_id"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    image_id: int = Field(foreign_key="image.id")

    subject: str                       # e.g. "red fox"
    category: str                      # e.g. "animal"
    attributes_json: str = Field(default="[]")   # JSON list[str]
    caption: str                       # natural-language description, embedded later
    confidence: float                  # 0..1

    status: ClassificationStatus = Field(default=ClassificationStatus.pending)
    raw_model_output: Optional[str] = None   # audit trail
    created_at: dt.datetime = Field(default_factory=utcnow)

    @property
    def attributes(self) -> list[str]:
        return json.loads(self.attributes_json)

    @attributes.setter
    def attributes(self, value: list[str]) -> None:
        self.attributes_json = json.dumps(value)


class Post(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    title: str
    body: str
    created_at: dt.datetime = Field(default_factory=utcnow)


class OwnerType(str, enum.Enum):
    image_caption = "image_caption"
    post_text = "post_text"


class Embedding(SQLModel, table=True):
    """Shared vector table for both image captions and post text. (M8)"""
    __table_args__ = (
        UniqueConstraint("owner_type", "owner_id", name="uq_embedding_owner"),
        Index("ix_embedding_owner", "owner_type", "owner_id"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    owner_type: OwnerType
    owner_id: int
    vector_json: str            # JSON list[float]
    model_name: str = Field(default="tfidf-svd-v1")
    created_at: dt.datetime = Field(default_factory=utcnow)

    @property
    def vector(self) -> list[float]:
        return json.loads(self.vector_json)

    @vector.setter
    def vector(self, value: list[float]) -> None:
        self.vector_json = json.dumps(value)


class PairingVerdict(str, enum.Enum):
    suggested = "suggested"      # passed the guard, awaiting human review
    approved = "approved"
    rejected = "rejected"        # human rejected
    guard_blocked = "guard_blocked"   # the mismatch guard refused it automatically
    no_match = "no_match"        # nothing cleared the bar for this post


class Pairing(SQLModel, table=True):
    """A post<->image candidate, with the guard's verdict attached. (M8, M3)"""
    __table_args__ = (
        Index("ix_pairing_post_id", "post_id"),
        Index("ix_pairing_image_id", "image_id"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    post_id: int = Field(foreign_key="post.id")
    image_id: Optional[int] = Field(default=None, foreign_key="image.id")  # None => no_match row

    similarity: float = 0.0
    verdict: PairingVerdict = PairingVerdict.suggested
    reason: str = ""             # human-readable explanation, always populated

    reviewed: bool = False
    created_at: dt.datetime = Field(default_factory=utcnow)


class CostLedger(SQLModel, table=True):
    """Per-call cost tracking for vision + embedding calls. (M6)"""
    id: Optional[int] = Field(default=None, primary_key=True)
    call_type: str            # "vision" | "embedding"
    ref_table: str            # "image" | "post"
    ref_id: int
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    created_at: dt.datetime = Field(default_factory=utcnow)
