"""
Embeddings: put image captions and post text int one shared vector space.

Uses SentenceTransformerEmbeddingProvider - small pretrained neural model that maps sentences with
similar meaning close together in vector space. Runs locally.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


class EmbeddingProvider(ABC):
    @abstractmethod
    def fit(self, corpus: list[str]) -> None:
        """Fit the shared space on the full corpus (all captions + all post texts)."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        ...


class TfidfEmbeddingProvider(EmbeddingProvider):
    def __init__(self, n_components: int = 32):
        self._n_components = n_components
        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self._svd: TruncatedSVD | None = None
        self._fitted = False

    def fit(self, corpus: list[str]) -> None:
        if len(corpus) < 2:
            corpus = corpus + ["placeholder document for fitting"]
        tfidf = self._vectorizer.fit_transform(corpus)
        n_comp = min(self._n_components, max(2, min(tfidf.shape) - 1))
        self._svd = TruncatedSVD(n_components=n_comp, random_state=42)
        self._svd.fit(tfidf)
        self._fitted = True

    def embed(self, text: str) -> list[float]:
        if not self._fitted:
            raise RuntimeError("EmbeddingProvider.fit() must be called before embed()")
        tfidf = self._vectorizer.transform([text])
        vec = self._svd.transform(tfidf)[0]
        return vec.tolist()


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """
    A genuinely semantic embedding provider: a small pretrained neural network
    (not a word-overlap trick like TF-IDF) that maps sentences with similar
    *meaning* close together in vector space, even with zero shared vocabulary.
    Runs fully locally/offline after the first download - no API key needed.

    Requires: pip install sentence-transformers
    First call downloads the model (~90MB for all-MiniLM-L6-v2) and caches it
    locally - needs internet once, then works fully offline.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # imported lazily
        self._model = SentenceTransformer(model_name)

    def fit(self, corpus: list[str]) -> None:
        # Pretrained model - no corpus-specific fitting needed, unlike TF-IDF.
        # Kept as a no-op so this class is a drop-in replacement for
        # TfidfEmbeddingProvider behind the same EmbeddingProvider interface.
        pass

    def embed(self, text: str) -> list[float]:
        return self._model.encode(text, convert_to_numpy=True).tolist()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    denom = (np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def upsert_embedding(session, owner_type: str, owner_id: int, vector: list[float],
                      model_name: str = "tfidf-svd-v1") -> None:
    """Persist a computed vector to the Embedding table (M2/M8: 'no missing
    vectors'). Upsert on (owner_type, owner_id) since the shared TF-IDF space
    gets refit as the corpus grows, so vectors are recomputed, not append-only."""
    from sqlmodel import select as _select
    from .models import Embedding

    existing = session.exec(
        _select(Embedding).where(Embedding.owner_type == owner_type,
                                  Embedding.owner_id == owner_id)
    ).first()
    if existing is None:
        row = Embedding(owner_type=owner_type, owner_id=owner_id, model_name=model_name)
        row.vector = vector
        session.add(row)
    else:
        existing.vector = vector
        existing.model_name = model_name
        session.add(existing)
    session.commit()