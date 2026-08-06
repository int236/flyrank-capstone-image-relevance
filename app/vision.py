"""
Vision tagging: image -> validated structured tags.

Design:
- VisionTagSchema is the *only* shape a classification can take. Anything the
  model returns gets parsed through this; if it doesn't validate, that's a
  hard error (retried), not a soft guess.
- confidence < CONFIDENCE_FLOOR => status=low_confidence. We store the tag
  but never let it silently participate as a "known good" match later
  (the matcher checks status before using a tag - see matcher.py).
- Three providers: MockVisionProvider (deterministic, offline, for tests/demo),
  ClaudeVisionProvider (real anthropic.messages.create call), and
  GeminiVisionProvider (real Gemini generate_content call, free-tier friendly).
  All swappable via the same interface so the rest of the system never knows
  which is live.
"""
from __future__ import annotations

import base64
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, Field, ValidationError, field_validator

CONFIDENCE_FLOOR = 0.6

VISION_JSON_SCHEMA_PROMPT = """You are a strict image-tagging classifier.
Return ONLY a JSON object (no prose, no markdown fences) with exactly these keys:
{
  "subject": string,            // canonical common name, e.g. "red fox" not "vulpes vulpes"
  "category": string,           // broad category, e.g. "animal"
  "attributes": [string, ...],  // visual attributes, e.g. ["orange fur", "bushy tail"]
  "caption": string,            // one sentence natural-language description
  "confidence": number          // 0.0-1.0, your true confidence in `subject`
}
If you are not confident, still fill in your best guess for `subject` but set
confidence low (< 0.5) rather than fabricating certainty."""


class VisionTagSchema(BaseModel):
    subject: str
    category: str
    attributes: list[str] = Field(default_factory=list)
    caption: str
    confidence: float

    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"confidence out of range: {v}")
        return v

    @field_validator("subject", "category", "caption")
    @classmethod
    def non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("field must not be empty")
        return v


@dataclass
class VisionResult:
    tags: VisionTagSchema
    raw_output: str
    input_tokens: int
    output_tokens: int


class VisionCallError(Exception):
    pass


class VisionProvider(ABC):
    @abstractmethod
    def classify(self, image_path: str) -> VisionResult:
        ...


class MockVisionProvider(VisionProvider):
    """
    Deterministic offline provider for tests/demo. Looks up a canned answer
    by filename keyword so the same fox.jpg always tags the same way -
    including synonym normalization ("vulpes_vulpes.jpg" -> subject "red fox")
    and one deliberately low-confidence case to exercise the flag path.
    """

    _CANNED = {
        "fox": VisionTagSchema(
            subject="red fox", category="animal",
            attributes=["orange fur", "bushy tail", "pointed ears"],
            caption="A red fox standing alert in tall grass.",
            confidence=0.93,
        ),
        "vulpes": VisionTagSchema(  # paraphrase / latin-name case
            subject="red fox", category="animal",
            attributes=["orange fur", "bushy tail"],
            caption="Vulpes vulpes, the red fox, captured mid-stride.",
            confidence=0.9,
        ),
        "wolf": VisionTagSchema(
            subject="gray wolf", category="animal",
            attributes=["gray fur", "yellow eyes", "large paws"],
            caption="A gray wolf walking through snow.",
            confidence=0.95,
        ),
        "dog": VisionTagSchema(
            subject="domestic dog", category="animal",
            attributes=["brown fur", "collar"],
            caption="A pet dog sitting on a porch.",
            confidence=0.9,
        ),
        "cat": VisionTagSchema(
            subject="domestic cat", category="animal",
            attributes=["gray fur", "green eyes"],
            caption="A cat curled up on a windowsill.",
            confidence=0.92,
        ),
        "blurry": VisionTagSchema(   # low-confidence exercise case
            subject="unknown canid", category="animal",
            attributes=["blurry", "partially obscured"],
            caption="An indistinct animal, possibly a fox or small dog.",
            confidence=0.35,
        ),
    }

    def classify(self, image_path: str) -> VisionResult:
        name = os.path.basename(image_path).lower()
        match = next((v for k, v in self._CANNED.items() if k in name), None)
        if match is None:
            match = VisionTagSchema(
                subject="unknown subject", category="unknown",
                attributes=[], caption=f"Unrecognized image: {name}",
                confidence=0.3,
            )
        raw = match.model_dump_json()
        return VisionResult(tags=match, raw_output=raw, input_tokens=350, output_tokens=90)


class ClaudeVisionProvider(VisionProvider):
    """Real provider. Requires ANTHROPIC_API_KEY in env and the `anthropic` package."""

    def __init__(self, model: str = "claude-sonnet-4-6", api_key: Optional[str] = None):
        import anthropic  # imported lazily so mock-only usage never needs the package
        self._client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self._model = model

    @staticmethod
    def _media_type(path: str) -> str:
        ext = path.lower().rsplit(".", 1)[-1]
        return {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
                "webp": "image/webp", "gif": "image/gif"}.get(ext, "image/jpeg")

    def classify(self, image_path: str) -> VisionResult:
        with open(image_path, "rb") as f:
            b64 = base64.standard_b64encode(f.read()).decode("utf-8")

        resp = self._client.messages.create(
            model=self._model,
            max_tokens=500,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64",
                                                  "media_type": self._media_type(image_path),
                                                  "data": b64}},
                    {"type": "text", "text": VISION_JSON_SCHEMA_PROMPT},
                ],
            }],
        )
        text_block = next((b.text for b in resp.content if b.type == "text"), "")
        try:
            parsed = json.loads(text_block)
            tags = VisionTagSchema(**parsed)
        except (json.JSONDecodeError, ValidationError) as e:
            raise VisionCallError(f"schema validation failed: {e}\nraw: {text_block[:500]}")

        usage = getattr(resp, "usage", None)
        return VisionResult(
            tags=tags, raw_output=text_block,
            input_tokens=getattr(usage, "input_tokens", 0),
            output_tokens=getattr(usage, "output_tokens", 0),
        )


class GeminiVisionProvider(VisionProvider):
    """Real provider using Google's Gemini API (has a free tier).
    Requires GEMINI_API_KEY in env and the `google-genai` package
    (`pip install google-genai`). Get a key at https://aistudio.google.com/apikey
    """

    def __init__(self, model: str = "gemini-flash-lite-latest", api_key: Optional[str] = None):
        from google import genai  # imported lazily, mirrors ClaudeVisionProvider
        self._client = genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY"))
        self._model = model

    @staticmethod
    def _media_type(path: str) -> str:
        ext = path.lower().rsplit(".", 1)[-1]
        return {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
                "webp": "image/webp", "gif": "image/gif"}.get(ext, "image/jpeg")

    def classify(self, image_path: str) -> VisionResult:
        from google.genai import types

        with open(image_path, "rb") as f:
            image_bytes = f.read()

        resp = self._client.models.generate_content(
            model=self._model,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=self._media_type(image_path)),
                VISION_JSON_SCHEMA_PROMPT,
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",  # forces valid JSON back, no markdown fences
            ),
        )

        text_block = resp.text or ""
        try:
            parsed = json.loads(text_block)
            tags = VisionTagSchema(**parsed)
        except (json.JSONDecodeError, ValidationError) as e:
            raise VisionCallError(f"schema validation failed: {e}\nraw: {text_block[:500]}")

        usage = getattr(resp, "usage_metadata", None)
        return VisionResult(
            tags=tags, raw_output=text_block,
            input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
        )


def classify_with_retries(provider: VisionProvider, image_path: str, max_attempts: int = 3):
    """Returns (VisionResult, attempts_used). Raises VisionCallError after exhausting retries."""
    last_err: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            return provider.classify(image_path), attempt
        except Exception as e:  # noqa: BLE001 - deliberately broad, this is the retry boundary
            last_err = e
            continue
    raise VisionCallError(f"vision classification failed after {max_attempts} attempts: {last_err}")
