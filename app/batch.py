"""
Batch classification job (M5).
Modeled on the "batch-existence-check" pattern: iterate a queue of items,
call the slow/bulk vision endpoint per item, retry transient failures,
record cost per call, and never let one bad image kill the whole run.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from .cost import record_cost
from .models import Image, ImageTag, ClassificationStatus
from .vision import (VisionProvider, VisionCallError, CONFIDENCE_FLOOR,
                      MockVisionProvider, classify_with_retries)

log = logging.getLogger("imgmatch.batch")
RATE_LIMIT_SLEEP_SECONDS = 13

@dataclass
class BatchReport:
    processed: int = 0
    ok: int = 0
    low_confidence: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)

def run_classification_batch(session: Session, provider: VisionProvider,
                              image_ids: list[int] | None = None,
                              max_attempts: int = 3,
                              force: bool = False) -> BatchReport:
    """
    Classifies every Image that doesn't yet have an ImageTag (or the subset given by image_ids). 
    If force=True, deletes any existing ImageTag first so images get reclassified (e.g. switching 
    from mock to a real provider). Writes ImageTag rows + CostLedger rows. Never raises on a 
    single-image failure; that image is recorded as `error` and the job continues.
    """
    report = BatchReport()
    is_rate_limited_provider = not isinstance(provider, MockVisionProvider)

    if force:
        existing_tags = session.exec(select(ImageTag)).all()
        for t in existing_tags:
            if image_ids is None or t.image_id in image_ids:
                session.delete(t)
        session.commit()

    query = select(Image)
    if image_ids is not None:
        query = query.where(Image.id.in_(image_ids))
    images = session.exec(query).all()

    for img in images:
        existing = session.exec(select(ImageTag).where(ImageTag.image_id == img.id)).first()
        if existing is not None and existing.status != ClassificationStatus.error:
            continue
        if existing is not None:
            session.delete(existing)   # retry a previously-failed image
            session.commit()

        report.processed += 1
        if is_rate_limited_provider:
            time.sleep(RATE_LIMIT_SLEEP_SECONDS)
        try:
            result, attempts = classify_with_retries(provider, img.path, max_attempts=max_attempts)
            status = (ClassificationStatus.ok if result.tags.confidence >= CONFIDENCE_FLOOR
                      else ClassificationStatus.low_confidence)
            tag = ImageTag(
                image_id=img.id,
                subject=result.tags.subject,
                category=result.tags.category,
                caption=result.tags.caption,
                confidence=result.tags.confidence,
                status=status,
                raw_model_output=result.raw_output,
            )
            tag.attributes = result.tags.attributes
            session.add(tag)
            try:
                session.commit()
            except IntegrityError:
                # a concurrent run already tagged this image - not an error
                session.rollback()
                log.warning("image %s already tagged by a concurrent run, skipping", img.id)
                report.processed -= 1
                continue

            record_cost(session, "vision", "image", img.id,
                        result.input_tokens, result.output_tokens)

            if status == ClassificationStatus.low_confidence:
                report.low_confidence += 1
                log.warning("image %s flagged low_confidence (%.2f) after %d attempt(s)",
                            img.id, result.tags.confidence, attempts)
            else:
                report.ok += 1

        except VisionCallError as e:
            report.failed += 1
            report.errors.append(f"image {img.id} ({img.path}): {e}")
            tag = ImageTag(
                image_id=img.id, subject="", category="", caption="",
                confidence=0.0, status=ClassificationStatus.error,
                raw_model_output=str(e),
            )
            session.add(tag)
            try:
                session.commit()
            except IntegrityError:
                session.rollback()
                log.warning("image %s already tagged by a concurrent run, skipping", img.id)
            log.error("image %s permanently failed: %s", img.id, e)

    return report