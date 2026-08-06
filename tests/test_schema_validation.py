import pytest
from pydantic import ValidationError

from app.vision import VisionTagSchema, CONFIDENCE_FLOOR, MockVisionProvider, classify_with_retries


def test_valid_payload_parses():
    tag = VisionTagSchema(subject="red fox", category="animal",
                           attributes=["orange fur"], caption="A fox.", confidence=0.9)
    assert tag.subject == "red fox"


def test_confidence_out_of_range_rejected():
    with pytest.raises(ValidationError):
        VisionTagSchema(subject="red fox", category="animal", caption="A fox.", confidence=1.4)


def test_empty_subject_rejected():
    with pytest.raises(ValidationError):
        VisionTagSchema(subject="", category="animal", caption="A fox.", confidence=0.9)


def test_missing_required_field_rejected():
    with pytest.raises(ValidationError):
        VisionTagSchema(category="animal", caption="A fox.", confidence=0.9)  # no subject


def test_low_confidence_is_flagged_not_guessed():
    """A blurry/ambiguous image should come back with confidence below the
    floor, and the caller (batch job) is expected to flag it rather than
    treat the subject as ground truth."""
    provider = MockVisionProvider()
    result, _attempts = classify_with_retries(provider, "images/blurry_canid_05.jpg")
    assert result.tags.confidence < CONFIDENCE_FLOOR
    assert result.tags.subject  # still has a best-guess, just flagged downstream
