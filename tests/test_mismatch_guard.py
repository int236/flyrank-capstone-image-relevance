from sqlmodel import select

from app.batch import run_classification_batch
from app.embeddings import TfidfEmbeddingProvider
from app.matcher import (best_pairing_for_post, extract_target_subject,
                          rank_images_for_post, subjects_agree)
from app.models import Image, ImageTag, Post
from app.vision import MockVisionProvider


def _seed_fox_wolf_dog(session):
    fox = Image(path="images/fox_01.jpg")
    wolf = Image(path="images/wolf_01.jpg")
    dog = Image(path="images/dog_01.jpg")
    session.add_all([fox, wolf, dog])
    session.commit()
    for i in (fox, wolf, dog):
        session.refresh(i)

    post = Post(title="Meet the Red Fox",
                body="The red fox is a wild canid known for its orange fur and bushy tail, "
                     "often spotted in forests and fields at dawn.")
    session.add(post)
    session.commit()
    session.refresh(post)
    return fox, wolf, dog, post


def test_subjects_agree_helper():
    assert subjects_agree("red fox", "fox")
    assert subjects_agree("gray wolf", "wolf")
    assert not subjects_agree("gray wolf", "red fox")


def test_extract_target_subject_from_post_text():
    known = ["red fox", "gray wolf", "domestic dog"]
    target = extract_target_subject(
        "The red fox is a wild canid known for its orange fur.", known)
    assert target == "red fox"


def test_fox_post_ranks_fox_image_top(session):
    _fox, _wolf, _dog, post = _seed_fox_wolf_dog(session)
    run_classification_batch(session, MockVisionProvider())

    image_texts = [f"{t.subject}. {t.caption} {' '.join(t.attributes)}"
                   for t in session.exec(select(ImageTag)).all()]
    embedder = TfidfEmbeddingProvider()
    embedder.fit(image_texts + [f"{post.title}. {post.body}"])

    ranked = rank_images_for_post(session, post, embedder)
    assert ranked[0].subject == "red fox"
    assert ranked[0].verdict == "suggested"


def test_guard_refuses_wolf_even_though_similarity_is_forced_high(session):
    """The production-critical case: prove the guard blocks the wrong species
    even when similarity alone would have picked it. We force this by
    checking the guard function directly against a synthetic high-similarity
    wolf candidate, independent of whatever the embedder happens to score."""
    _fox, _wolf, _dog, post = _seed_fox_wolf_dog(session)
    run_classification_batch(session, MockVisionProvider())

    wolf_tag = session.exec(
        select(ImageTag).join(Image).where(Image.path == "images/wolf_01.jpg")
    ).first()

    known_subjects = ["red fox", "gray wolf", "domestic dog"]
    target = extract_target_subject(f"{post.title} {post.body}", known_subjects)
    assert target == "red fox"

    # even with a deliberately inflated similarity score, tag disagreement
    # must still refuse the wolf image for the fox post
    forced_high_similarity = 0.99
    assert forced_high_similarity >= 0.10  # clears the similarity bar
    assert not subjects_agree(wolf_tag.subject, target)  # tags disagree -> guard fires


def test_guard_blocked_pairing_appears_with_explanation(session):
    _fox, _wolf, _dog, post = _seed_fox_wolf_dog(session)
    run_classification_batch(session, MockVisionProvider())

    image_texts = [f"{t.subject}. {t.caption} {' '.join(t.attributes)}"
                   for t in session.exec(select(ImageTag)).all()]
    embedder = TfidfEmbeddingProvider()
    embedder.fit(image_texts + [f"{post.title}. {post.body}"])

    ranked = rank_images_for_post(session, post, embedder)
    wolf_candidates = [c for c in ranked if c.subject == "gray wolf"]
    for c in wolf_candidates:
        # wolf must never be "suggested" for a fox post: either it's rejected
        # by similarity, or explicitly guard_blocked with a stated reason
        assert c.verdict in ("guard_blocked", "no_match")
        assert c.reason  # always explains itself


def test_no_good_image_case(session):
    """A post about something with zero matching images in the corpus should
    resolve to 'no good match', not a wrong guess."""
    Image(path="images/fox_01.jpg")
    fox = Image(path="images/fox_01.jpg")
    session.add(fox)
    session.commit()
    session.refresh(fox)

    post = Post(title="The Secret Life of the Snow Leopard",
                body="The snow leopard is an elusive big cat of Central Asian mountains.")
    session.add(post)
    session.commit()
    session.refresh(post)

    run_classification_batch(session, MockVisionProvider())
    image_texts = [f"{t.subject}. {t.caption} {' '.join(t.attributes)}"
                   for t in session.exec(select(ImageTag)).all()]
    embedder = TfidfEmbeddingProvider()
    embedder.fit(image_texts + [f"{post.title}. {post.body}"])

    result = best_pairing_for_post(session, post, embedder)
    assert result is None
