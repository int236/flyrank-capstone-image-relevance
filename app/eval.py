"""
Small labeled eval (M10): for each post, is the top-1 *suggested* (guard-passed)
image actually the correct species? Reuses the same corpus as the demo/tests.
"""
from __future__ import annotations

from sqlmodel import Session, SQLModel, create_engine, select

from .embeddings import TfidfEmbeddingProvider
from .matcher import best_pairing_for_post
from .models import ImageTag, Post
from .seed import seed_corpus, POSTS
from .vision import MockVisionProvider
from .batch import run_classification_batch

def run_eval() -> dict:
    # Dedicated in-memory DB, independent of app.db's module-level engine and
    # the process's working directory 
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        seed_corpus(session)
        run_classification_batch(session, MockVisionProvider())
        image_texts = [f"{t.subject}. {t.caption} {' '.join(t.attributes)}"
                       for t in session.exec(select(ImageTag)).all()]
        posts_text = [f"{p.title}. {p.body}" for p in session.exec(select(Post)).all()]
        embedder = TfidfEmbeddingProvider()
        embedder.fit(image_texts + posts_text)

        correct = 0
        total = 0
        details = []
        for post_row, expected_subject in POSTS:
            post = session.exec(select(Post).where(Post.title == post_row["title"])).first()
            if post is None:
                raise RuntimeError(
                    f"seed_corpus did not create a post titled {post_row['title']!r} - "
                    f"seed.py and eval.py are out of sync"
                )
            top = best_pairing_for_post(session, post, embedder)
            total += 1
            got_subject = top.subject if top else None
            if expected_subject is None:
                is_correct = got_subject is None   # correctly abstained
            else:
                is_correct = (got_subject is not None
                              and expected_subject.lower() in got_subject.lower())
            correct += int(is_correct)
            details.append({
                "post": post.title, "expected": expected_subject,
                "got": got_subject, "correct": is_correct,
            })

        precision = correct / total if total else 0.0
        return {"top1_precision": precision, "correct": correct, "total": total, "details": details}

if __name__ == "__main__":
    import json
    print(json.dumps(run_eval(), indent=2))