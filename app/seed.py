"""
Seed corpus. Filenames double as the MockVisionProvider's lookup key (see
vision.py) - swap this for ~50 real image paths and ClaudeVisionProvider to
go live; nothing downstream changes.
"""
from __future__ import annotations

from sqlmodel import Session, select

from .models import Image, Post

CORPUS = [
    "images/fox_01.jpg", "images/fox_02.jpg", "images/fox_03.jpg",
    "images/vulpes_vulpes_04.jpg",          # latin-name paraphrase case
    "images/wolf_01.jpg", "images/wolf_02.jpg",
    "images/dog_01.jpg", "images/dog_02.jpg",
    "images/cat_01.jpg", "images/cat_02.jpg",
    "images/blurry_canid_05.jpg",           # deliberately low-confidence
]

# (post fields, expected top-1 subject substring; None = "no good image expected")
POSTS: list[tuple[dict, str | None]] = [
    ({"title": "Meet the Red Fox",
      "body": "The red fox is one of the most widespread wild canids, known for its "
              "orange coat and bushy tail, often seen at dusk near forest edges."},
     "red fox"),
    ({"title": "The Gray Wolf's Comeback",
      "body": "Once hunted to near extinction, the gray wolf has returned to parts of "
              "its historic range, moving in packs through snowy forest terrain."},
     "gray wolf"),
    ({"title": "Why Dogs Make Great Companions",
      "body": "The domestic dog has lived alongside humans for thousands of years, "
              "prized as a loyal pet and working companion."},
     "domestic dog"),
    ({"title": "Understanding Your Cat's Behavior",
      "body": "The domestic cat is an independent yet affectionate pet, known for "
              "curling up in warm spots and hunting small toys indoors."},
     "domestic cat"),
    ({"title": "The Secret Life of the Snow Leopard",
      "body": "The snow leopard is an elusive big cat living in high-altitude mountains "
              "of Central Asia, rarely photographed in the wild."},
     None),  # no snow leopard image in the corpus -> should be "no good match"
]


def seed_corpus(session: Session) -> None:
    if session.exec(select(Image)).first() is not None:
        return  # already seeded
    for path in CORPUS:
        session.add(Image(path=path))
    for fields, _expected in POSTS:
        session.add(Post(**fields))
    session.commit()

if __name__ == "__main__":
    from sqlmodel import Session
    from .db import engine 

    with Session(engine) as session:
        seed_corpus(session)
        print("Database seeded successfully.")