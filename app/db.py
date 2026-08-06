from sqlmodel import SQLModel, create_engine, Session

DB_URL = "sqlite:///./imgmatch.db"
engine = create_engine(DB_URL, connect_args={"check_same_thread": False})


def init_db() -> None:
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session
