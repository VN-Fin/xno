from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
from xno import settings
import logging


__all__ = ["SqlSession"]

engine = create_engine(
    settings.postgresql_url,
    poolclass=QueuePool,
    pool_size=4,
    max_overflow=20,
    pool_timeout=30,
    pool_recycle=1_800,
    pool_pre_ping=True,
)
SqlSession = sessionmaker(bind=engine, autoflush=False, autocommit=False)
with engine.connect() as conn:
    conn.execute(text("SELECT 1"))
logging.info("PostgreSQL connection test successful.")


if __name__ == "__main__":
    print("PostgreSQL session initialized.")
    # Example usage
    if SqlSession is None:
        raise RuntimeError("PostgreSQL session is not initialized")

    with SqlSession.begin() as session:
        result = session.execute(text("SELECT 1"))
        print(result.scalar())

    # Or
    with SqlSession() as session:
        result = session.execute(text("SELECT 1"))
        print(result.scalar())