from typing import Union, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
from xno import settings
import logging


SqlSession: Optional[sessionmaker] = None


def init_postgresql():
    global SqlSession
    if SqlSession is None:
        engine = create_engine(
            settings.postgresql_url,
            poolclass=QueuePool,
            pool_size=4,
            max_overflow=20,
            pool_timeout=30,
            pool_recycle=1_800,
            pool_pre_ping=True,
            echo=False,
        )
        logging.info("Connecting to PSQL")
        SqlSession = sessionmaker(
            bind=engine,
            autoflush=False,
            autocommit=False
        )


if __name__ == "__main__":
    init_postgresql()
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