import threading
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import QueuePool
from xno import settings
import logging


__all__ = ["SqlSession"]

_lock = threading.Lock()


class SqlSession:
    _sessionmakers: dict[str, sessionmaker] = {}

    def __init__(self, db_name: str):
        self.db_name = db_name
        self._session = None

    def __enter__(self):
        if self.db_name not in self._sessionmakers:
            with _lock:
                if self.db_name not in self._sessionmakers:  # double-check
                    logging.debug("Creating new sessionmaker for db=%s", self.db_name)
                    engine = create_engine(
                        settings.postgresql_url(self.db_name),
                        poolclass=QueuePool,
                        echo=False,
                        pool_pre_ping=True,
                    )
                    self._sessionmakers[self.db_name] = sessionmaker(
                        bind=engine,
                        autoflush=False,
                        autocommit=False,
                    )
                    # List all tables in the connected database
                    with engine.connect() as conn:
                        result = conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema='public'"))
                        tables = result.fetchall()
                        print(f"Connected to DB '{self.db_name}'. Available tables: {[table[0] for table in tables]}")
        Session = scoped_session(self._sessionmakers[self.db_name])
        self._session = Session()
        return self._session

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if exc_type is None:
                self._session.commit()
            else:
                self._session.rollback()
        finally:
            self._session.close()
            self._session = None


if __name__ == "__main__":
    print("PostgreSQL session initialized.")
    with SqlSession("xno_execution") as session:
        result = session.execute(text("SELECT * FROM public.strategy_overview LIMIT 10"))
        print(result.fetchall())