from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
from xno import settings

_engine = create_engine(
    settings.postgresql_url,
    poolclass=QueuePool,
    pool_size=4,
    max_overflow=20,
    pool_timeout=30,
    pool_recycle=1_800,
    pool_pre_ping=True,
    echo=False,
)

SqlSession = sessionmaker(
    bind=_engine,
    autoflush=False,
    autocommit=False
)
