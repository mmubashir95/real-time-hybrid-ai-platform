import logging
from collections.abc import AsyncIterator
from functools import lru_cache

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import ConfigurationError, load_settings

logger = logging.getLogger(__name__)


class DatabaseError(RuntimeError):
    pass


@lru_cache
def get_engine() -> AsyncEngine:
    settings = load_settings()
    return create_async_engine(settings.database_url, pool_pre_ping=True)


@lru_cache
def get_session_factory() -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(get_engine(), expire_on_commit=False)


async def get_session() -> AsyncIterator[AsyncSession]:
    try:
        session_factory = get_session_factory()
    except ConfigurationError as error:
        logger.error("Database configuration is invalid: %s", error)
        raise DatabaseError("Database is not configured.") from error

    async with session_factory() as session:
        yield session
