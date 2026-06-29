"""
Integration test fixtures using testcontainers.
Requires Docker running locally (Colima or Docker Desktop).
"""

import asyncio
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from billing_anomaly_detector.infrastructure.persistence.models import Base


@pytest.fixture(scope="session")
def postgres_url():
    """
    Provides a real Postgres URL for integration tests.
    Uses testcontainers to spin up a fresh pgvector instance.
    Requires Docker running locally.
    """
    try:
        from testcontainers.postgres import PostgresContainer
        with PostgresContainer("pgvector/pgvector:pg16") as pg:
            # testcontainers gives a sync URL — convert to asyncpg format
            url = pg.get_connection_url().replace(
                "postgresql+psycopg2://", "postgresql+asyncpg://"
            )
            yield url
    except Exception:
        pytest.skip("Docker not available — skipping integration tests")


@pytest.fixture(scope="session")
def event_loop():
    """Session-scoped event loop so all integration tests share one loop."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def db_engine(postgres_url):
    engine = create_async_engine(postgres_url, echo=False)
    async with engine.connect() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.commit()
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
async def db_session(db_engine) -> AsyncSession:
    """Per-test session — rolls back after each test to keep tests isolated."""
    factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        yield session
        await session.rollback()
