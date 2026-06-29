from collections.abc import AsyncGenerator

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from billing_anomaly_detector.domain.ports import (
    AnomalyRepository,
    EmbeddingPort,
    ExplanationPort,
    InvoiceRepository,
)
from billing_anomaly_detector.infrastructure.config import Settings
from billing_anomaly_detector.infrastructure.persistence.anomaly_repository import (
    SqlAlchemyAnomalyRepository,
)
from billing_anomaly_detector.infrastructure.persistence.invoice_repository import (
    SqlAlchemyInvoiceRepository,
)


def get_settings(request: Request) -> Settings:
    """Returns the Settings singleton stored on app.state at startup."""
    return request.app.state.settings  # type: ignore[no-any-return]


async def get_session(
    request: Request,
) -> AsyncGenerator[AsyncSession, None]:
    """
    Yields one AsyncSession per request.
    Commits on success, rolls back on exception, always closes.
    FastAPI calls this for every route that depends on it.
    """
    async with request.app.state.session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def get_invoice_repo(
    session: AsyncSession = Depends(get_session),
) -> InvoiceRepository:
    """Creates a per-request InvoiceRepository backed by the session."""
    return SqlAlchemyInvoiceRepository(session)


def get_anomaly_repo(
    session: AsyncSession = Depends(get_session),
) -> AnomalyRepository:
    """Creates a per-request AnomalyRepository backed by the session."""
    return SqlAlchemyAnomalyRepository(session)


def get_embedding_adapter(request: Request) -> EmbeddingPort:
    """Returns the singleton embedding adapter from app.state."""
    return request.app.state.embedding_adapter  # type: ignore[no-any-return]


def get_explanation_chain(request: Request) -> ExplanationPort:
    """Returns the singleton explanation chain from app.state."""
    return request.app.state.explanation_chain  # type: ignore[no-any-return]
