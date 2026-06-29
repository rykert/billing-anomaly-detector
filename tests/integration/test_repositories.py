"""
Integration tests for SQLAlchemy repositories against a real Postgres.
Each test runs in a transaction that's rolled back afterward — no test
data persists between tests.
"""

import uuid
from datetime import date
from decimal import Decimal

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from billing_anomaly_detector.domain.value_objects import AnomalyScore, ClaimCode, MemberId, Money
from billing_anomaly_detector.domain.entities import Invoice
from billing_anomaly_detector.infrastructure.persistence.invoice_repository import (
    SqlAlchemyInvoiceRepository,
)
from billing_anomaly_detector.infrastructure.persistence.anomaly_repository import (
    SqlAlchemyAnomalyRepository,
)

from tests.conftest import make_invoice, make_detection_result


@pytest.mark.integration
class TestInvoiceRepository:
    async def test_add_and_get(self, db_session: AsyncSession) -> None:
        repo = SqlAlchemyInvoiceRepository(db_session)
        invoice = make_invoice()

        await repo.add(invoice)
        await db_session.flush()

        retrieved = await repo.get(invoice.id)
        assert retrieved is not None
        assert retrieved.id == invoice.id
        assert retrieved.member_id.value == "MBR123456"

    async def test_list_unembedded(self, db_session: AsyncSession) -> None:
        repo = SqlAlchemyInvoiceRepository(db_session)
        invoice = make_invoice(embedding=None)
        await repo.add(invoice)
        await db_session.flush()

        unembedded = await repo.list_unembedded(limit=10)
        ids = [i.id for i in unembedded]
        assert invoice.id in ids

    async def test_update_embedding(self, db_session: AsyncSession) -> None:
        repo = SqlAlchemyInvoiceRepository(db_session)
        invoice = make_invoice()
        await repo.add(invoice)
        await db_session.flush()

        embedding = [0.1] * 1536
        await repo.update_embedding(invoice.id, embedding)
        await db_session.flush()

        embedded = await repo.list_all_embeddings()
        ids = [row[0] for row in embedded]
        assert invoice.id in ids

    async def test_get_returns_none_for_missing(self, db_session: AsyncSession) -> None:
        repo = SqlAlchemyInvoiceRepository(db_session)
        result = await repo.get(uuid.uuid4())
        assert result is None


@pytest.mark.integration
class TestAnomalyRepository:
    async def test_add_and_get_by_invoice(self, db_session: AsyncSession) -> None:
        invoice_repo = SqlAlchemyInvoiceRepository(db_session)
        anomaly_repo = SqlAlchemyAnomalyRepository(db_session)

        invoice = make_invoice()
        await invoice_repo.add(invoice)
        await db_session.flush()

        result = make_detection_result(invoice_id=invoice.id, score=0.85)
        await anomaly_repo.add(result)
        await db_session.flush()

        retrieved = await anomaly_repo.get_by_invoice(invoice.id)
        assert retrieved is not None
        assert retrieved.score.value == pytest.approx(0.85, abs=1e-6)

    async def test_list_above_threshold(self, db_session: AsyncSession) -> None:
        invoice_repo = SqlAlchemyInvoiceRepository(db_session)
        anomaly_repo = SqlAlchemyAnomalyRepository(db_session)

        high_invoice = make_invoice()
        low_invoice = make_invoice()
        await invoice_repo.add(high_invoice)
        await invoice_repo.add(low_invoice)
        await db_session.flush()

        await anomaly_repo.add(make_detection_result(invoice_id=high_invoice.id, score=0.90))
        await anomaly_repo.add(make_detection_result(invoice_id=low_invoice.id, score=0.10))
        await db_session.flush()

        above = await anomaly_repo.list_above_threshold(threshold=0.80, limit=10)
        ids = [r.invoice_id for r in above]
        assert high_invoice.id in ids
        assert low_invoice.id not in ids
