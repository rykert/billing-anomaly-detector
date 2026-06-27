from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from billing_anomaly_detector.domain.entities import Invoice
from billing_anomaly_detector.domain.ports import InvoiceRepository
from billing_anomaly_detector.domain.value_objects import ClaimCode, MemberId, Money
from billing_anomaly_detector.infrastructure.persistence.models import InvoiceModel


class SqlAlchemyInvoiceRepository(InvoiceRepository):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, invoice: Invoice) -> None:
        model = self._to_model(invoice)
        self._session.add(model)
        await self._session.flush()

    async def get(self, invoice_id: UUID) -> Invoice | None:
        model = await self._session.get(InvoiceModel, invoice_id)
        return self._to_domain(model) if model else None

    async def update_embedding(
        self, invoice_id: UUID, embedding: list[float]
    ) -> None:
        stmt = (
            update(InvoiceModel)
            .where(InvoiceModel.id == invoice_id)
            .values(embedding=embedding)
        )
        await self._session.execute(stmt)
        await self._session.flush()

    async def list_unembedded(self, limit: int = 500) -> list[Invoice]:
        stmt = (
            select(InvoiceModel)
            .where(InvoiceModel.embedding.is_(None))
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return [self._to_domain(m) for m in result.scalars().all()]

    async def list_all_embeddings(
        self,
    ) -> list[tuple[UUID, list[float]]]:
        stmt = select(InvoiceModel.id, InvoiceModel.embedding).where(
            InvoiceModel.embedding.isnot(None)
        )
        result = await self._session.execute(stmt)
        return [(row.id, list(row.embedding)) for row in result.all()]

    @staticmethod
    def _to_model(invoice: Invoice) -> InvoiceModel:
        return InvoiceModel(
            id=invoice.id,
            member_id=invoice.member_id.value,
            claim_code=invoice.claim_code.value,
            provider_npi=invoice.provider_npi,
            billed_amount=invoice.billed_amount.amount,
            billed_currency=invoice.billed_amount.currency,
            allowed_amount=invoice.allowed_amount.amount,
            allowed_currency=invoice.allowed_amount.currency,
            service_date=invoice.service_date,
            embedding=invoice.embedding,
        )

    @staticmethod
    def _to_domain(model: InvoiceModel) -> Invoice:
        return Invoice(
            id=model.id,
            member_id=MemberId(model.member_id),
            claim_code=ClaimCode(model.claim_code),
            provider_npi=model.provider_npi,
            billed_amount=Money(model.billed_amount, model.billed_currency),
            allowed_amount=Money(model.allowed_amount, model.allowed_currency),
            service_date=model.service_date,
            embedding=(
                list(model.embedding) if model.embedding is not None else None
            ),
        )
