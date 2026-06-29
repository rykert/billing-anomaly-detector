from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from billing_anomaly_detector.domain.entities import DetectionResult
from billing_anomaly_detector.domain.ports import AnomalyRepository
from billing_anomaly_detector.domain.value_objects import AnomalyScore
from billing_anomaly_detector.infrastructure.persistence.models import AnomalyResultModel


class SqlAlchemyAnomalyRepository(AnomalyRepository):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, result: DetectionResult) -> None:
        model = self._to_model(result)
        self._session.add(model)
        await self._session.flush()

    async def update_explanation(
        self, invoice_id: UUID, explanation: str
    ) -> None:
        stmt = (
            update(AnomalyResultModel)
            .where(AnomalyResultModel.invoice_id == invoice_id)
            .values(explanation=explanation)
        )
        await self._session.execute(stmt)
        await self._session.flush()

    async def list_above_threshold(
        self, threshold: float, limit: int = 20
    ) -> list[DetectionResult]:
        stmt = (
            select(AnomalyResultModel)
            .where(AnomalyResultModel.score >= threshold)
            .order_by(AnomalyResultModel.score.desc())
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return [self._to_domain(m) for m in result.scalars().all()]

    async def get_by_invoice(
        self, invoice_id: UUID
    ) -> DetectionResult | None:
        stmt = select(AnomalyResultModel).where(
            AnomalyResultModel.invoice_id == invoice_id
        )
        result = await self._session.execute(stmt)
        model = result.scalar_one_or_none()
        return self._to_domain(model) if model else None

    @staticmethod
    def _to_model(result: DetectionResult) -> AnomalyResultModel:
        return AnomalyResultModel(
            id=result.id,
            invoice_id=result.invoice_id,
            score=result.score.value,
            explanation=result.explanation,
        )

    @staticmethod
    def _to_domain(model: AnomalyResultModel) -> DetectionResult:
        return DetectionResult(
            id=model.id,
            invoice_id=model.invoice_id,
            score=AnomalyScore(model.score),
            explanation=model.explanation,
            detected_at=model.detected_at,
        )
