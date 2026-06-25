from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from uuid import UUID, uuid4

from .events import AnomalyDetected, DomainEvent
from .value_objects import AnomalyScore, ClaimCode, MemberId, Money


@dataclass(slots=True)
class Invoice:
    """Aggregate root — a single billed claim line."""

    id: UUID
    member_id: MemberId
    claim_code: ClaimCode
    provider_npi: str
    billed_amount: Money
    allowed_amount: Money
    service_date: date
    embedding: list[float] | None = field(default=None, repr=False)
    _events: list[DomainEvent] = field(default_factory=list, repr=False)

    @classmethod
    def create(
        cls,
        member_id: MemberId,
        claim_code: ClaimCode,
        provider_npi: str,
        billed_amount: Money,
        allowed_amount: Money,
        service_date: date,
    ) -> Invoice:
        return cls(
            id=uuid4(),
            member_id=member_id,
            claim_code=claim_code,
            provider_npi=provider_npi,
            billed_amount=billed_amount,
            allowed_amount=allowed_amount,
            service_date=service_date,
        )

    @property
    def billed_to_allowed_ratio(self) -> Decimal:
        return self.billed_amount.ratio_to(self.allowed_amount)

    def attach_embedding(self, embedding: list[float]) -> None:
        self.embedding = embedding

    def flag_anomaly(self, score: AnomalyScore, threshold: float) -> DetectionResult:
        result = DetectionResult.create(invoice_id=self.id, score=score)
        if score.exceeds(threshold):
            self._events.append(
                AnomalyDetected(invoice_id=self.id, score=score, occurred_at=datetime.utcnow())
            )
        return result

    def pull_events(self) -> list[DomainEvent]:
        events, self._events = self._events, []
        return events


@dataclass(slots=True)
class DetectionResult:
    id: UUID
    invoice_id: UUID
    score: AnomalyScore
    explanation: str | None = None
    detected_at: datetime = field(default_factory=datetime.utcnow)

    @classmethod
    def create(cls, invoice_id: UUID, score: AnomalyScore) -> DetectionResult:
        return cls(id=uuid4(), invoice_id=invoice_id, score=score)

    def attach_explanation(self, explanation: str) -> None:
        self.explanation = explanation
