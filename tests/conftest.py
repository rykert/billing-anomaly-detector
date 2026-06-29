"""
Shared fixtures for unit tests.
Provides factory functions for creating domain objects without touching
the database or any external services.
"""

import uuid
from datetime import date
from decimal import Decimal

import pytest

from billing_anomaly_detector.domain.entities import DetectionResult, Invoice
from billing_anomaly_detector.domain.value_objects import (
    AnomalyScore,
    ClaimCode,
    MemberId,
    Money,
)


def make_invoice(
    id: uuid.UUID | None = None,
    member_id: str = "MBR123456",
    claim_code: str = "99213",
    provider_npi: str = "1234567890",
    billed_amount: str = "150.00",
    allowed_amount: str = "100.00",
    service_date: date | None = None,
    embedding: list[float] | None = None,
) -> Invoice:
    """Factory function for creating Invoice instances in tests."""
    return Invoice(
        id=id or uuid.uuid4(),
        member_id=MemberId(member_id),
        claim_code=ClaimCode(claim_code),
        provider_npi=provider_npi,
        billed_amount=Money(Decimal(billed_amount)),
        allowed_amount=Money(Decimal(allowed_amount)),
        service_date=service_date or date(2009, 1, 15),
        embedding=embedding,
    )


def make_detection_result(
    invoice_id: uuid.UUID | None = None,
    score: float = 0.85,
    explanation: str | None = None,
) -> DetectionResult:
    """Factory function for creating DetectionResult instances in tests."""
    return DetectionResult(
        id=uuid.uuid4(),
        invoice_id=invoice_id or uuid.uuid4(),
        score=AnomalyScore(score),
        explanation=explanation,
    )
