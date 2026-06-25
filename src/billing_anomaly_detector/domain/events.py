from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from .value_objects import AnomalyScore


class DomainEvent:
    """Marker base class for all domain events."""


@dataclass(frozen=True, slots=True)
class AnomalyDetected(DomainEvent):
    invoice_id: UUID
    score: AnomalyScore
    occurred_at: datetime
