from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class IndexResponse(BaseModel):
    indexed: int
    message: str


class DetectResponse(BaseModel):
    total_scored: int
    total_flagged: int


class AnomalyResultResponse(BaseModel):
    id: UUID
    invoice_id: UUID
    score: float
    explanation: str | None
    detected_at: datetime


class ExplainResponse(BaseModel):
    invoice_id: UUID
    score: float
    explanation: str
