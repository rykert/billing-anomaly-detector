import uuid
from datetime import date, datetime
from decimal import Decimal

from pgvector.sqlalchemy import Vector
from sqlalchemy import Date, DateTime, ForeignKey, Numeric, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class InvoiceModel(Base):
    __tablename__ = "invoices"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    member_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    claim_code: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    provider_npi: Mapped[str] = mapped_column(String(20), nullable=False)
    billed_amount: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    billed_currency: Mapped[str] = mapped_column(String(3), nullable=False, default="USD")
    allowed_amount: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    allowed_currency: Mapped[str] = mapped_column(String(3), nullable=False, default="USD")
    service_date: Mapped[date] = mapped_column(Date, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(1536), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class AnomalyResultModel(Base):
    __tablename__ = "anomaly_results"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    invoice_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("invoices.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    score: Mapped[float] = mapped_column(nullable=False)
    explanation: Mapped[str | None] = mapped_column(Text, nullable=True)
    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
