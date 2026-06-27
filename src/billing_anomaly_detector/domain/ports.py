from __future__ import annotations

from abc import ABC, abstractmethod
from uuid import UUID

from .entities import DetectionResult, Invoice
from .value_objects import AnomalyScore


class InvoiceRepository(ABC):
    @abstractmethod
    async def add(self, invoice: Invoice) -> None: ...

    @abstractmethod
    async def get(self, invoice_id: UUID) -> Invoice | None: ...

    @abstractmethod
    async def update_embedding(
        self, invoice_id: UUID, embedding: list[float]
    ) -> None: ...

    @abstractmethod
    async def list_unembedded(self, limit: int = 500) -> list[Invoice]: ...

    @abstractmethod
    async def list_all_embeddings(
        self,
    ) -> list[tuple[UUID, list[float]]]: ...


class AnomalyRepository(ABC):
    @abstractmethod
    async def add(self, result: DetectionResult) -> None: ...

    @abstractmethod
    async def list_above_threshold(
        self, threshold: float, limit: int = 20
    ) -> list[DetectionResult]: ...

    @abstractmethod
    async def get_by_invoice(
        self, invoice_id: UUID
    ) -> DetectionResult | None: ...


class EmbeddingPort(ABC):
    @abstractmethod
    async def embed(self, text: str) -> list[float]: ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


class ExplanationPort(ABC):
    @abstractmethod
    async def explain(
        self,
        invoice: Invoice,
        score: AnomalyScore,
        neighbors: list[Invoice],
    ) -> str: ...
