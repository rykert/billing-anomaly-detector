from uuid import UUID

from billing_anomaly_detector.application.services.cosine_scorer import (
    find_neighbors,
)
from billing_anomaly_detector.domain.ports import (
    AnomalyRepository,
    ExplanationPort,
    InvoiceRepository,
)


class ExplainAnomalyUseCase:
    """
    Generates a plain-English explanation for a flagged invoice.
    Caches the result — subsequent calls return the stored explanation
    without hitting the LLM again.

    Returns None if the invoice or its detection result doesn't exist,
    so the route handler can return a 404.
    """

    def __init__(
        self,
        invoice_repo: InvoiceRepository,
        anomaly_repo: AnomalyRepository,
        explanation_port: ExplanationPort,
    ) -> None:
        self._invoice_repo = invoice_repo
        self._anomaly_repo = anomaly_repo
        self._explanation_port = explanation_port

    async def execute(
        self, invoice_id: UUID
    ) -> dict[str, object] | None:
        invoice = await self._invoice_repo.get(invoice_id)
        if invoice is None:
            return None

        result = await self._anomaly_repo.get_by_invoice(invoice_id)
        if result is None:
            return None

        if result.explanation:
            return {
                "invoice_id": invoice_id,
                "score": result.score.value,
                "explanation": result.explanation,
            }

        if invoice.embedding is None:
            return None

        all_embeddings = await self._invoice_repo.list_all_embeddings()
        neighbor_ids = find_neighbors(
            invoice_id, invoice.embedding, all_embeddings, limit=3
        )

        neighbors = []
        for nid in neighbor_ids:
            neighbor = await self._invoice_repo.get(nid)
            if neighbor is not None:
                neighbors.append(neighbor)

        explanation = await self._explanation_port.explain(
            invoice, result.score, neighbors
        )

        await self._anomaly_repo.update_explanation(invoice_id, explanation)

        return {
            "invoice_id": invoice_id,
            "score": result.score.value,
            "explanation": explanation,
        }
