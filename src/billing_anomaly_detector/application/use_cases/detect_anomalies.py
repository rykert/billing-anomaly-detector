from billing_anomaly_detector.application.services.cosine_scorer import (
    compute_centroid,
    score_embeddings,
)
from billing_anomaly_detector.domain.ports import AnomalyRepository, InvoiceRepository


class DetectAnomaliesUseCase:
    """
    Scores all embedded invoices by cosine distance from the centroid.
    Invoices above threshold are flagged — DetectionResults are stored
    and AnomalyDetected domain events are emitted.

    Only fetches full Invoice objects for flagged invoices (those above
    threshold) to avoid the N+1 problem on the full 5,050-row dataset.
    The scored list is sorted descending, so we break as soon as we
    hit the first score below threshold.

    Known limitation (Phase 2 TODO): only flagged invoices get
    DetectionResults stored. Full dataset scores live only in memory
    during a run. For comprehensive eval reporting, extend to store all.
    """

    def __init__(
        self,
        invoice_repo: InvoiceRepository,
        anomaly_repo: AnomalyRepository,
        threshold: float = 0.80,
    ) -> None:
        self._invoice_repo = invoice_repo
        self._anomaly_repo = anomaly_repo
        self._threshold = threshold

    async def execute(self) -> dict[str, int]:
        """Returns {'total_scored': N, 'total_flagged': M}."""
        all_embeddings = await self._invoice_repo.list_all_embeddings()
        if not all_embeddings:
            return {"total_scored": 0, "total_flagged": 0}

        centroid = compute_centroid([emb for _, emb in all_embeddings])
        scored = score_embeddings(all_embeddings, centroid)

        total_flagged = 0
        for invoice_id, score in scored:
            if not score.exceeds(self._threshold):
                break  # sorted descending — everything below is also below threshold

            invoice = await self._invoice_repo.get(invoice_id)
            if invoice is None:
                continue

            result = invoice.flag_anomaly(score, self._threshold)
            await self._anomaly_repo.add(result)

            events = invoice.pull_events()
            total_flagged += len(events)

        return {
            "total_scored": len(scored),
            "total_flagged": total_flagged,
        }
