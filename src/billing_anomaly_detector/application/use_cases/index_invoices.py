from billing_anomaly_detector.application.services.invoice_text import (
    invoice_to_text,
)
from billing_anomaly_detector.domain.ports import EmbeddingPort, InvoiceRepository


class IndexInvoicesUseCase:
    """
    Fetches unembedded invoices in batches, generates embeddings,
    and writes them back. Must complete before DetectAnomaliesUseCase runs.

    Only calls ports — never concrete adapters.
    The database session (and its transaction) is managed by the caller
    via dependency injection in the API layer (Phase 5).
    """

    def __init__(
        self,
        invoice_repo: InvoiceRepository,
        embedding_port: EmbeddingPort,
        batch_size: int = 100,
    ) -> None:
        self._invoice_repo = invoice_repo
        self._embedding_port = embedding_port
        self._batch_size = batch_size

    async def execute(self) -> int:
        """Returns the total number of invoices indexed."""
        total_indexed = 0

        while True:
            invoices = await self._invoice_repo.list_unembedded(
                limit=self._batch_size
            )
            if not invoices:
                break

            texts = [invoice_to_text(inv) for inv in invoices]
            embeddings = await self._embedding_port.embed_batch(texts)

            for invoice, embedding in zip(invoices, embeddings, strict=True):
                await self._invoice_repo.update_embedding(invoice.id, embedding)

            total_indexed += len(invoices)
            print(f"  Indexed {total_indexed} invoices...")

        return total_indexed
