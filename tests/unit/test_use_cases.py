import uuid
from unittest.mock import AsyncMock

from tests.conftest import make_invoice

from billing_anomaly_detector.application.use_cases.detect_anomalies import (
    DetectAnomaliesUseCase,
)
from billing_anomaly_detector.application.use_cases.index_invoices import (
    IndexInvoicesUseCase,
)
from billing_anomaly_detector.domain.ports import (
    AnomalyRepository,
    EmbeddingPort,
    InvoiceRepository,
)


class TestIndexInvoicesUseCase:
    async def test_returns_zero_when_nothing_to_index(self) -> None:
        mock_repo = AsyncMock(spec=InvoiceRepository)
        mock_repo.list_unembedded.return_value = []
        mock_embedding = AsyncMock(spec=EmbeddingPort)

        use_case = IndexInvoicesUseCase(mock_repo, mock_embedding)
        result = await use_case.execute()

        assert result == 0
        mock_embedding.embed_batch.assert_not_called()

    async def test_embeds_and_stores_batch(self) -> None:
        invoice1 = make_invoice()
        invoice2 = make_invoice()
        embedding_1 = [0.1] * 1536
        embedding_2 = [0.2] * 1536

        mock_repo = AsyncMock(spec=InvoiceRepository)
        mock_repo.list_unembedded.side_effect = [
            [invoice1, invoice2],  # first call — two invoices
            [],                    # second call — done
        ]

        mock_embedding = AsyncMock(spec=EmbeddingPort)
        mock_embedding.embed_batch.return_value = [embedding_1, embedding_2]

        use_case = IndexInvoicesUseCase(mock_repo, mock_embedding, batch_size=10)
        result = await use_case.execute()

        assert result == 2
        mock_embedding.embed_batch.assert_called_once()
        assert mock_repo.update_embedding.call_count == 2

    async def test_processes_multiple_batches(self) -> None:
        invoices_batch1 = [make_invoice() for _ in range(3)]
        invoices_batch2 = [make_invoice() for _ in range(2)]

        mock_repo = AsyncMock(spec=InvoiceRepository)
        mock_repo.list_unembedded.side_effect = [
            invoices_batch1,
            invoices_batch2,
            [],
        ]

        mock_embedding = AsyncMock(spec=EmbeddingPort)
        mock_embedding.embed_batch.side_effect = [
            [[0.1] * 4 for _ in range(3)],
            [[0.2] * 4 for _ in range(2)],
        ]

        use_case = IndexInvoicesUseCase(mock_repo, mock_embedding, batch_size=3)
        result = await use_case.execute()

        assert result == 5
        assert mock_embedding.embed_batch.call_count == 2


class TestDetectAnomaliesUseCase:
    async def test_returns_zeros_when_no_embeddings(self) -> None:
        mock_invoice_repo = AsyncMock(spec=InvoiceRepository)
        mock_invoice_repo.list_all_embeddings.return_value = []
        mock_anomaly_repo = AsyncMock(spec=AnomalyRepository)

        use_case = DetectAnomaliesUseCase(
            mock_invoice_repo, mock_anomaly_repo, threshold=0.030
        )
        result = await use_case.execute()

        assert result == {"total_scored": 0, "total_flagged": 0}
        mock_anomaly_repo.add.assert_not_called()

    async def test_scores_all_embeddings(self) -> None:
        id1 = uuid.uuid4()
        id2 = uuid.uuid4()

        mock_invoice_repo = AsyncMock(spec=InvoiceRepository)
        mock_invoice_repo.list_all_embeddings.return_value = [
            (id1, [1.0, 0.0, 0.0]),
            (id2, [0.0, 1.0, 0.0]),
        ]
        mock_invoice_repo.get.return_value = make_invoice()
        mock_anomaly_repo = AsyncMock(spec=AnomalyRepository)

        use_case = DetectAnomaliesUseCase(
            mock_invoice_repo, mock_anomaly_repo, threshold=0.030
        )
        result = await use_case.execute()

        assert result["total_scored"] == 2
