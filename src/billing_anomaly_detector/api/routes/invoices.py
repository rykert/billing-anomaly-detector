from fastapi import APIRouter, Depends

from billing_anomaly_detector.api.dependencies import (
    get_embedding_adapter,
    get_invoice_repo,
)
from billing_anomaly_detector.api.schemas import IndexResponse
from billing_anomaly_detector.application.use_cases.index_invoices import (
    IndexInvoicesUseCase,
)
from billing_anomaly_detector.domain.ports import EmbeddingPort, InvoiceRepository

router = APIRouter()


@router.post("/index", response_model=IndexResponse)
async def index_invoices(
    invoice_repo: InvoiceRepository = Depends(get_invoice_repo),
    embedding_port: EmbeddingPort = Depends(get_embedding_adapter),
) -> IndexResponse:
    """
    Embed all unembedded invoices and store their vectors.
    Run this once after loading data, and again whenever new invoices are added.
    This is a long-running operation (~5 minutes for 5,050 invoices via Azure).
    """
    use_case = IndexInvoicesUseCase(invoice_repo, embedding_port)
    indexed = await use_case.execute()
    return IndexResponse(
        indexed=indexed,
        message=f"Successfully embedded {indexed} invoices.",
    )
