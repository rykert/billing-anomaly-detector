from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException

from billing_anomaly_detector.api.dependencies import (
    get_anomaly_repo,
    get_explanation_chain,
    get_invoice_repo,
    get_settings,
)
from billing_anomaly_detector.api.schemas import (
    AnomalyResultResponse,
    DetectResponse,
    ExplainResponse,
)
from billing_anomaly_detector.application.use_cases.detect_anomalies import (
    DetectAnomaliesUseCase,
)
from billing_anomaly_detector.application.use_cases.explain_anomaly import (
    ExplainAnomalyUseCase,
)
from billing_anomaly_detector.domain.ports import (
    AnomalyRepository,
    ExplanationPort,
    InvoiceRepository,
)
from billing_anomaly_detector.infrastructure.config import Settings

router = APIRouter()


@router.post("/detect", response_model=DetectResponse)
async def detect_anomalies(
    invoice_repo: InvoiceRepository = Depends(get_invoice_repo),
    anomaly_repo: AnomalyRepository = Depends(get_anomaly_repo),
    settings: Settings = Depends(get_settings),
) -> DetectResponse:
    """
    Score all embedded invoices by cosine distance from the centroid.
    Flags invoices above the configured threshold and stores DetectionResults.
    Run this after /invoices/index completes.
    """
    use_case = DetectAnomaliesUseCase(
        invoice_repo, anomaly_repo, settings.anomaly_threshold
    )
    result = await use_case.execute()
    return DetectResponse(**result)


@router.get("", response_model=list[AnomalyResultResponse])
async def list_anomalies(
    threshold: float = 0.80,
    limit: int = 20,
    anomaly_repo: AnomalyRepository = Depends(get_anomaly_repo),
) -> list[AnomalyResultResponse]:
    """
    List anomalies above the given threshold, sorted by score descending.
    Default threshold: 0.80. Default limit: 20.
    """
    results = await anomaly_repo.list_above_threshold(threshold, limit)
    return [
        AnomalyResultResponse(
            id=r.id,
            invoice_id=r.invoice_id,
            score=r.score.value,
            explanation=r.explanation,
            detected_at=r.detected_at,
        )
        for r in results
    ]


@router.get("/{invoice_id}/explain", response_model=ExplainResponse)
async def explain_anomaly(
    invoice_id: UUID,
    invoice_repo: InvoiceRepository = Depends(get_invoice_repo),
    anomaly_repo: AnomalyRepository = Depends(get_anomaly_repo),
    explanation_port: ExplanationPort = Depends(get_explanation_chain),
) -> ExplainResponse:
    """
    Generate a plain-English explanation for a flagged invoice using GPT-5.
    The explanation is cached — subsequent calls return instantly.
    Returns 404 if the invoice doesn't exist or hasn't been flagged.
    """
    use_case = ExplainAnomalyUseCase(invoice_repo, anomaly_repo, explanation_port)
    result = await use_case.execute(invoice_id)

    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"No anomaly found for invoice {invoice_id}. "
            "Ensure the invoice exists and detection has been run.",
        )

    return ExplainResponse(**result)
