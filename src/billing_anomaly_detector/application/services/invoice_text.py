from billing_anomaly_detector.domain.entities import Invoice


def invoice_to_text(invoice: Invoice) -> str:
    """
    Convert an Invoice domain entity to a descriptive text string for embedding.

    Richer, domain-specific text produces more semantically meaningful vectors.
    The billed-to-allowed ratio is included explicitly because it is the
    single strongest anomaly signal — the embedding model will cluster
    high-ratio claims separately from normal ones.
    """
    return (
        f"Healthcare billing claim. "
        f"Member ID: {invoice.member_id.value}. "
        f"Procedure code: {invoice.claim_code.value}. "
        f"Provider NPI: {invoice.provider_npi}. "
        f"Billed amount: ${invoice.billed_amount.amount:.2f} "
        f"{invoice.billed_amount.currency}. "
        f"Allowed amount: ${invoice.allowed_amount.amount:.2f} "
        f"{invoice.allowed_amount.currency}. "
        f"Billed-to-allowed ratio: {invoice.billed_to_allowed_ratio:.3f}. "
        f"Service date: {invoice.service_date}."
    )
