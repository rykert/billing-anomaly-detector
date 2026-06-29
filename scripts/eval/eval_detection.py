"""
Evaluation: measure precision, recall, and F1 of the anomaly detector
against the 50 known anomalies injected by the ETL.

Ground truth rule: invoices with billed_amount / allowed_amount > 4.0
are known anomalies. This threshold captures all five injected anomaly
types without overlapping with normal claims (which have ratios 1.0-2.2x).
"""

import asyncio
from decimal import Decimal

from sqlalchemy import select

from billing_anomaly_detector.infrastructure.config import get_settings
from billing_anomaly_detector.infrastructure.persistence.database import (
    build_engine,
    build_session_factory,
)
from billing_anomaly_detector.infrastructure.persistence.models import (
    AnomalyResultModel,
    InvoiceModel,
)

GROUND_TRUTH_RATIO = Decimal("4.0")


async def run_eval() -> None:
    settings = get_settings()
    engine = build_engine(settings.database_url)
    factory = build_session_factory(engine)

    async with factory() as session:
        # ── Ground truth ──────────────────────────────────────────────────
        invoice_stmt = select(
            InvoiceModel.id,
            InvoiceModel.billed_amount,
            InvoiceModel.allowed_amount,
        )
        invoice_rows = (await session.execute(invoice_stmt)).all()

        ground_truth = set()
        for row in invoice_rows:
            if row.allowed_amount > 0:
                ratio = Decimal(str(row.billed_amount)) / Decimal(str(row.allowed_amount))
                if ratio > GROUND_TRUTH_RATIO:
                    ground_truth.add(row.id)

        # ── Detected ──────────────────────────────────────────────────────
        anomaly_stmt = select(
            AnomalyResultModel.invoice_id,
            AnomalyResultModel.score,
        )
        anomaly_rows = (await session.execute(anomaly_stmt)).all()
        detected = {row.invoice_id for row in anomaly_rows}

        # ── Confusion matrix ──────────────────────────────────────────────
        tp = len(ground_truth & detected)
        fp = len(detected - ground_truth)
        fn = len(ground_truth - detected)
        tn = len(invoice_rows) - tp - fp - fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        # ── Report ────────────────────────────────────────────────────────
        print(f"\n{'='*54}")
        print("  BILLING ANOMALY DETECTOR — EVALUATION RESULTS")
        print(f"{'='*54}")
        print(f"\n  Total invoices:              {len(invoice_rows):>6}")
        print(f"  Ground truth anomalies:      {len(ground_truth):>6}  (ratio > {GROUND_TRUTH_RATIO}x)")
        print(f"  Detected anomalies:          {len(detected):>6}  (score >= {settings.anomaly_threshold})")
        print(f"\n  True Positives:              {tp:>6}  (correctly flagged)")
        print(f"  False Positives:             {fp:>6}  (normal claims flagged)")
        print(f"  False Negatives:             {fn:>6}  (anomalies missed)")
        print(f"  True Negatives:              {tn:>6}  (normal claims ignored)")
        print(f"\n  Precision:                   {precision:>6.3f}")
        print(f"  Recall:                      {recall:>6.3f}")
        print(f"  F1 Score:                    {f1:>6.3f}")
        print(f"{'='*54}\n")

        if fn > 0:
            print(f"  NOTE: {fn} anomaly/anomalies missed. The cosine similarity")
            print(f"  approach has a ceiling on structured billing data where")
            print(f"  all claims look textually similar. Isolation Forest")
            print(f"  (planned Phase 2) will improve recall on ratio-based")
            print(f"  anomalies by operating on raw numerical features.")
            print()

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(run_eval())
