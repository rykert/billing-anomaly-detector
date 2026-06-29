import uuid
from decimal import Decimal

from billing_anomaly_detector.domain.value_objects import AnomalyScore

from tests.conftest import make_invoice


class TestInvoice:
    def test_billed_to_allowed_ratio(self) -> None:
        invoice = make_invoice(billed_amount="150.00", allowed_amount="100.00")
        assert invoice.billed_to_allowed_ratio == Decimal("1.5")

    def test_attach_embedding(self) -> None:
        invoice = make_invoice()
        embedding = [0.1] * 1536
        invoice.attach_embedding(embedding)
        assert invoice.embedding == embedding

    def test_flag_anomaly_above_threshold_emits_event(self) -> None:
        invoice = make_invoice()
        score = AnomalyScore(0.85)
        result = invoice.flag_anomaly(score, threshold=0.80)

        events = invoice.pull_events()
        assert len(events) == 1
        assert result.score == score

    def test_flag_anomaly_below_threshold_no_event(self) -> None:
        invoice = make_invoice()
        score = AnomalyScore(0.75)
        invoice.flag_anomaly(score, threshold=0.80)

        events = invoice.pull_events()
        assert len(events) == 0

    def test_pull_events_drains_buffer(self) -> None:
        invoice = make_invoice()
        invoice.flag_anomaly(AnomalyScore(0.85), threshold=0.80)

        first_pull = invoice.pull_events()
        second_pull = invoice.pull_events()

        assert len(first_pull) == 1
        assert len(second_pull) == 0  # drained on first pull


class TestDetectionResult:
    def test_attach_explanation(self) -> None:
        from tests.conftest import make_detection_result
        result = make_detection_result()
        assert result.explanation is None

        result.attach_explanation("This claim is unusual.")
        assert result.explanation == "This claim is unusual."
