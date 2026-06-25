import pytest
from decimal import Decimal

from billing_anomaly_detector.domain.value_objects import AnomalyScore, Money


def test_money_rejects_negative_amount() -> None:
    with pytest.raises(ValueError):
        Money(Decimal("-1.00"))


def test_anomaly_score_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        AnomalyScore(1.5)


def test_money_ratio() -> None:
    billed = Money(Decimal("150.00"))
    allowed = Money(Decimal("100.00"))
    assert billed.ratio_to(allowed) == Decimal("1.5")
