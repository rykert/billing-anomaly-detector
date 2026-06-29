from decimal import Decimal

import pytest

from billing_anomaly_detector.domain.value_objects import (
    AnomalyScore,
    ClaimCode,
    MemberId,
    Money,
)


class TestMoney:
    def test_rejects_negative_amount(self) -> None:
        with pytest.raises(ValueError, match="negative"):
            Money(Decimal("-1.00"))

    def test_rejects_invalid_currency(self) -> None:
        with pytest.raises(ValueError, match="3-letter"):
            Money(Decimal("10.00"), "DOLLARS")

    def test_ratio_to_computes_correctly(self) -> None:
        billed = Money(Decimal("150.00"))
        allowed = Money(Decimal("100.00"))
        assert billed.ratio_to(allowed) == Decimal("1.5")

    def test_ratio_to_raises_on_zero_denominator(self) -> None:
        with pytest.raises(ValueError, match="zero"):
            Money(Decimal("150.00")).ratio_to(Money(Decimal("0.00")))

    def test_add_same_currency(self) -> None:
        a = Money(Decimal("50.00"))
        b = Money(Decimal("75.00"))
        assert (a + b).amount == Decimal("125.00")

    def test_add_different_currency_raises(self) -> None:
        with pytest.raises(ValueError, match="currencies"):
            Money(Decimal("50.00"), "USD") + Money(Decimal("50.00"), "EUR")


class TestAnomalyScore:
    def test_rejects_above_one(self) -> None:
        with pytest.raises(ValueError):
            AnomalyScore(1.5)

    def test_rejects_below_zero(self) -> None:
        with pytest.raises(ValueError):
            AnomalyScore(-0.1)

    def test_accepts_boundary_values(self) -> None:
        assert AnomalyScore(0.0).value == 0.0
        assert AnomalyScore(1.0).value == 1.0

    def test_exceeds_threshold(self) -> None:
        assert AnomalyScore(0.85).exceeds(0.80) is True

    def test_does_not_exceed_threshold(self) -> None:
        assert AnomalyScore(0.75).exceeds(0.80) is False

    def test_exactly_at_threshold_exceeds(self) -> None:
        assert AnomalyScore(0.80).exceeds(0.80) is True


class TestClaimCode:
    def test_rejects_too_short(self) -> None:
        with pytest.raises(ValueError):
            ClaimCode("AB")

    def test_rejects_too_long(self) -> None:
        with pytest.raises(ValueError):
            ClaimCode("A" * 11)

    def test_accepts_valid_code(self) -> None:
        assert ClaimCode("99213").value == "99213"


class TestMemberId:
    def test_rejects_empty_string(self) -> None:
        with pytest.raises(ValueError):
            MemberId("")

    def test_rejects_whitespace_only(self) -> None:
        with pytest.raises(ValueError):
            MemberId("   ")

    def test_accepts_valid_id(self) -> None:
        assert MemberId("MBR123").value == "MBR123"
