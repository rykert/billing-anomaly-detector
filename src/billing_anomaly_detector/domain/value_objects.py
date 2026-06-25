from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class Money:
    amount: Decimal
    currency: str = "USD"

    def __post_init__(self) -> None:
        if self.amount < 0:
            raise ValueError("Money amount cannot be negative")
        if len(self.currency) != 3:
            raise ValueError("Currency must be a 3-letter ISO code")

    def __add__(self, other: Money) -> Money:
        if self.currency != other.currency:
            raise ValueError("Cannot add Money of different currencies")
        return Money(self.amount + other.amount, self.currency)

    def ratio_to(self, other: Money) -> Decimal:
        if other.amount == 0:
            raise ValueError("Cannot compute ratio against zero amount")
        return self.amount / other.amount


@dataclass(frozen=True, slots=True)
class MemberId:
    value: str

    def __post_init__(self) -> None:
        if not self.value.strip():
            raise ValueError("MemberId cannot be empty")


@dataclass(frozen=True, slots=True)
class ClaimCode:
    value: str  # HCPCS/CPT code

    def __post_init__(self) -> None:
        if not (3 <= len(self.value) <= 10):
            raise ValueError(f"Invalid claim code: {self.value!r}")


@dataclass(frozen=True, slots=True)
class AnomalyScore:
    value: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.value <= 1.0:
            raise ValueError("AnomalyScore must be between 0.0 and 1.0")

    def exceeds(self, threshold: float) -> bool:
        return self.value >= threshold
