"""PR-Piet schone-PR test: correcte, eenvoudige code zonder bewuste problemen."""

from __future__ import annotations


def clamp(value: int, low: int, high: int) -> int:
    """Begrens een waarde aan [low, high]."""
    if low > high:
        raise ValueError("low mag niet groter zijn dan high")
    return max(low, min(high, value))


def is_even(value: int) -> bool:
    """True als value even is."""
    return value % 2 == 0
