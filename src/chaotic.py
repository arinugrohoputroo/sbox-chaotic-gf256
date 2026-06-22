"""Logistic Map chaotic sequence generator."""

from __future__ import annotations


def logistic_map(r: float, x0: float, n: int = 512, skip: int = 100) -> list[float]:
    """
    Generate chaotic sequence using Logistic Map: x_{n+1} = r * x_n * (1 - x_n).

    Args:
        r: Control parameter (chaotic regime typically 3.57 - 4.0).
        x0: Initial seed in (0, 1).
        n: Total iterations including skipped transient.
        skip: Number of initial iterations to discard.

    Returns:
        List of chaotic values in (0, 1).
    """
    if not 0 < x0 < 1:
        raise ValueError("x0 must be in the open interval (0, 1)")
    if r <= 0:
        raise ValueError("r must be positive")

    x = x0
    total = n + skip
    sequence: list[float] = []

    for _ in range(total):
        x = r * x * (1.0 - x)
        sequence.append(x)

    return sequence[skip:]


def chaotic_permutation(sequence: list[float], size: int = 256) -> list[int]:
    """
    Build a bijective permutation of 0..size-1 from chaotic values.

    Uses rank-order sorting to guarantee uniqueness even when scaled
    integers collide.
    """
    if len(sequence) < size:
        raise ValueError(f"Need at least {size} chaotic values, got {len(sequence)}")

    indexed = [(sequence[i], i) for i in range(size)]
    indexed.sort(key=lambda item: item[0])

    permutation = [0] * size
    for rank, (_, original_index) in enumerate(indexed):
        permutation[original_index] = rank

    return permutation
