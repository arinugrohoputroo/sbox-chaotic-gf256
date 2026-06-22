"""S-box construction pipeline using Logistic Map and GF(2^8) operations."""

from __future__ import annotations

# Grid-search tuned defaults (see tune_parameters.py)
TUNED_R = 4.0
TUNED_X0 = 0.5

from src.chaotic import chaotic_permutation, logistic_map  # noqa: E402
from src.gf256 import (
    affine_transform,
    build_inverse_table,
    constant_from_seed,
    matrix_from_seed,
)


def is_bijective(sbox: list[int]) -> bool:
    """Check whether S-box is a permutation of 0..255."""
    return len(sbox) == 256 and len(set(sbox)) == 256


def build_sbox(r: float = TUNED_R, x0: float = TUNED_X0) -> tuple[list[int], dict]:
    """
    Construct an 8x8 S-box using Logistic Map + GF(2^8) inverse + affine transform.

    Pipeline:
        1. Generate chaotic sequence (Logistic Map)
        2. Derive bijective permutation via rank-order
        3. Apply multiplicative inverse in GF(2^8)
        4. Apply affine transformation over GF(2)

    Returns:
        Tuple of (sbox list, metadata dict).
    """
    sequence = logistic_map(r=r, x0=x0, n=512, skip=100)
    permutation = chaotic_permutation(sequence, size=256)

    inv_table = build_inverse_table()
    intermediate = [inv_table[permutation[i]] for i in range(256)]

    matrix = matrix_from_seed(sequence[:64])
    constant = constant_from_seed(sequence[64:128])

    sbox = [affine_transform(intermediate[i], matrix, constant) for i in range(256)]

    metadata = {
        "r": r,
        "x0": x0,
        "bijective": is_bijective(sbox),
        "matrix": matrix,
        "constant": constant,
        "permutation_sample": permutation[:16],
    }
    return sbox, metadata


def sbox_to_table(sbox: list[int]) -> list[list[str]]:
    """Format S-box as 16x16 hex table for display."""
    table = []
    for row in range(16):
        table.append([f"{sbox[row * 16 + col]:02X}" for col in range(16)])
    return table
