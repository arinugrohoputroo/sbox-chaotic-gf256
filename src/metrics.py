"""Cryptographic metrics for 8-bit S-boxes: NL, SAC, BIC-NL, BIC-SAC, LAP, DAP."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.gf256 import AES_SBOX


@dataclass
class SBoxMetrics:
    nl_min: float
    nl_avg: float
    sac_avg: float
    sac_max_dev: float
    bic_nl_min: float
    bic_nl_avg: float
    bic_sac_max_dev: float
    bic_sac_avg_dev: float
    lap: float
    dap: float
    sac_matrix: np.ndarray

    def as_dict(self) -> dict[str, float]:
        return {
            "NL (min)": self.nl_min,
            "NL (avg)": self.nl_avg,
            "SAC (avg)": self.sac_avg,
            "SAC max dev from 0.5": self.sac_max_dev,
            "BIC-NL (min)": self.bic_nl_min,
            "BIC-NL (avg)": self.bic_nl_avg,
            "BIC-SAC max dev from 0.25": self.bic_sac_max_dev,
            "BIC-SAC avg dev from 0.25": self.bic_sac_avg_dev,
            "LAP": self.lap,
            "DAP": self.dap,
        }


def _truth_table(sbox: list[int], output_bit: int) -> list[int]:
    """Boolean function for one output bit across all 256 inputs."""
    return [(sbox[x] >> output_bit) & 1 for x in range(256)]


def _parity(value: int) -> int:
    """GF(2) parity of integer bits."""
    return bin(value).count("1") & 1


def walsh_max_abs(truth: list[int]) -> int:
    """Maximum absolute Walsh-Hadamard coefficient."""
    max_abs = 0
    for mask in range(256):
        total = 0
        for x in range(256):
            sign = 1 - 2 * (_parity(mask & x) ^ truth[x])
            total += sign
        max_abs = max(max_abs, abs(total))
    return max_abs


def nonlinearity(truth: list[int]) -> int:
    """Nonlinearity of an 8-variable Boolean function."""
    return 128 - walsh_max_abs(truth) // 2


def compute_nl(sbox: list[int]) -> tuple[float, float]:
    """Return (min NL, average NL) across output bit functions."""
    nls = [nonlinearity(_truth_table(sbox, bit)) for bit in range(8)]
    return float(min(nls)), float(sum(nls) / len(nls))


def compute_sac_matrix(sbox: list[int]) -> np.ndarray:
    """
    SAC matrix sac[i][j]: probability output bit i flips when input bit j is flipped.
    Shape: (8 output bits, 8 input bits).
    """
    matrix = np.zeros((8, 8), dtype=np.float64)
    for j in range(8):
        flip_mask = 1 << (7 - j)
        for x in range(256):
            diff = sbox[x] ^ sbox[x ^ flip_mask]
            for i in range(8):
                matrix[i, j] += (diff >> i) & 1
    matrix /= 256.0
    return matrix


def compute_sac(sbox: list[int]) -> tuple[float, float, np.ndarray]:
    """Return average SAC, max deviation from 0.5, and SAC matrix."""
    matrix = compute_sac_matrix(sbox)
    avg = float(matrix.mean())
    max_dev = float(np.max(np.abs(matrix - 0.5)))
    return avg, max_dev, matrix


def compute_bic_nl(sbox: list[int]) -> tuple[float, float]:
    """BIC-NL: nonlinearity of XOR pairs of output bit functions."""
    nls = []
    for i in range(8):
        for k in range(i + 1, 8):
            xor_truth = [
                ((sbox[x] >> i) & 1) ^ ((sbox[x] >> k) & 1) for x in range(256)
            ]
            nls.append(nonlinearity(xor_truth))
    return float(min(nls)), float(sum(nls) / len(nls))


def compute_bic_sac(sbox: list[int]) -> tuple[float, float]:
    """
    BIC-SAC: deviation of joint flip probability from 0.25 for output bit pairs
    when one input bit is flipped.
    """
    deviations = []
    for j in range(8):
        flip_mask = 1 << (7 - j)
        for i in range(8):
            for k in range(i + 1, 8):
                both = 0
                for x in range(256):
                    diff = sbox[x] ^ sbox[x ^ flip_mask]
                    bi = (diff >> i) & 1
                    bk = (diff >> k) & 1
                    both += bi & bk
                prob = both / 256.0
                deviations.append(abs(prob - 0.25))
    return float(max(deviations)), float(sum(deviations) / len(deviations))


def compute_lap(sbox: list[int]) -> float:
    """Linear Approximation Probability (maximum absolute bias)."""
    max_bias = 0.0
    for a in range(1, 256):
        for b in range(1, 256):
            count = 0
            for x in range(256):
                if _parity(a & x) == _parity(b & sbox[x]):
                    count += 1
            bias = abs(count / 256.0 - 0.5)
            max_bias = max(max_bias, bias)
    return max_bias


def compute_dap(sbox: list[int]) -> float:
    """Differential Approximation Probability (max DDT entry / 256, dx != 0)."""
    max_prob = 0.0
    for dx in range(1, 256):
        counts: dict[int, int] = {}
        for x in range(256):
            dy = sbox[x] ^ sbox[x ^ dx]
            counts[dy] = counts.get(dy, 0) + 1
        max_prob = max(max_prob, max(counts.values()) / 256.0)
    return max_prob


def evaluate_sbox(sbox: list[int]) -> SBoxMetrics:
    """Compute all six cryptographic criteria."""
    nl_min, nl_avg = compute_nl(sbox)
    sac_avg, sac_max_dev, sac_matrix = compute_sac(sbox)
    bic_nl_min, bic_nl_avg = compute_bic_nl(sbox)
    bic_sac_max, bic_sac_avg = compute_bic_sac(sbox)
    lap = compute_lap(sbox)
    dap = compute_dap(sbox)

    return SBoxMetrics(
        nl_min=nl_min,
        nl_avg=nl_avg,
        sac_avg=sac_avg,
        sac_max_dev=sac_max_dev,
        bic_nl_min=bic_nl_min,
        bic_nl_avg=bic_nl_avg,
        bic_sac_max_dev=bic_sac_max,
        bic_sac_avg_dev=bic_sac_avg,
        lap=lap,
        dap=dap,
        sac_matrix=sac_matrix,
    )


def evaluate_aes_sbox() -> SBoxMetrics:
    """Benchmark metrics for the standard AES S-box."""
    return evaluate_sbox(AES_SBOX)


# Target thresholds for 8-bit S-box quality (operator, threshold).
METRIC_TARGETS: dict[str, tuple[str, float]] = {
    "NL (min)": (">=", 104.0),
    "NL (avg)": (">=", 104.0),
    "SAC (avg)": ("~", 0.5),
    "SAC max dev from 0.5": ("<=", 0.0625),
    "BIC-NL (min)": (">=", 100.0),
    "BIC-NL (avg)": (">=", 104.0),
    "BIC-SAC max dev from 0.25": ("<=", 0.070313),
    "BIC-SAC avg dev from 0.25": ("<=", 0.025),
    "LAP": ("<=", 0.125),
    "DAP": ("<=", 0.03125),
}


def metric_passes(name: str, value: float, tolerance: float = 0.01) -> bool:
    """Return True when a metric meets its target threshold."""
    op, target = METRIC_TARGETS[name]
    if op == ">=":
        return value >= target
    if op == "<=":
        return value <= target
    return abs(value - target) <= tolerance


def count_passing_metrics(metrics: SBoxMetrics) -> tuple[int, int]:
    """Return (passed, total) metric count for a given S-box."""
    values = metrics.as_dict()
    passed = sum(1 for name, val in values.items() if metric_passes(name, val))
    return passed, len(values)
