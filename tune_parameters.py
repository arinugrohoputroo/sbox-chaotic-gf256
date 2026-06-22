"""Grid/random search for Logistic Map parameters (r, x0) with balanced crypto metrics."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics import (  # noqa: E402
    compute_bic_nl,
    compute_bic_sac,
    compute_dap,
    compute_lap,
    compute_nl,
    compute_sac,
    evaluate_aes_sbox,
    evaluate_sbox,
)
from src.sbox_builder import TUNED_R, TUNED_X0  # noqa: E402
from src.sbox_builder import build_sbox  # noqa: E402

DEFAULT_R = TUNED_R
DEFAULT_X0 = TUNED_X0


def composite_score(metrics, include_lap: bool = True) -> float:
    """Higher is better; targets approximate AES-quality criteria."""
    score = (
        metrics.nl_min * 4.0
        + metrics.nl_avg * 2.0
        + metrics.bic_nl_min * 1.5
        + metrics.bic_nl_avg * 0.5
        + (0.0625 - metrics.sac_max_dev) * 200.0
        + (0.070312 - metrics.bic_sac_max_dev) * 150.0
        + (0.015625 - metrics.dap) * 400.0
    )
    if include_lap:
        score += (0.0625 - metrics.lap) * 250.0
    return score


def fast_metrics(sbox: list[int]):
    nl_min, nl_avg = compute_nl(sbox)
    _, sac_max_dev, _ = compute_sac(sbox)
    bic_nl_min, bic_nl_avg = compute_bic_nl(sbox)
    bic_sac_max, bic_sac_avg = compute_bic_sac(sbox)
    dap = compute_dap(sbox)
    return {
        "nl_min": nl_min,
        "nl_avg": nl_avg,
        "sac_max_dev": sac_max_dev,
        "bic_nl_min": bic_nl_min,
        "bic_nl_avg": bic_nl_avg,
        "bic_sac_max_dev": bic_sac_max,
        "bic_sac_avg_dev": bic_sac_avg,
        "dap": dap,
    }


def _fast_score(fm: dict) -> float:
    return (
        fm["nl_min"] * 4.0
        + fm["nl_avg"] * 2.0
        + fm["bic_nl_min"] * 1.5
        + fm["bic_nl_avg"] * 0.5
        + (0.0625 - fm["sac_max_dev"]) * 200.0
        + (0.070312 - fm["bic_sac_max_dev"]) * 150.0
        + (0.015625 - fm["dap"]) * 400.0
    )


def sample_params(count: int, seed: int) -> list[tuple[float, float]]:
    rng = random.Random(seed)
    samples: set[tuple[float, float]] = set()

    r_grid = [round(3.57 + i * 0.01, 2) for i in range(44)]
    x0_grid = [round(0.01 + i * 0.01, 2) for i in range(99)]

    for r in r_grid:
        for x0 in x0_grid:
            samples.add((r, x0))

    extra = []
    while len(extra) < count:
        r = round(rng.uniform(3.57, 4.0), 6)
        x0 = round(rng.uniform(0.001, 0.999), 9)
        extra.append((r, x0))

    all_samples = list(samples) + extra
    rng.shuffle(all_samples)
    return all_samples[:count]


def tune(count: int = 500, top_k: int = 40, seed: int = 42) -> dict:
    aes = evaluate_aes_sbox()
    candidates: list[tuple[float, float, float]] = []

    print(f"Phase 1: fast scan on {count} parameter pairs...", flush=True)
    t0 = time.time()
    for r, x0 in sample_params(count, seed):
        sbox, meta = build_sbox(r=r, x0=x0)
        if not meta["bijective"]:
            continue
        fm = fast_metrics(sbox)
        candidates.append((_fast_score(fm), r, x0))

    candidates.sort(reverse=True)
    print(f"  {len(candidates)} bijective pairs in {time.time() - t0:.1f}s", flush=True)

    print(f"Phase 2: full evaluation on top {top_k}...", flush=True)
    ranked: list[tuple[float, float, float, object]] = []
    for _, r, x0 in candidates[:top_k]:
        sbox, _ = build_sbox(r=r, x0=x0)
        metrics = evaluate_sbox(sbox)
        ranked.append((composite_score(metrics), r, x0, metrics))

    ranked.sort(reverse=True)
    best_score, best_r, best_x0, best_metrics = ranked[0]

    result = {
        "r": best_r,
        "x0": best_x0,
        "score": best_score,
        "metrics": best_metrics.as_dict(),
        "aes_metrics": aes.as_dict(),
        "top5": [
            {
                "r": r,
                "x0": x0,
                "score": sc,
                "nl_min": m.nl_min,
                "sac_max_dev": m.sac_max_dev,
                "lap": m.lap,
                "dap": m.dap,
            }
            for sc, r, x0, m in ranked[:5]
        ],
    }
    print(f"Done in {time.time() - t0:.1f}s", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune Logistic Map S-box parameters")
    parser.add_argument("--count", type=int, default=500, help="Number of parameter pairs")
    parser.add_argument("--top-k", type=int, default=40, help="Full-eval shortlist size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("report/tuned_params.json"))
    args = parser.parse_args()

    result = tune(count=args.count, top_k=args.top_k, seed=args.seed)

    print("\nBest parameters:")
    print(f"  r  = {result['r']}")
    print(f"  x0 = {result['x0']}")
    print(f"  score = {result['score']:.2f}")
    print("\nMetrics vs AES:")
    for key, value in result["metrics"].items():
        aes_val = result["aes_metrics"][key]
        print(f"  {key}: {value:.6f}  (AES: {aes_val:.6f})")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
