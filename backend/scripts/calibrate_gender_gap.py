#!/usr/bin/env python3
"""
Calibrate gender wage multipliers so the simulated global gap matches target.

Writes a JSON payload (default: backend/app/data/gender_gap_calibration.json)
used by reference_data.py at import time.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

from app.i18n import Language
from app.services import persona_manager as pm


EXCLUDED_OCCUPATIONS = {"emeryt", "rencista", "student", "bezrobotny"}


def _evaluate_gap(samples: int, seed: int, scale: float, base_delta: float) -> tuple[float, int, int]:
    pm.GENDER_WAGE_GAP = {
        "M": 1.0 + base_delta * scale,
        "F": 1.0 - base_delta * scale,
    }
    random.seed(seed)
    manager = pm.PersonaManager(language=Language.PL)
    male_incomes: list[int] = []
    female_incomes: list[int] = []

    for i in range(samples):
        persona = manager.generate_persona(index=i)
        if persona.occupation in EXCLUDED_OCCUPATIONS:
            continue
        if persona.gender == "M":
            male_incomes.append(persona.income)
        else:
            female_incomes.append(persona.income)

    if not male_incomes or not female_incomes:
        return 0.0, len(male_incomes), len(female_incomes)

    male_avg = sum(male_incomes) / len(male_incomes)
    female_avg = sum(female_incomes) / len(female_incomes)
    gap = (male_avg - female_avg) / male_avg
    return gap, len(male_incomes), len(female_incomes)


def calibrate(
    samples: int,
    seed: int,
    target_gap: float,
    base_delta: float,
    lo: float,
    hi: float,
    max_iter: int,
) -> dict:
    gap_lo, _, _ = _evaluate_gap(samples, seed, lo, base_delta)
    gap_hi, _, _ = _evaluate_gap(samples, seed, hi, base_delta)

    # If target not bracketed, pick closest endpoint.
    if not (min(gap_lo, gap_hi) <= target_gap <= max(gap_lo, gap_hi)):
        scale = lo if abs(gap_lo - target_gap) < abs(gap_hi - target_gap) else hi
        gap, male_n, female_n = _evaluate_gap(samples, seed, scale, base_delta)
        return {
            "scale": scale,
            "gap": gap,
            "male_n": male_n,
            "female_n": female_n,
            "bracketed": False,
        }

    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        gap_mid, _, _ = _evaluate_gap(samples, seed, mid, base_delta)
        if gap_mid > target_gap:
            hi = mid
        else:
            lo = mid

    scale = (lo + hi) / 2.0
    gap, male_n, female_n = _evaluate_gap(samples, seed, scale, base_delta)
    return {
        "scale": scale,
        "gap": gap,
        "male_n": male_n,
        "female_n": female_n,
        "bracketed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-gap", type=float, default=0.17)
    parser.add_argument("--base-delta", type=float, default=0.085)
    parser.add_argument("--min-scale", type=float, default=0.2)
    parser.add_argument("--max-scale", type=float, default=2.0)
    parser.add_argument("--max-iter", type=int, default=18)
    parser.add_argument(
        "--out",
        type=str,
        default="backend/app/data/gender_gap_calibration.json",
    )
    args = parser.parse_args()

    result = calibrate(
        samples=args.samples,
        seed=args.seed,
        target_gap=args.target_gap,
        base_delta=args.base_delta,
        lo=args.min_scale,
        hi=args.max_scale,
        max_iter=args.max_iter,
    )

    scale = result["scale"]
    payload = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "target_gap": args.target_gap,
        "observed_gap": result["gap"],
        "base_delta": args.base_delta,
        "scale": scale,
        "M": 1.0 + args.base_delta * scale,
        "F": 1.0 - args.base_delta * scale,
        "samples": args.samples,
        "seed": args.seed,
        "male_n": result["male_n"],
        "female_n": result["female_n"],
        "bracketed": result["bracketed"],
    }

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    print("Saved gender gap calibration:", out_path)
    print(json.dumps(payload, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
