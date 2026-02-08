"""Score calibration utilities for post-SSR mapping."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class IsotonicCalibrator:
    """Non-decreasing scalar calibrator based on isotonic regression."""

    x_sorted: np.ndarray
    y_fitted: np.ndarray
    clip_min: float = 1.0
    clip_max: float = 5.0

    def transform(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        if self.x_sorted.size == 0:
            return np.clip(arr, self.clip_min, self.clip_max)
        out = np.interp(
            arr,
            self.x_sorted,
            self.y_fitted,
            left=float(self.y_fitted[0]),
            right=float(self.y_fitted[-1]),
        )
        return np.clip(out, self.clip_min, self.clip_max)

    def to_dict(self) -> dict:
        return {
            "type": "isotonic_v1",
            "x_sorted": [float(x) for x in self.x_sorted.tolist()],
            "y_fitted": [float(y) for y in self.y_fitted.tolist()],
            "clip_min": float(self.clip_min),
            "clip_max": float(self.clip_max),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "IsotonicCalibrator":
        if payload.get("type") != "isotonic_v1":
            raise ValueError(f"Unsupported calibrator type: {payload.get('type')}")
        x_sorted = np.asarray(payload.get("x_sorted", []), dtype=float)
        y_fitted = np.asarray(payload.get("y_fitted", []), dtype=float)
        if x_sorted.shape != y_fitted.shape:
            raise ValueError("Invalid calibrator payload: x_sorted and y_fitted shapes differ.")
        return cls(
            x_sorted=x_sorted,
            y_fitted=y_fitted,
            clip_min=float(payload.get("clip_min", 1.0)),
            clip_max=float(payload.get("clip_max", 5.0)),
        )

    def save_json(self, path: Path, *, metadata: dict | None = None) -> None:
        payload = self.to_dict()
        if metadata:
            payload["metadata"] = metadata
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load_json(cls, path: Path) -> "IsotonicCalibrator":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(payload)


@dataclass(frozen=True)
class DomainCalibrationPolicy:
    """Domain-aware calibration policy with per-domain isotonic calibrators."""

    default_domain: str
    calibrators: dict[str, IsotonicCalibrator]

    def _resolve_candidates(self, domain_hint: str | None) -> list[str]:
        if not domain_hint:
            return []
        key = str(domain_hint).strip().lower()
        if not key:
            return []

        candidates: list[str] = [key]
        if key.endswith("_pl") or key.endswith("_en"):
            base = key[:-3]
            if base:
                candidates.append(base)

        if key.startswith("purchase_intent_short"):
            candidates.extend(["purchase_intent", "ecommerce"])
        elif key.startswith("purchase_intent"):
            candidates.append("ecommerce")
        elif key == "ecommerce":
            candidates.append("purchase_intent")
        elif key.startswith("review_long"):
            candidates.append("general")

        # Deduplicate while preserving priority order.
        seen: set[str] = set()
        ordered: list[str] = []
        for candidate in candidates:
            if candidate not in seen:
                seen.add(candidate)
                ordered.append(candidate)
        return ordered

    def select(self, domain_hint: str | None = None) -> IsotonicCalibrator | None:
        if not self.calibrators:
            return None
        for candidate in self._resolve_candidates(domain_hint):
            if candidate in self.calibrators:
                return self.calibrators[candidate]
        return self.calibrators.get(self.default_domain)

    def to_dict(self) -> dict:
        return {
            "type": "domain_calibration_v1",
            "default_domain": self.default_domain,
            "domains": {k: v.to_dict() for k, v in self.calibrators.items()},
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "DomainCalibrationPolicy":
        if payload.get("type") != "domain_calibration_v1":
            raise ValueError(f"Unsupported policy type: {payload.get('type')}")
        domains_raw = payload.get("domains", {})
        calibrators = {
            str(k).strip().lower(): IsotonicCalibrator.from_dict(v)
            for k, v in domains_raw.items()
            if isinstance(v, dict)
        }
        if not calibrators:
            raise ValueError("Domain calibration policy has no calibrators.")
        default_domain = str(payload.get("default_domain", "")).strip().lower() or next(iter(calibrators))
        if default_domain not in calibrators:
            default_domain = next(iter(calibrators))
        return cls(default_domain=default_domain, calibrators=calibrators)

    def save_json(self, path: Path, *, metadata: dict | None = None) -> None:
        payload = self.to_dict()
        if metadata:
            payload["metadata"] = metadata
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load_json(cls, path: Path) -> "DomainCalibrationPolicy":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(payload)


def fit_isotonic_calibrator(scores: np.ndarray, labels: np.ndarray) -> IsotonicCalibrator:
    """Fit non-decreasing mapping y=f(x) with pair-adjacent violators."""
    x = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=float)
    if x.size == 0 or y.size == 0 or x.shape != y.shape:
        raise ValueError("scores and labels must be non-empty arrays of equal shape.")

    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ys = y[order]
    ws = np.ones_like(ys, dtype=float)

    avg: list[float] = []
    w_sum: list[float] = []
    x_sum: list[float] = []
    count: list[int] = []
    for xi, yi, wi in zip(xs, ys, ws):
        avg.append(float(yi))
        w_sum.append(float(wi))
        x_sum.append(float(xi))
        count.append(1)

        while len(avg) >= 2 and avg[-2] > avg[-1]:
            merged_w = w_sum[-2] + w_sum[-1]
            merged_y = (avg[-2] * w_sum[-2] + avg[-1] * w_sum[-1]) / merged_w
            merged_x = x_sum[-2] + x_sum[-1]
            merged_count = count[-2] + count[-1]
            avg[-2] = float(merged_y)
            w_sum[-2] = float(merged_w)
            x_sum[-2] = float(merged_x)
            count[-2] = int(merged_count)
            avg.pop()
            w_sum.pop()
            x_sum.pop()
            count.pop()

    y_fit = np.empty_like(ys)
    x_fit = np.empty_like(xs)
    pos = 0
    for block_avg, block_x_sum, block_count in zip(avg, x_sum, count):
        block_mean_x = block_x_sum / block_count
        y_fit[pos : pos + block_count] = block_avg
        x_fit[pos : pos + block_count] = block_mean_x
        pos += block_count

    return IsotonicCalibrator(x_sorted=x_fit, y_fitted=y_fit)
