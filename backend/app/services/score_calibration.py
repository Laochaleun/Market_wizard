"""Score calibration utilities for post-SSR mapping."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np


@dataclass(frozen=True)
class IsotonicCalibrator:
    """Non-decreasing scalar calibrator based on isotonic regression."""

    x_sorted: np.ndarray
    y_fitted: np.ndarray
    clip_min: float = 1.0
    clip_max: float = 5.0

    def transform(self, x: np.ndarray, *, uncertainty: np.ndarray | None = None) -> np.ndarray:
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
class PiecewiseIsotonicCalibrator:
    """Two-segment monotonic calibrator with continuity at split point."""

    split_x: float
    lower: IsotonicCalibrator
    upper: IsotonicCalibrator

    def transform(self, x: np.ndarray, *, uncertainty: np.ndarray | None = None) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        if arr.size == 0:
            return arr
        out = np.empty_like(arr, dtype=float)
        lower_mask = arr <= self.split_x
        if np.any(lower_mask):
            out[lower_mask] = self.lower.transform(arr[lower_mask])
        if np.any(~lower_mask):
            out[~lower_mask] = self.upper.transform(arr[~lower_mask])
        return np.clip(out, self.lower.clip_min, self.lower.clip_max)

    def to_dict(self) -> dict:
        return {
            "type": "piecewise_isotonic_v1",
            "split_x": float(self.split_x),
            "lower": self.lower.to_dict(),
            "upper": self.upper.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "PiecewiseIsotonicCalibrator":
        if payload.get("type") != "piecewise_isotonic_v1":
            raise ValueError(f"Unsupported calibrator type: {payload.get('type')}")
        lower = IsotonicCalibrator.from_dict(payload["lower"])
        upper = IsotonicCalibrator.from_dict(payload["upper"])
        return cls(
            split_x=float(payload.get("split_x", 3.0)),
            lower=lower,
            upper=upper,
        )


@dataclass(frozen=True)
class Piecewise3IsotonicCalibrator:
    """Three-segment monotonic calibrator with continuity at split points."""

    split_x1: float
    split_x2: float
    lower: IsotonicCalibrator
    middle: IsotonicCalibrator
    upper: IsotonicCalibrator

    def transform(self, x: np.ndarray, *, uncertainty: np.ndarray | None = None) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        if arr.size == 0:
            return arr
        out = np.empty_like(arr, dtype=float)
        lower_mask = arr <= self.split_x1
        middle_mask = (arr > self.split_x1) & (arr <= self.split_x2)
        upper_mask = arr > self.split_x2
        if np.any(lower_mask):
            out[lower_mask] = self.lower.transform(arr[lower_mask])
        if np.any(middle_mask):
            out[middle_mask] = self.middle.transform(arr[middle_mask])
        if np.any(upper_mask):
            out[upper_mask] = self.upper.transform(arr[upper_mask])
        return np.clip(out, self.lower.clip_min, self.lower.clip_max)

    def to_dict(self) -> dict:
        return {
            "type": "piecewise3_isotonic_v1",
            "split_x1": float(self.split_x1),
            "split_x2": float(self.split_x2),
            "lower": self.lower.to_dict(),
            "middle": self.middle.to_dict(),
            "upper": self.upper.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Piecewise3IsotonicCalibrator":
        if payload.get("type") != "piecewise3_isotonic_v1":
            raise ValueError(f"Unsupported calibrator type: {payload.get('type')}")
        lower = IsotonicCalibrator.from_dict(payload["lower"])
        middle = IsotonicCalibrator.from_dict(payload["middle"])
        upper = IsotonicCalibrator.from_dict(payload["upper"])
        return cls(
            split_x1=float(payload.get("split_x1", 2.8)),
            split_x2=float(payload.get("split_x2", 3.6)),
            lower=lower,
            middle=middle,
            upper=upper,
        )


@dataclass(frozen=True)
class TrustRegionCalibrator:
    """Wraps a base calibrator and limits shift from raw score by max_delta."""

    base: "AnyCalibrator"
    max_delta: float

    def transform(self, x: np.ndarray, *, uncertainty: np.ndarray | None = None) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        base_pred = self.base.transform(arr, uncertainty=uncertainty)
        d = max(0.0, float(self.max_delta))
        lo = arr - d
        hi = arr + d
        return np.clip(base_pred, lo, hi)

    def to_dict(self) -> dict:
        return {
            "type": "trust_region_v1",
            "max_delta": float(self.max_delta),
            "base": self.base.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "TrustRegionCalibrator":
        if payload.get("type") != "trust_region_v1":
            raise ValueError(f"Unsupported calibrator type: {payload.get('type')}")
        return cls(
            base=calibrator_from_dict(payload["base"]),
            max_delta=float(payload.get("max_delta", 0.0)),
        )


@dataclass(frozen=True)
class EntropyAwareCalibrator:
    """Selects calibrator branch by normalized PMF entropy threshold."""

    entropy_threshold: float
    low_entropy: "AnyCalibrator"
    high_entropy: "AnyCalibrator"

    def transform(self, x: np.ndarray, *, uncertainty: np.ndarray | None = None) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        if uncertainty is None:
            return self.low_entropy.transform(arr)
        unc = np.asarray(uncertainty, dtype=float)
        if unc.size == 1 and arr.size > 1:
            unc = np.full(arr.shape, float(unc.ravel()[0]), dtype=float)
        if unc.shape != arr.shape:
            raise ValueError("uncertainty shape must match input scores.")
        out = np.empty_like(arr, dtype=float)
        low_mask = unc <= float(self.entropy_threshold)
        if np.any(low_mask):
            out[low_mask] = self.low_entropy.transform(arr[low_mask], uncertainty=unc[low_mask])
        if np.any(~low_mask):
            out[~low_mask] = self.high_entropy.transform(arr[~low_mask], uncertainty=unc[~low_mask])
        return np.clip(out, 1.0, 5.0)

    def to_dict(self) -> dict:
        return {
            "type": "entropy_aware_v1",
            "entropy_threshold": float(self.entropy_threshold),
            "low_entropy": self.low_entropy.to_dict(),
            "high_entropy": self.high_entropy.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "EntropyAwareCalibrator":
        if payload.get("type") != "entropy_aware_v1":
            raise ValueError(f"Unsupported calibrator type: {payload.get('type')}")
        return cls(
            entropy_threshold=float(payload.get("entropy_threshold", 0.5)),
            low_entropy=calibrator_from_dict(payload["low_entropy"]),
            high_entropy=calibrator_from_dict(payload["high_entropy"]),
        )


AnyCalibrator: TypeAlias = (
    IsotonicCalibrator
    | PiecewiseIsotonicCalibrator
    | Piecewise3IsotonicCalibrator
    | TrustRegionCalibrator
    | EntropyAwareCalibrator
)


def calibrator_from_dict(payload: dict) -> AnyCalibrator:
    ctype = payload.get("type")
    if ctype == "isotonic_v1":
        return IsotonicCalibrator.from_dict(payload)
    if ctype == "piecewise_isotonic_v1":
        return PiecewiseIsotonicCalibrator.from_dict(payload)
    if ctype == "piecewise3_isotonic_v1":
        return Piecewise3IsotonicCalibrator.from_dict(payload)
    if ctype == "trust_region_v1":
        return TrustRegionCalibrator.from_dict(payload)
    if ctype == "entropy_aware_v1":
        return EntropyAwareCalibrator.from_dict(payload)
    raise ValueError(f"Unsupported calibrator type: {ctype}")


@dataclass(frozen=True)
class DomainCalibrationPolicy:
    """Domain-aware calibration policy with per-domain isotonic calibrators."""

    default_domain: str
    calibrators: dict[str, AnyCalibrator]

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

    def select(self, domain_hint: str | None = None) -> AnyCalibrator | None:
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
            str(k).strip().lower(): calibrator_from_dict(v)
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


def fit_piecewise_isotonic_calibrator(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    split_quantile: float = 0.6,
) -> PiecewiseIsotonicCalibrator:
    """Fit a two-segment isotonic mapping with continuity at split."""
    x = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=float)
    if x.size == 0 or y.size == 0 or x.shape != y.shape:
        raise ValueError("scores and labels must be non-empty arrays of equal shape.")
    if x.size < 20:
        iso = fit_isotonic_calibrator(x, y)
        return PiecewiseIsotonicCalibrator(split_x=float(np.median(x)), lower=iso, upper=iso)

    q = float(np.clip(split_quantile, 0.2, 0.8))
    split_x = float(np.quantile(x, q))
    lower_mask = x <= split_x
    upper_mask = x > split_x
    if lower_mask.sum() < 10 or upper_mask.sum() < 10:
        iso = fit_isotonic_calibrator(x, y)
        return PiecewiseIsotonicCalibrator(split_x=split_x, lower=iso, upper=iso)

    lower = fit_isotonic_calibrator(x[lower_mask], y[lower_mask])
    upper = fit_isotonic_calibrator(x[upper_mask], y[upper_mask])

    lower_boundary = float(lower.transform(np.array([split_x], dtype=float))[0])
    upper_boundary = float(upper.transform(np.array([split_x], dtype=float))[0])
    continuity_shift = lower_boundary - upper_boundary
    upper_shifted = IsotonicCalibrator(
        x_sorted=upper.x_sorted.astype(float),
        y_fitted=np.clip(upper.y_fitted.astype(float) + continuity_shift, upper.clip_min, upper.clip_max),
        clip_min=upper.clip_min,
        clip_max=upper.clip_max,
    )
    return PiecewiseIsotonicCalibrator(
        split_x=split_x,
        lower=lower,
        upper=upper_shifted,
    )


def fit_piecewise3_isotonic_calibrator(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    split_quantiles: tuple[float, float] = (0.4, 0.75),
) -> Piecewise3IsotonicCalibrator:
    """Fit a three-segment isotonic mapping with continuity."""
    x = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=float)
    if x.size == 0 or y.size == 0 or x.shape != y.shape:
        raise ValueError("scores and labels must be non-empty arrays of equal shape.")
    if x.size < 30:
        iso = fit_isotonic_calibrator(x, y)
        mid = float(np.median(x))
        return Piecewise3IsotonicCalibrator(split_x1=mid, split_x2=mid, lower=iso, middle=iso, upper=iso)

    q1 = float(np.clip(split_quantiles[0], 0.2, 0.6))
    q2 = float(np.clip(split_quantiles[1], q1 + 0.05, 0.9))
    split_x1 = float(np.quantile(x, q1))
    split_x2 = float(np.quantile(x, q2))

    lower_mask = x <= split_x1
    middle_mask = (x > split_x1) & (x <= split_x2)
    upper_mask = x > split_x2
    if lower_mask.sum() < 10 or middle_mask.sum() < 10 or upper_mask.sum() < 10:
        iso = fit_isotonic_calibrator(x, y)
        return Piecewise3IsotonicCalibrator(
            split_x1=split_x1,
            split_x2=split_x2,
            lower=iso,
            middle=iso,
            upper=iso,
        )

    lower = fit_isotonic_calibrator(x[lower_mask], y[lower_mask])
    middle = fit_isotonic_calibrator(x[middle_mask], y[middle_mask])
    upper = fit_isotonic_calibrator(x[upper_mask], y[upper_mask])

    lower_b = float(lower.transform(np.array([split_x1], dtype=float))[0])
    middle_b1 = float(middle.transform(np.array([split_x1], dtype=float))[0])
    middle_shift = lower_b - middle_b1
    middle_shifted = IsotonicCalibrator(
        x_sorted=middle.x_sorted.astype(float),
        y_fitted=np.clip(middle.y_fitted.astype(float) + middle_shift, middle.clip_min, middle.clip_max),
        clip_min=middle.clip_min,
        clip_max=middle.clip_max,
    )

    middle_b2 = float(middle_shifted.transform(np.array([split_x2], dtype=float))[0])
    upper_b = float(upper.transform(np.array([split_x2], dtype=float))[0])
    upper_shift = middle_b2 - upper_b
    upper_shifted = IsotonicCalibrator(
        x_sorted=upper.x_sorted.astype(float),
        y_fitted=np.clip(upper.y_fitted.astype(float) + upper_shift, upper.clip_min, upper.clip_max),
        clip_min=upper.clip_min,
        clip_max=upper.clip_max,
    )

    return Piecewise3IsotonicCalibrator(
        split_x1=split_x1,
        split_x2=split_x2,
        lower=lower,
        middle=middle_shifted,
        upper=upper_shifted,
    )
