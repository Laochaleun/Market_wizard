"""Tests for score calibration utilities."""

from pathlib import Path

import numpy as np

from app.services.score_calibration import (
    DomainCalibrationPolicy,
    IsotonicCalibrator,
    fit_isotonic_calibrator,
)


def test_isotonic_fit_is_monotonic_and_clipped() -> None:
    scores = np.array([1.0, 2.0, 2.5, 3.0, 4.0, 5.0], dtype=float)
    labels = np.array([1.0, 1.5, 2.0, 3.5, 4.0, 5.0], dtype=float)
    cal = fit_isotonic_calibrator(scores, labels)

    x_grid = np.array([0.5, 1.0, 2.2, 3.7, 5.0, 5.5], dtype=float)
    y = cal.transform(x_grid)
    assert np.all(np.diff(y) >= -1e-12)
    assert float(y.min()) >= 1.0
    assert float(y.max()) <= 5.0


def test_isotonic_json_roundtrip(tmp_path: Path) -> None:
    cal = IsotonicCalibrator(
        x_sorted=np.array([1.0, 2.0, 3.0], dtype=float),
        y_fitted=np.array([1.2, 2.1, 4.7], dtype=float),
    )
    out = tmp_path / "calibrator.json"
    cal.save_json(out, metadata={"note": "test"})

    loaded = IsotonicCalibrator.load_json(out)
    x = np.array([1.5, 2.5], dtype=float)
    assert np.allclose(cal.transform(x), loaded.transform(x))


def test_domain_policy_select_and_roundtrip(tmp_path: Path) -> None:
    general = IsotonicCalibrator(
        x_sorted=np.array([1.0, 5.0], dtype=float),
        y_fitted=np.array([1.0, 5.0], dtype=float),
    )
    ecommerce = IsotonicCalibrator(
        x_sorted=np.array([1.0, 5.0], dtype=float),
        y_fitted=np.array([2.0, 4.0], dtype=float),
    )
    policy = DomainCalibrationPolicy(
        default_domain="general",
        calibrators={"general": general, "ecommerce": ecommerce},
    )
    out = tmp_path / "policy.json"
    policy.save_json(out, metadata={"note": "test"})

    loaded = DomainCalibrationPolicy.load_json(out)
    x = np.array([1.0], dtype=float)
    assert np.allclose(loaded.select("general").transform(x), np.array([1.0]))
    assert np.allclose(loaded.select("ecommerce").transform(x), np.array([2.0]))
    assert np.allclose(loaded.select("purchase_intent").transform(x), np.array([2.0]))
    assert np.allclose(loaded.select("purchase_intent_short_en").transform(x), np.array([2.0]))
    assert np.allclose(loaded.select("review_long_en").transform(x), np.array([1.0]))
    assert loaded.select("unknown") is not None


def test_domain_policy_prefers_typed_and_language_specific_keys() -> None:
    general = IsotonicCalibrator(
        x_sorted=np.array([1.0, 5.0], dtype=float),
        y_fitted=np.array([1.0, 5.0], dtype=float),
    )
    short_en = IsotonicCalibrator(
        x_sorted=np.array([1.0, 5.0], dtype=float),
        y_fitted=np.array([1.5, 4.5], dtype=float),
    )
    short_pl = IsotonicCalibrator(
        x_sorted=np.array([1.0, 5.0], dtype=float),
        y_fitted=np.array([1.7, 4.3], dtype=float),
    )
    policy = DomainCalibrationPolicy(
        default_domain="general",
        calibrators={
            "general": general,
            "purchase_intent_short_en": short_en,
            "purchase_intent_short_pl": short_pl,
        },
    )
    x = np.array([1.0], dtype=float)
    assert np.allclose(policy.select("purchase_intent_short_en").transform(x), np.array([1.5]))
    assert np.allclose(policy.select("purchase_intent_short_pl").transform(x), np.array([1.7]))
    # Fallback to general when typed key does not exist.
    assert np.allclose(policy.select("review_long_en").transform(x), np.array([1.0]))
