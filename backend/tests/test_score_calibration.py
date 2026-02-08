"""Tests for score calibration utilities."""

from pathlib import Path

import numpy as np

from app.services.score_calibration import (
    DomainCalibrationPolicy,
    EntropyAwareCalibrator,
    IsotonicCalibrator,
    Piecewise3IsotonicCalibrator,
    PiecewiseIsotonicCalibrator,
    TrustRegionCalibrator,
    fit_isotonic_calibrator,
    fit_piecewise3_isotonic_calibrator,
    fit_piecewise_isotonic_calibrator,
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


def test_piecewise_fit_and_json_roundtrip() -> None:
    scores = np.array([1.0, 1.5, 2.0, 2.4, 2.8, 3.2, 3.6, 4.2, 4.8], dtype=float)
    labels = np.array([1.1, 1.2, 1.7, 2.0, 2.2, 3.5, 3.8, 4.3, 4.8], dtype=float)
    cal = fit_piecewise_isotonic_calibrator(scores, labels, split_quantile=0.6)

    x = np.array([1.2, 2.7, 3.1, 4.6], dtype=float)
    y = cal.transform(x)
    assert np.all(np.diff(y) >= -1e-12)
    assert float(y.min()) >= 1.0
    assert float(y.max()) <= 5.0

    payload = cal.to_dict()
    loaded = PiecewiseIsotonicCalibrator.from_dict(payload)
    assert np.allclose(cal.transform(x), loaded.transform(x))


def test_piecewise3_fit_and_json_roundtrip() -> None:
    scores = np.array(
        [1.0, 1.1, 1.4, 1.8, 2.1, 2.5, 2.9, 3.2, 3.5, 3.9, 4.2, 4.6, 4.9],
        dtype=float,
    )
    labels = np.array(
        [1.0, 1.1, 1.2, 1.5, 1.9, 2.4, 2.8, 3.2, 3.6, 4.0, 4.2, 4.5, 4.9],
        dtype=float,
    )
    cal = fit_piecewise3_isotonic_calibrator(scores, labels, split_quantiles=(0.35, 0.72))
    x = np.array([1.2, 2.0, 2.8, 3.6, 4.7], dtype=float)
    y = cal.transform(x)
    assert np.all(np.diff(y) >= -1e-12)
    assert float(y.min()) >= 1.0
    assert float(y.max()) <= 5.0

    payload = cal.to_dict()
    loaded = Piecewise3IsotonicCalibrator.from_dict(payload)
    assert np.allclose(cal.transform(x), loaded.transform(x))


def test_trust_region_limits_shift_and_roundtrip() -> None:
    base = IsotonicCalibrator(
        x_sorted=np.array([1.0, 3.0, 5.0], dtype=float),
        y_fitted=np.array([1.8, 3.8, 4.9], dtype=float),
    )
    tr = TrustRegionCalibrator(base=base, max_delta=0.25)
    x = np.array([1.0, 2.0, 3.0, 4.5], dtype=float)
    y = tr.transform(x)
    assert np.all(y <= x + 0.25 + 1e-12)
    assert np.all(y >= x - 0.25 - 1e-12)

    payload = tr.to_dict()
    loaded = TrustRegionCalibrator.from_dict(payload)
    assert np.allclose(tr.transform(x), loaded.transform(x))


def test_entropy_aware_calibrator_switches_by_uncertainty() -> None:
    low = IsotonicCalibrator(
        x_sorted=np.array([1.0, 5.0], dtype=float),
        y_fitted=np.array([1.0, 5.0], dtype=float),
    )
    high = IsotonicCalibrator(
        x_sorted=np.array([1.0, 5.0], dtype=float),
        y_fitted=np.array([1.4, 4.6], dtype=float),
    )
    cal = EntropyAwareCalibrator(entropy_threshold=0.5, low_entropy=low, high_entropy=high)
    x = np.array([2.0, 2.0], dtype=float)
    u = np.array([0.4, 0.7], dtype=float)
    y = cal.transform(x, uncertainty=u)
    assert y[0] != y[1]
    payload = cal.to_dict()
    loaded = EntropyAwareCalibrator.from_dict(payload)
    assert np.allclose(y, loaded.transform(x, uncertainty=u))
