"""Tests for frontend demographic gender mapping."""

import sys
from pathlib import Path

from app.i18n import Language


ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = ROOT / "frontend"
if str(FRONTEND_DIR) not in sys.path:
    sys.path.insert(0, str(FRONTEND_DIR))

import main as frontend_main  # noqa: E402


def test_normalize_gender_value_maps_k_to_f():
    assert frontend_main.normalize_gender_value("K") == "F"


def test_normalize_gender_value_keeps_f_and_m():
    assert frontend_main.normalize_gender_value("F") == "F"
    assert frontend_main.normalize_gender_value("M") == "M"


def test_normalize_gender_value_maps_all_to_none():
    assert frontend_main.normalize_gender_value("Wszystkie") is None
    assert frontend_main.normalize_gender_value("All") is None


def test_normalize_target_audience_uses_normalized_gender():
    result = frontend_main.normalize_target_audience(
        Language.PL,
        age_min=25,
        age_max=45,
        gender="K",
        income_level="Wszystkie",
        location_type="Wszystkie",
        region="Wszystkie regiony",
    )
    assert result["gender"] == "F"


def test_target_audience_to_ui_gender_for_en_and_pl():
    target = {"age_min": 25, "age_max": 45, "gender": "F"}
    _, _, gender_en, _, _, _ = frontend_main.target_audience_to_ui(Language.EN, target)
    _, _, gender_pl, _, _, _ = frontend_main.target_audience_to_ui(Language.PL, target)
    assert gender_pl == "K"
    assert gender_en == "F"


def test_normalize_region_value_maps_all_to_none():
    assert frontend_main.normalize_region_value("Wszystkie regiony") is None
    assert frontend_main.normalize_region_value("All regions") is None


def test_normalize_target_audience_sets_region():
    result = frontend_main.normalize_target_audience(
        Language.EN,
        age_min=30,
        age_max=50,
        gender="F",
        income_level="All",
        location_type="All",
        region="mazowieckie",
    )
    assert result["region"] == "mazowieckie"
