"""
Persona Manager - Generation of synthetic consumer personas.

Supports demographic stratification and optional integration with 
GUS (Główny Urząd Statystyczny) BDL API for real Polish demographics.
"""

import json
import logging
import random
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import httpx

from app.config import get_settings
from app.models import DemographicProfile, Persona
from app.i18n import Language, FIRST_NAMES, LOCATIONS, OCCUPATIONS
from app.data.reference_data import (
    OCCUPATION_INCOME_DATA,
    OCCUPATION_POPULATION_WEIGHTS,
    OCCUPATION_GENDER_WEIGHTS,
    OCCUPATION_MIN_EDUCATION,
    OVERQUALIFICATION_RATE,
    REGIONAL_WAGE_INDEX,
    GENDER_WAGE_GAP,
    LOCATION_WAGE_INDEX,
    LOCATION_POPULATION_WEIGHTS,
    PENSION_BY_GENDER,
    CITY_TO_REGION,
    RETIREMENT_AGE_BY_GENDER,
    get_regional_multiplier,
    get_gender_distribution_for_age,
    get_education_distribution_for_age,
    get_marital_status_distribution_for_age,
    get_has_children_probability,
    get_occupation_gender_weight,
    get_min_education_for_occupation,
)

# =============================================================================
# Income distribution by level (PLN netto/month for PL)
# Based on GUS "Rozkład wynagrodzeń w gospodarce narodowej" grudzień 2024
# Decyl 1: do 4242 brutto (~3080 netto)
# Mediana: 7267 brutto (~5280 netto)
# Decyl 9: od 12500 brutto (~9100 netto)
# =============================================================================
INCOME_RANGES = {
    Language.PL: {
        "very_low": (1500, 3500),   # Dolny kwintyl - do 3500 netto
        "low": (3500, 5000),         # Poniżej mediany
        "medium": (5000, 7000),      # Około mediany (5280)
        "high": (7000, 10000),       # Powyżej mediany
        "very_high": (10000, 25000), # Górny kwintyl - top earners
    },
    Language.EN: {
        "very_low": (1500, 3500),
        "low": (3500, 5000),
        "medium": (5000, 7000),
        "high": (7000, 10000),
        "very_high": (10000, 25000),
    },
}

# Education levels (ISCED classification, GUS NSP 2021)
EDUCATION_LEVELS = {
    Language.PL: [
        "podstawowe",
        "zasadnicze zawodowe",
        "średnie",
        "policealne",
        "wyższe",
    ],
    Language.EN: [
        "primary",
        "vocational",
        "secondary",
        "post-secondary",
        "higher",
    ],
}

# Mapping between PL and EN education levels for bilingual filter support
EDUCATION_PL_TO_EN = {
    "podstawowe": "primary",
    "zasadnicze zawodowe": "vocational",
    "średnie": "secondary",
    "policealne": "post-secondary",
    "wyższe": "higher",
}
EDUCATION_EN_TO_PL = {v: k for k, v in EDUCATION_PL_TO_EN.items()}


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

OCCUPATION_EN_TO_PL = {
    "manager": "menedżer",
    "director": "dyrektor",
    "entrepreneur": "przedsiębiorca",
    "doctor": "lekarz",
    "dentist": "dentysta",
    "lawyer": "prawnik",
    "architect": "architekt",
    "pharmacist": "farmaceuta",
    "software developer": "programista",
    "engineer": "inżynier",
    "teacher": "nauczyciel",
    "accountant": "księgowy",
    "graphic designer": "grafik",
    "nurse": "pielęgniarka",
    "technician": "technik",
    "office worker": "pracownik biurowy",
    "secretary": "sekretarka",
    "sales associate": "sprzedawca",
    "hairdresser": "fryzjer",
    "waiter": "kelner",
    "chef": "kucharz",
    "police officer": "policjant",
    "firefighter": "strażak",
    "security guard": "ochroniarz",
    "farmer": "rolnik",
    "mechanic": "mechanik",
    "electrician": "elektryk",
    "construction worker": "pracownik budowlany",
    "carpenter": "stolarz",
    "welder": "spawacz",
    "driver": "kierowca",
    "production operator": "operator produkcji",
    "warehouse worker": "magazynier",
    "cleaner": "sprzątaczka",
    "student": "student",
    "retiree": "emeryt",
    "disability pensioner": "rencista",
    "unemployed": "bezrobotny",
}

_missing_en = {
    occ.get("name", "")
    for occ in OCCUPATIONS.get(Language.EN, [])
    if occ.get("name", "") not in OCCUPATION_EN_TO_PL
}
if _missing_en:
    raise ValueError(f"Missing EN->PL occupation mapping for: {sorted(_missing_en)}")


class GUSClient:
    """Client for GUS BDL API to fetch demographic data (with caching)."""

    BASE_URL = "https://bdl.stat.gov.pl/api/v1"
    
    # Default age distribution based on Polish census data 2023
    # Used as fallback when API is unavailable
    DEFAULT_AGE_DISTRIBUTION = {
        "18-24": 2_800_000,
        "25-34": 5_100_000,
        "35-44": 6_200_000,
        "45-54": 5_100_000,
        "55-64": 4_800_000,
        "65-74": 4_100_000,
        "75+": 2_900_000,
    }
    
    # Default income distribution by age (PLN/month net)
    INCOME_BY_AGE = {
        "18-24": {"mean": 3500, "std": 800},
        "25-34": {"mean": 5500, "std": 1500},
        "35-44": {"mean": 7000, "std": 2500},
        "45-54": {"mean": 6500, "std": 2000},
        "55-64": {"mean": 5500, "std": 1500},
        "65-74": {"mean": 3500, "std": 1000},
        "75+": {"mean": 2800, "std": 600},
    }
    
    # Location distribution based on GUS 2024 data (fallback)
    LOCATION_DISTRIBUTION = LOCATION_POPULATION_WEIGHTS
    
    # Gender distribution (approximate 51.5% F, 48.5% M in Poland)
    GENDER_DISTRIBUTION = {"M": 0.485, "F": 0.515}

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or get_settings().gus_api_key
        self.base_url = get_settings().gus_api_base_url.rstrip("/")
        self.headers = {"Accept": "application/json"}
        if self.api_key:
            self.headers["X-ClientId"] = self.api_key
        self.cache_ttl = timedelta(hours=get_settings().gus_cache_ttl_hours)
        self.unit_id_poland = get_settings().gus_unit_id_poland
        self._live_loaded = False
        self._live_age_groups: list[tuple[int, int | None, int]] = []
        self._live_gender_distribution: dict[str, float] | None = None
        self._live_location_distribution: dict[str, float] | None = None
        self._live_income_mean: float | None = None
        self._cache_path = Path(__file__).resolve().parents[2] / "data" / "gus_cache.json"
        self._income_net_ratio = max(0.5, min(1.0, get_settings().gus_income_net_ratio))

    def _request(self, path: str, params: dict | None = None) -> dict:
        url = f"{self.base_url}{path}"
        resp = httpx.get(url, params=params, headers=self.headers, timeout=15.0)
        resp.raise_for_status()
        return resp.json()

    def _get_items(self, payload: dict) -> list[dict]:
        if isinstance(payload, dict):
            if "results" in payload and isinstance(payload["results"], list):
                return payload["results"]
            if "data" in payload and isinstance(payload["data"], list):
                return payload["data"]
        return []

    def _cache_valid(self) -> bool:
        if not self._cache_path.exists():
            return False
        try:
            raw = json.loads(self._cache_path.read_text(encoding="utf-8"))
            ts = raw.get("timestamp")
            if not ts:
                return False
            cached_at = datetime.fromisoformat(ts)
        except Exception:
            return False
        return datetime.now() - cached_at <= self.cache_ttl

    def _load_cache(self) -> bool:
        if not self._cache_valid():
            return False
        try:
            raw = json.loads(self._cache_path.read_text(encoding="utf-8"))
            self._live_age_groups = raw.get("age_groups", [])
            self._live_gender_distribution = raw.get("gender_distribution")
            self._live_location_distribution = raw.get("location_distribution")
            self._live_income_mean = raw.get("income_mean")
            return True
        except Exception:
            return False

    def _save_cache(self) -> None:
        data = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "age_groups": self._live_age_groups,
            "gender_distribution": self._live_gender_distribution,
            "location_distribution": self._live_location_distribution,
            "income_mean": self._live_income_mean,
        }
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")

    def _ensure_live_loaded(self) -> None:
        if self._live_loaded:
            return
        settings = get_settings()
        if not settings.gus_use_live:
            logger.info("GUS live disabled; using default distributions.")
            self._live_loaded = True
            return
        if self._load_cache():
            logger.info("GUS live data loaded from cache.")
            if self._live_income_mean:
                logger.info("GUS income mean loaded: %.2f", self._live_income_mean)
            self._live_loaded = True
            return
        try:
            self._fetch_live_data()
            self._save_cache()
            logger.info("GUS live data fetched and cached.")
            if self._live_income_mean:
                logger.info("GUS income mean loaded: %.2f", self._live_income_mean)
        except Exception:
            # Fall back to defaults silently
            logger.info("GUS live fetch failed; using default distributions.")
            pass
        self._live_loaded = True

    def _find_subject_id(self, query: str, must_have: list[str] | None = None) -> str | None:
        payload = self._request("/subjects/search", {"name": query})
        items = self._get_items(payload)
        if must_have:
            for item in items:
                name = (item.get("name") or "").lower()
                if all(token in name for token in must_have):
                    return item.get("id")
        return items[0].get("id") if items else None

    def _list_variables(self, subject_id: str) -> list[dict]:
        variables: list[dict] = []
        page = 0
        while True:
            payload = self._request(
                "/variables",
                {"subject-id": subject_id, "page": page, "pageSize": 100},
            )
            items = self._get_items(payload)
            if not items:
                break
            variables.extend(items)
            page += 1
        return variables

    def _variable_label(self, item: dict) -> str:
        if item.get("name"):
            return str(item["name"])
        parts = [item.get(k) for k in ["n1", "n2", "n3", "n4", "n5"] if item.get(k)]
        return " ".join([str(p) for p in parts])

    def _fetch_data_for_variables(self, variable_ids: list[str]) -> list[dict]:
        data: list[dict] = []
        for var_id in variable_ids:
            payload = self._request(f"/data/by-unit/{self.unit_id_poland}", {"var-id": var_id})
            data.extend(self._get_items(payload))
        # Keep only latest year per variable if year info present
        by_var: dict[str, dict] = {}
        for item in data:
            var_id = str(item.get("variableId") or item.get("id") or "")
            year = item.get("year") or item.get("yearId") or 0
            try:
                year = int(year)
            except Exception:
                year = 0
            prev = by_var.get(var_id)
            if not prev or year >= int(prev.get("year") or prev.get("yearId") or 0):
                by_var[var_id] = item
        return list(by_var.values())
        return data

    def _parse_age_groups(self, variables: list[dict]) -> list[tuple[int, int | None, int]]:
        age_groups: list[tuple[int, int | None, int]] = []
        variable_ids = []
        var_map = {}
        for item in variables:
            label = self._variable_label(item)
            if "wiek" not in label.lower():
                continue
            if "ogółem" not in label.lower():
                continue
            var_id = str(item.get("id"))
            variable_ids.append(var_id)
            var_map[var_id] = label

        if not variable_ids:
            return age_groups

        data = self._fetch_data_for_variables(variable_ids)
        for item in data:
            var_id = str(item.get("variableId") or item.get("id") or "")
            value = item.get("val") or item.get("value")
            if value is None:
                continue
            label = var_map.get(var_id, "")
            match = re.search(r"(\\d{1,2})\\s*[–-]\\s*(\\d{1,2})", label)
            if match:
                min_age = int(match.group(1))
                max_age = int(match.group(2))
                age_groups.append((min_age, max_age, int(value)))
                continue
            if "i więcej" in label.lower() or "+" in label:
                match = re.search(r"(\\d{1,2})", label)
                if match:
                    min_age = int(match.group(1))
                    age_groups.append((min_age, None, int(value)))

        return age_groups

    def _parse_gender_distribution(self, variables: list[dict]) -> dict[str, float] | None:
        variable_ids = []
        var_map = {}
        for item in variables:
            label = self._variable_label(item).lower()
            if "płeć" not in label and "plec" not in label:
                continue
            if "ogółem" in label:
                continue
            if "mężczyźni" in label or "kobiety" in label:
                var_id = str(item.get("id"))
                variable_ids.append(var_id)
                var_map[var_id] = label

        if not variable_ids:
            return None

        data = self._fetch_data_for_variables(variable_ids)
        male = female = 0
        for item in data:
            var_id = str(item.get("variableId") or item.get("id") or "")
            value = item.get("val") or item.get("value") or 0
            label = var_map.get(var_id, "")
            if "mężczyźni" in label:
                male += int(value)
            elif "kobiety" in label:
                female += int(value)
        total = male + female
        if total <= 0:
            return None
        return {"M": male / total, "F": female / total}

    def _parse_location_distribution(self, variables: list[dict]) -> dict[str, float] | None:
        variable_ids = []
        var_map = {}
        for item in variables:
            label = self._variable_label(item).lower()
            if "miejsce zamieszkania" not in label and "miasto" not in label and "wieś" not in label:
                continue
            if "miasto" in label or "wieś" in label:
                var_id = str(item.get("id"))
                variable_ids.append(var_id)
                var_map[var_id] = label

        if not variable_ids:
            return None

        data = self._fetch_data_for_variables(variable_ids)
        urban = rural = 0
        for item in data:
            var_id = str(item.get("variableId") or item.get("id") or "")
            value = item.get("val") or item.get("value") or 0
            label = var_map.get(var_id, "")
            if "miasto" in label:
                urban += int(value)
            elif "wieś" in label:
                rural += int(value)
        total = urban + rural
        if total <= 0:
            return None
        return {"urban": urban / total, "rural": rural / total, "suburban": 0.0}

    def _parse_income_mean(self, variables: list[dict]) -> float | None:
        variable_ids = []
        var_map = {}
        for item in variables:
            label = self._variable_label(item).lower()
            if "wynagrod" not in label:
                continue
            if "przeciętne miesięczne" not in label and "przecietne miesieczne" not in label:
                continue
            var_id = str(item.get("id"))
            variable_ids.append(var_id)
            var_map[var_id] = label

        if not variable_ids:
            return None

        data = self._fetch_data_for_variables(variable_ids)
        values = []
        for item in data:
            value = item.get("val") or item.get("value")
            if value is None:
                continue
            try:
                values.append(float(value))
            except Exception:
                continue
        if not values:
            return None
        return float(max(values))

    def _fetch_live_data(self) -> None:
        subject_id = self._find_subject_id("ludność", must_have=["ludno", "wiek"])
        if subject_id:
            variables = self._list_variables(subject_id)
            self._live_age_groups = self._parse_age_groups(variables) or self._live_age_groups
            self._live_gender_distribution = self._parse_gender_distribution(variables) or self._live_gender_distribution

        location_subject_id = self._find_subject_id("miejsce zamieszkania", must_have=["miejsce", "zamieszkania"])
        if location_subject_id:
            variables = self._list_variables(location_subject_id)
            self._live_location_distribution = self._parse_location_distribution(variables) or self._live_location_distribution

        income_subject_id = self._find_subject_id("wynagrodzenia", must_have=["wynagrod"])
        if income_subject_id:
            variables = self._list_variables(income_subject_id)
            self._live_income_mean = self._parse_income_mean(variables) or self._live_income_mean

    def get_age_distribution(self) -> dict[str, int]:
        """
        Get population by age groups.
        Returns cached/default data (API calls are done sync at startup if needed).
        """
        self._ensure_live_loaded()
        if self._live_age_groups:
            # Return as string labels for compatibility
            out: dict[str, int] = {}
            for min_age, max_age, count in self._live_age_groups:
                label = f"{min_age}-{max_age}" if max_age is not None else f"{min_age}+"
                out[label] = count
            return out
        return self.DEFAULT_AGE_DISTRIBUTION.copy()
    
    def get_income_for_age(self, age: int) -> int:
        """Get realistic income based on age from GUS wage statistics."""
        self._ensure_live_loaded()
        age_group = self._get_age_group(age)
        params = self.INCOME_BY_AGE.get(age_group, {"mean": 5000, "std": 1500})

        # Optionally scale mean/std using live GUS average wage (gross).
        mean = params["mean"]
        std = params["std"]
        if self._live_income_mean:
            # Scale by ratio vs default overall mean to keep age profile shape.
            default_means = [v["mean"] for v in self.INCOME_BY_AGE[Language.PL].values()]
            default_base = sum(default_means) / len(default_means)
            ratio = self._live_income_mean / default_base if default_base else 1.0
            mean = mean * ratio * self._income_net_ratio
            std = std * ratio * self._income_net_ratio

        # Sample from normal distribution
        income = random.gauss(mean, std)
        # Clamp to reasonable range
        income = max(2000, min(25000, income))
        return int(round(income, -2))  # Round to nearest 100
    
    def _get_age_group(self, age: int) -> str:
        """Map age to age group."""
        if age < 25:
            return "18-24"
        elif age < 35:
            return "25-34"
        elif age < 45:
            return "35-44"
        elif age < 55:
            return "45-54"
        elif age < 65:
            return "55-64"
        elif age < 75:
            return "65-74"
        else:
            return "75+"
    
    def sample_age(self) -> int:
        """Sample age based on population distribution."""
        self._ensure_live_loaded()
        if self._live_age_groups:
            total = sum(count for _, _, count in self._live_age_groups)
            if total > 0:
                weights = [count / total for _, _, count in self._live_age_groups]
                selected = random.choices(self._live_age_groups, weights=weights)[0]
                min_age, max_age, _ = selected
                if max_age is None:
                    max_age = min_age + 15
                return random.randint(min_age, max_age)

        age_dist = self.get_age_distribution()
        total = sum(age_dist.values())
        weights = [v / total for v in age_dist.values()]
        
        age_groups = list(age_dist.keys())
        selected_group = random.choices(age_groups, weights=weights)[0]
        
        # Parse age range and sample uniformly within
        if "+" in selected_group:
            min_age = int(selected_group.replace("+", ""))
            max_age = min_age + 15  # e.g., 75+ -> 75-90
        else:
            parts = selected_group.split("-")
            min_age, max_age = int(parts[0]), int(parts[1])
        
        return random.randint(min_age, max_age)
    
    def sample_gender(self, age: int | None = None) -> str:
        """
        Sample gender based on age-dependent population distribution.
        
        Uses GUS 2024 demographic data - women live longer, so gender
        distribution changes with age (e.g., 80+: 30% M, 70% F).
        
        Args:
            age: Person's age. If None, uses average distribution.
        """
        self._ensure_live_loaded()
        
        # If age provided, use age-specific distribution from reference data
        if age is not None:
            gender_dist = get_gender_distribution_for_age(age)
            return random.choices(
                list(gender_dist.keys()),
                weights=list(gender_dist.values()),
            )[0]
        
        # Fallback: use live GUS data if available
        if self._live_gender_distribution:
            return random.choices(
                list(self._live_gender_distribution.keys()),
                weights=list(self._live_gender_distribution.values()),
            )[0]
        
        # Default average distribution
        return random.choices(
            list(self.GENDER_DISTRIBUTION.keys()),
            weights=list(self.GENDER_DISTRIBUTION.values())
        )[0]
    
    def sample_location_type(self) -> str:
        """Sample location type based on urbanization statistics."""
        self._ensure_live_loaded()
        if self._live_location_distribution:
            return random.choices(
                list(self._live_location_distribution.keys()),
                weights=list(self._live_location_distribution.values()),
            )[0]
        return random.choices(
            list(self.LOCATION_DISTRIBUTION.keys()),
            weights=list(self.LOCATION_DISTRIBUTION.values())
        )[0]


class PersonaManager:
    """
    Manages generation of synthetic consumer personas.
    
    Supports:
    - Random generation within demographic constraints
    - Stratified sampling based on target audience
    - GUS-based realistic demographics for Polish market
    - Multilingual support (PL/EN)
    """

    def __init__(
        self, 
        language: Language = Language.PL,
        use_gus_demographics: bool = True,
    ):
        self.language = language
        self.use_gus_demographics = use_gus_demographics
        
        if self.use_gus_demographics:
            self.gus_client = GUSClient()

    def _normalize_region(self, region: str | None) -> str | None:
        value = (region or "").strip().lower()
        return value or None

    def _location_matches_region(self, location: str, region: str) -> bool:
        mapped = CITY_TO_REGION.get(location)
        if mapped:
            return mapped.lower() == region

        # Rural labels are stored as phrases (e.g., "wieś na Mazowszu").
        rural_region_markers = {
            "mazows": "mazowieckie",
            "małopols": "małopolskie",
            "podkarpac": "podkarpackie",
            "śląsk": "śląskie",
            "wielkopol": "wielkopolskie",
            "pomorz": "pomorskie",
            "warm": "warmińsko-mazurskie",
            "podlasi": "podlaskie",
            "świętokrz": "świętokrzyskie",
            "lubelszcz": "lubelskie",
            "łódzk": "łódzkie",
            "dolnym ślą": "dolnośląskie",
        }
        lowered = location.lower()
        for marker, marker_region in rural_region_markers.items():
            if marker in lowered:
                return marker_region == region
        return False

    def _normalize_education(self, education: str | None) -> str | None:
        """Normalize education string to match target language."""
        if not education:
            return None
            
        edu_lower = education.lower()
        
        if self.language == Language.PL:
            # If already PL key
            if edu_lower in EDUCATION_PL_TO_EN:
                return edu_lower
            # If EN key, translate to PL
            if edu_lower in EDUCATION_EN_TO_PL:
                return EDUCATION_EN_TO_PL[edu_lower]
                
        else: # EN
            # If already EN key (which are values in PL_TO_EN dict, keys in EN_TO_PL)
            if edu_lower in EDUCATION_EN_TO_PL:
                return edu_lower
            # If PL key, translate to EN
            if edu_lower in EDUCATION_PL_TO_EN:
                return EDUCATION_PL_TO_EN[edu_lower]
                
        # Fallback: return as is
        return education

    def _education_to_pl_key(self, education: str | None) -> str | None:
        """Normalize education to PL key for internal comparisons."""
        if not education:
            return None
        edu_lower = education.lower()
        if edu_lower in EDUCATION_EN_TO_PL:
            return EDUCATION_EN_TO_PL[edu_lower]
        if edu_lower in EDUCATION_PL_TO_EN:
            return edu_lower
        return edu_lower

    def _education_meets_min(self, education: str, min_education: str) -> bool:
        hierarchy = ["podstawowe", "zasadnicze zawodowe", "średnie", "policealne", "wyższe"]
        if education not in hierarchy or min_education not in hierarchy:
            return True
        return hierarchy.index(education) >= hierarchy.index(min_education)

    def _occupation_to_pl(self, occupation: str | None) -> str | None:
        if not occupation:
            return None
        if self.language == Language.PL:
            return occupation
        return OCCUPATION_EN_TO_PL.get(occupation, occupation)

    def _pick_location(self, lang: Language, location_type: str, region: str | None) -> str:
        candidates = list(LOCATIONS[lang][location_type])
        if not region:
            return random.choice(candidates)
        regional_candidates = [c for c in candidates if self._location_matches_region(c, region)]
        if regional_candidates:
            return random.choice(regional_candidates)
        return random.choice(candidates)

    def generate_persona(
        self,
        profile: DemographicProfile | None = None,
        index: int = 0,
    ) -> Persona:
        """Generate a single persona within demographic constraints."""
        profile = profile or DemographicProfile()
        lang = self.language

        # Use GUS demographics for PL, or random for EN
        if self.use_gus_demographics:
            return self._generate_persona_gus(profile, index)
        else:
            return self._generate_persona_random(profile, index)
    
    def _generate_persona_gus(
        self,
        profile: DemographicProfile,
        index: int,
    ) -> Persona:
        """Generate persona using GUS demographic data."""
        lang = self.language
        gus = self.gus_client
        
        # IMPORTANT: Determine age FIRST (needed for gender distribution)
        if profile.age_min != 18 or profile.age_max != 80:
            # Custom age range specified
            age = random.randint(profile.age_min, profile.age_max)
        else:
            age = gus.sample_age()
            # Clamp to profile constraints
            age = max(profile.age_min, min(profile.age_max, age))

        # Determine gender based on age (women live longer - GUS 2024)
        if profile.gender:
            gender = profile.gender
        else:
            gender = gus.sample_gender(age)  # Age-dependent distribution

        # Determine location type (use GUS urbanization stats if not specified)
        if profile.location_type:
            location_type = profile.location_type
        else:
            location_type = gus.sample_location_type()
        region = self._normalize_region(profile.region)

        # Get location from i18n data with optional region filter.
        location = self._pick_location(lang, location_type, region)

        # Select occupation appropriate for age (using population weights)
        occupation_data = self._select_occupation_for_age(
            age,
            gender,
            education_filter=profile.education,
        )
        occupation = occupation_data["name"]

        # Determine income based on occupation and experience
        if profile.income_level:
            income_ranges = INCOME_RANGES[lang]
            income_range = income_ranges[profile.income_level]
            income = random.randint(income_range[0], income_range[1])
        else:
            # Use GUS 2024 based income with all modifiers
            income = self._calculate_income(
                age,
                occupation_data,
                gender,
                location_type,
                location,
                region=region,
            )

        # Generate name from i18n data
        name = random.choice(FIRST_NAMES[lang][gender])
        
        # Select education: use filter if provided, otherwise age-based distribution
        if profile.education:
            education = self._normalize_education(profile.education)
        else:
            raw_edu = self._select_education_for_age(age, occupation)
            education = self._normalize_education(raw_edu)
        
        # Select marital status based on age (GUS NSP 2021)
        marital_status = self._select_marital_status_for_age(age)
        
        # Determine if has children (GUS 2024)
        has_children = self._select_has_children(age)

        return Persona(
            name=f"{name}_{index}",
            age=age,
            gender=gender,
            income=income,
            location=location,
            location_type=location_type,
            education=education,
            occupation=occupation,
            marital_status=marital_status,
            has_children=has_children,
        )

    def _generate_persona_random(
        self,
        profile: DemographicProfile,
        index: int,
    ) -> Persona:
        """Generate persona using random sampling (for non-PL or when GUS disabled)."""
        lang = self.language

        # IMPORTANT: Determine age FIRST (needed for gender distribution)
        age = random.randint(profile.age_min, profile.age_max)

        # Determine gender based on age (women live longer - GUS 2024)
        if profile.gender:
            gender = profile.gender
        else:
            gender_dist = get_gender_distribution_for_age(age)
            gender = random.choices(
                list(gender_dist.keys()),
                weights=list(gender_dist.values()),
            )[0]

        # Determine location type
        if profile.location_type:
            location_type = profile.location_type
        else:
            # Use diverse distribution for random generation
            location_type = random.choices(
                ["metropolis", "large_city", "medium_city", "small_city", "rural"],
                weights=[0.15, 0.20, 0.20, 0.20, 0.25],
            )[0]
        region = self._normalize_region(profile.region)

        # Determine location with optional region filter.
        location = self._pick_location(lang, location_type, region)

        # Select occupation appropriate for age (using population weights)
        occupation_data = self._select_occupation_for_age(
            age,
            gender,
            education_filter=profile.education,
        )
        occupation = occupation_data["name"]

        # Determine income based on occupation and experience
        if profile.income_level:
            income_ranges = INCOME_RANGES[lang]
            income_range = income_ranges[profile.income_level]
            income = random.randint(income_range[0], income_range[1])
        else:
            # Use GUS 2024 based income with all modifiers
            income = self._calculate_income(
                age,
                occupation_data,
                gender,
                location_type,
                location,
                region=region,
            )

        # Generate name from i18n data
        name = random.choice(FIRST_NAMES[lang][gender])
        
        # Select education: use filter if provided, otherwise age-based distribution
        if profile.education:
            education = self._normalize_education(profile.education)
        else:
            raw_edu = self._select_education_for_age(age, occupation)
            education = self._normalize_education(raw_edu)
        
        # Select marital status based on age (GUS NSP 2021)
        marital_status = self._select_marital_status_for_age(age)
        
        # Determine if has children (GUS 2024)
        has_children = self._select_has_children(age)

        return Persona(
            name=f"{name}_{index}",
            age=age,
            gender=gender,
            income=income,
            location=location,
            location_type=location_type,
            education=education,
            occupation=occupation,
            marital_status=marital_status,
            has_children=has_children,
        )

    def _select_occupation_for_age(
        self,
        age: int,
        gender: str = "M",
        education_filter: str | None = None,
    ) -> dict:
        """
        Select a realistic occupation based on agent's age and gender.
        
        Uses GUS BAEL 2024 occupation distribution data combined with
        gender-specific weights to create realistic occupational distribution.
        
        Example: mechanik has 97% male / 3% female distribution based on GUS data.
        
        Source: GUS BAEL 2024, ISCO-08 occupation groups, MEN 2024
        """
        lang = self.language
        valid_occupations = []
        weights = []
        
        education_filter_pl = self._education_to_pl_key(education_filter)
        for occ in OCCUPATIONS[lang]:
            min_age = occ.get("min_age", 18)
            max_age = occ.get("max_age", 100)
            if min_age <= age <= max_age:
                occ_name = occ.get("name", "")

                if education_filter_pl:
                    occ_name_pl = self._occupation_to_pl(occ_name) or occ_name
                    min_education = get_min_education_for_occupation(occ_name_pl)
                    if not self._education_meets_min(education_filter_pl, min_education):
                        continue
                
                # Use PL names for weights so EN behaves like PL
                occ_name_pl = self._occupation_to_pl(occ_name) or occ_name
                # Get population weight from reference data
                base_weight = OCCUPATION_POPULATION_WEIGHTS.get(occ_name_pl, 0.01)
                
                # Apply gender-specific weight modifier (GUS BAEL 2024)
                gender_weight = get_occupation_gender_weight(occ_name_pl, gender)
                weight = base_weight * gender_weight
                
                # Special handling for age-dependent statuses
                if occ_name_pl == "student" and 18 <= age <= 27:
                    # Students are common in 18-27 age group
                    weight = (0.30 if age <= 24 else 0.10) * gender_weight
                elif occ_name_pl == "emeryt":
                    # Retirees become more common with age, gender-specific threshold
                    retirement_age = RETIREMENT_AGE_BY_GENDER.get(gender, 65)
                    if age >= retirement_age + 10:
                        weight = 0.90 * gender_weight
                    elif age >= retirement_age + 5:
                        weight = 0.75 * gender_weight
                    elif age >= retirement_age:
                        weight = 0.50 * gender_weight
                    elif age >= retirement_age - 5:
                        weight = 0.20 * gender_weight
                    else:
                        weight = 0.0
                elif occ_name_pl == "rencista":
                    weight = 0.02 * gender_weight  # ~2% of 35+ population
                
                valid_occupations.append(occ)
                weights.append(weight)
        
        if not valid_occupations:
            # Fallback to generic occupation
            if lang == Language.PL:
                return {"name": "pracownik", "min_age": 18, "max_age": 100, "income_min": 3000, "income_max": 6000}
            else:
                return {"name": "worker", "min_age": 18, "max_age": 100, "income_min": 3000, "income_max": 6000}
        
        # Normalize weights and select
        total_weight = sum(weights) or 1.0
        normalized_weights = [w / total_weight for w in weights]
        
        return random.choices(valid_occupations, weights=normalized_weights)[0]

    def _select_education_for_age(self, age: int, occupation: str) -> str:
        """
        Select education level based on age distribution and occupation requirements.
        
        Uses GUS NSP 2021 data for age-based distribution and respects occupation
        minimum requirements. Implements overqualification (20% work below qualifications).
        
        Source: GUS NSP 2021, CBOS 2024
        """
        # Get age-based education distribution
        edu_distribution = get_education_distribution_for_age(age)
        
        # Get minimum education for occupation
        min_education = get_min_education_for_occupation(occupation)
        
        # Education level hierarchy for comparison
        edu_hierarchy = ["podstawowe", "zasadnicze zawodowe", "średnie", "policealne", "wyższe"]
        min_idx = edu_hierarchy.index(min_education) if min_education in edu_hierarchy else 0
        
        # Sample from distribution first
        education = random.choices(
            list(edu_distribution.keys()),
            weights=list(edu_distribution.values()),
        )[0]
        
        # Check if education meets occupation requirements
        edu_idx = edu_hierarchy.index(education) if education in edu_hierarchy else 0
        
        if edu_idx < min_idx:
            # Person doesn't have enough education for this occupation
            # Either they're overqualified (working below their level) or we need to adjust
            if random.random() < OVERQUALIFICATION_RATE:
                # They're actually overqualified but working in simpler job
                # Keep higher education from distribution
                pass
            else:
                # Adjust education to match occupation minimum
                education = min_education
        
        return education

    def _select_marital_status_for_age(self, age: int) -> str:
        """
        Select marital status based on age distribution.
        
        Source: GUS NSP 2021, Tablica 3.8
        """
        status_distribution = get_marital_status_distribution_for_age(age)
        return random.choices(
            list(status_distribution.keys()),
            weights=list(status_distribution.values()),
        )[0]

    def _select_has_children(self, age: int) -> bool:
        """
        Determine if persona has children based on age probability.
        
        Source: GUS 2024 demographic data
        """
        probability = get_has_children_probability(age)
        return random.random() < probability

    def _calculate_income(
        self, 
        age: int, 
        occupation: dict, 
        gender: str = "M",
        location_type: str = "urban",
        location: str | None = None,
        region: str | None = None,
    ) -> int:
        """
        Calculate realistic income using GUS 2024 data with modifiers.
        
        Factors applied:
        - Occupation base (from OCCUPATION_INCOME_DATA or fallback)
        - Experience factor (age - min_age progression)
        - Gender gap (GUS 2024: ~17% difference)
        - Regional factor (Mazowieckie +16%, Podkarpackie -14%)
        - Location type (urban +8%, rural -12%)
        
        Sources: GUS Struktura wynagrodzeń 2024, Sedlak & Sedlak 2024
        """
        occ_name = occupation.get("name", "")
        
        # Get income data from reference data (verified GUS 2024)
        if occ_name in OCCUPATION_INCOME_DATA:
            income_data = OCCUPATION_INCOME_DATA[occ_name]
            median = income_data.get("median", 5000)
            p25 = income_data.get("p25", median * 0.7)
            p75 = income_data.get("p75", median * 1.4)
            min_age = income_data.get("min_age", occupation.get("min_age", 18))
        else:
            # Fallback to occupation dict (i18n.py data)
            income_min = occupation.get("income_min", 3000)
            income_max = occupation.get("income_max", 8000)
            median = (income_min + income_max) / 2
            p25 = income_min
            p75 = income_max
            min_age = occupation.get("min_age", 18)
        
        # Special case: pensioners and disability pensioners
        # Note: PENSION_BY_GENDER data is already in netto (ZUS 2024)
        if occ_name in ("emeryt", "retiree"):
            pension_data = PENSION_BY_GENDER.get(gender, {"median": 3000, "std": 800})
            pension_netto = random.gauss(pension_data["median"], pension_data["std"])
            # Older retirees often have lower pensions
            if age > 75:
                pension_netto *= 0.92
            return int(round(max(1500, pension_netto), -2))
        
        if occ_name in ("rencista", "disability pensioner"):
            # Disability pension is lower than regular pension (netto)
            base_pension = random.gauss(2000, 400)
            return int(round(max(1500, base_pension), -2))
        
        # Experience factor (0.0 at career start, 1.0 at peak ~20 years)
        career_years = max(0, age - min_age)
        peak_years = 20
        experience_factor = min(1.0, career_years / peak_years)
        
        # After peak (age ~55+), slight decline
        if age > 55:
            experience_factor *= max(0.88, 1.0 - (age - 55) * 0.012)
        
        # Calculate base income based on experience
        # Early career: closer to p25, peak career: closer to p75
        base_income = p25 + (p75 - p25) * experience_factor
        
        # Apply modifiers
        # 1. Gender gap (GUS 2024: men +8.5%, women -8.5% vs median)
        gender_factor = GENDER_WAGE_GAP.get(gender, 1.0)
        
        # 2. Location type factor
        location_factor = LOCATION_WAGE_INDEX.get(location_type, 1.0)
        
        # 3. Regional factor (if location is a known city)
        regional_factor = 1.0
        if location:
            regional_factor = get_regional_multiplier(location, region)
        
        # Combine factors
        final_income = base_income * gender_factor * location_factor * regional_factor
        
        # Add random variation (±10%)
        variation = random.uniform(-0.10, 0.10)
        final_income *= (1 + variation)
        
        # Ensure minimum realistic income (national minimum wage netto ~3000 PLN)
        final_income = max(p25 * 0.85, final_income)
        
        return int(round(final_income, -2))  # Round to nearest 100

    def generate_population(
        self,
        profile: DemographicProfile | None = None,
        n_agents: int = 100,
    ) -> List[Persona]:
        """
        Generate a population of synthetic personas.
        
        Uses GUS statistics for realistic Polish demographics when language=PL.
        """
        personas = []

        for i in range(n_agents):
            persona = self.generate_persona(profile=profile, index=i)
            personas.append(persona)

        return personas
