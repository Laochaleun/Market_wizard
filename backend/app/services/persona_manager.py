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


# Income distribution by level (PLN/month for PL, $/month for EN)
INCOME_RANGES = {
    Language.PL: {
        "low": (2000, 4000),
        "medium": (4000, 8000),
        "high": (8000, 20000),
    },
    Language.EN: {
        "low": (2000, 3500),
        "medium": (3500, 7000),
        "high": (7000, 15000),
    },
}

# Education levels
EDUCATION_LEVELS = {
    Language.PL: ["podstawowe", "średnie", "wyższe"],
    Language.EN: ["high school", "some college", "bachelor's degree", "master's degree"],
}


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
    
    # Location distribution based on GUS data
    LOCATION_DISTRIBUTION = {
        "urban": 0.60,      # 60% in cities
        "suburban": 0.22,   # 22% in suburban areas
        "rural": 0.18,      # 18% in rural areas
    }
    
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
    
    def sample_gender(self) -> str:
        """Sample gender based on population distribution."""
        self._ensure_live_loaded()
        if self._live_gender_distribution:
            return random.choices(
                list(self._live_gender_distribution.keys()),
                weights=list(self._live_gender_distribution.values()),
            )[0]
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
        self.use_gus_demographics = use_gus_demographics and (language == Language.PL)
        
        if self.use_gus_demographics:
            self.gus_client = GUSClient()

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
        
        # Determine gender (use GUS distribution if not specified)
        if profile.gender:
            gender = profile.gender
        else:
            gender = gus.sample_gender()

        # Determine age (use GUS distribution, but respect constraints)
        if profile.age_min != 18 or profile.age_max != 80:
            # Custom age range specified
            age = random.randint(profile.age_min, profile.age_max)
        else:
            age = gus.sample_age()
            # Clamp to profile constraints
            age = max(profile.age_min, min(profile.age_max, age))

        # Determine location type (use GUS urbanization stats if not specified)
        if profile.location_type:
            location_type = profile.location_type
        else:
            location_type = gus.sample_location_type()

        # Get location from i18n data
        location = random.choice(LOCATIONS[lang][location_type])

        # Determine income using GUS wage statistics
        if profile.income_level:
            income_ranges = INCOME_RANGES[lang]
            income_range = income_ranges[profile.income_level]
            income = random.randint(income_range[0], income_range[1])
        else:
            # Use GUS-based income that correlates with age
            income = gus.get_income_for_age(age)

        # Generate name, education, occupation from i18n data
        name = random.choice(FIRST_NAMES[lang][gender])
        education = random.choice(EDUCATION_LEVELS[lang])
        occupation = random.choice(OCCUPATIONS[lang])

        return Persona(
            name=f"{name}_{index}",
            age=age,
            gender=gender,
            income=income,
            location=location,
            location_type=location_type,
            education=education,
            occupation=occupation,
        )

    def _generate_persona_random(
        self,
        profile: DemographicProfile,
        index: int,
    ) -> Persona:
        """Generate persona using random sampling (for non-PL or when GUS disabled)."""
        lang = self.language

        # Determine gender
        if profile.gender:
            gender = profile.gender
        else:
            gender = random.choice(["M", "F"])

        # Determine age
        age = random.randint(profile.age_min, profile.age_max)

        # Determine location type
        if profile.location_type:
            location_type = profile.location_type
        else:
            location_type = random.choices(
                ["urban", "suburban", "rural"],
                weights=[0.6, 0.25, 0.15],
            )[0]

        # Determine location
        location = random.choice(LOCATIONS[lang][location_type])

        # Determine income
        income_ranges = INCOME_RANGES[lang]
        if profile.income_level:
            income_range = income_ranges[profile.income_level]
        else:
            if age < 25:
                income_range = income_ranges["low"]
            elif location_type == "urban" and age > 30:
                income_range = random.choice([income_ranges["medium"], income_ranges["high"]])
            else:
                income_range = income_ranges["medium"]

        income = random.randint(income_range[0], income_range[1])

        # Generate name, education, occupation
        name = random.choice(FIRST_NAMES[lang][gender])
        education = random.choice(EDUCATION_LEVELS[lang])
        occupation = random.choice(OCCUPATIONS[lang])

        return Persona(
            name=f"{name}_{index}",
            age=age,
            gender=gender,
            income=income,
            location=location,
            location_type=location_type,
            education=education,
            occupation=occupation,
        )

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
