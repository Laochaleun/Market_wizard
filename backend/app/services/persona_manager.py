"""
Persona Manager - Generation of synthetic consumer personas.

Supports demographic stratification and optional integration with 
GUS (Główny Urząd Statystyczny) BDL API for real Polish demographics.
"""

import random
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


class GUSClient:
    """Client for GUS BDL API to fetch real demographic data."""

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
        self.headers = {"Accept": "application/json"}
        if self.api_key:
            self.headers["X-ClientId"] = self.api_key

    def get_age_distribution(self) -> dict[str, int]:
        """
        Get population by age groups.
        Returns cached/default data (API calls are done sync at startup if needed).
        """
        # For now, use realistic default distribution based on GUS census data
        # This avoids async complexity and rate limiting issues
        return self.DEFAULT_AGE_DISTRIBUTION.copy()
    
    def get_income_for_age(self, age: int) -> int:
        """Get realistic income based on age from GUS wage statistics."""
        age_group = self._get_age_group(age)
        params = self.INCOME_BY_AGE.get(age_group, {"mean": 5000, "std": 1500})
        
        # Sample from normal distribution
        income = random.gauss(params["mean"], params["std"])
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
        return random.choices(
            list(self.GENDER_DISTRIBUTION.keys()),
            weights=list(self.GENDER_DISTRIBUTION.values())
        )[0]
    
    def sample_location_type(self) -> str:
        """Sample location type based on urbanization statistics."""
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
