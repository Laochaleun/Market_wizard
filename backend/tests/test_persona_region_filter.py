"""Tests for region-aware persona generation."""

from app.i18n import Language
from app.models import DemographicProfile
from app.services.persona_manager import PersonaManager


def test_persona_manager_prefers_region_matching_locations():
    manager = PersonaManager(language=Language.PL, use_gus_demographics=False)
    profile = DemographicProfile(
        age_min=30,
        age_max=40,
        gender="F",
        location_type="metropolis",
        region="mazowieckie",
    )

    personas = manager.generate_population(profile=profile, n_agents=12)

    # For metropolis + mazowieckie, available mapped location should be Warsaw.
    assert personas
    assert all(p.location == "Warszawa" for p in personas)
    assert all(p.gender == "F" for p in personas)
