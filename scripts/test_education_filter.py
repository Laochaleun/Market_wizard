
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.services.persona_manager import PersonaManager
from app.models.schemas import DemographicProfile
from app.i18n import Language

def test_education_normalization():
    print("Testing education normalization logic...")

    # Test 1: PL Manager with EN input
    print("\n--- Test 1: PL Manager with EN input ('primary') ---")
    manager_pl = PersonaManager(language=Language.PL)
    profile_en_input = DemographicProfile(
        age_min=25, age_max=35, 
        education="primary" # English input
    )
    persona_pl = manager_pl.generate_persona(profile_en_input)
    print(f"Input education: 'primary'")
    print(f"Persona education: '{persona_pl.education}'")
    
    if persona_pl.education == "podstawowe":
        print("✅ SUCCESS: Normalized to 'podstawowe'")
    else:
        print(f"❌ FAILED: Expected 'podstawowe', got '{persona_pl.education}'")

    # Test 2: EN Manager with PL input
    print("\n--- Test 2: EN Manager with PL input ('wyższe') ---")
    manager_en = PersonaManager(language=Language.EN)
    profile_pl_input = DemographicProfile(
        age_min=25, age_max=35,
        education="wyższe" # Polish input
    )
    persona_en = manager_en.generate_persona(profile_pl_input)
    print(f"Input education: 'wyższe'")
    print(f"Persona education: '{persona_en.education}'")

    if persona_en.education == "higher":
        print("✅ SUCCESS: Normalized to 'higher'")
    else:
        print(f"❌ FAILED: Expected 'higher', got '{persona_en.education}'")

    # Test 3: Auto-generation normalization (PL)
    print("\n--- Test 3: Auto-generation normalization (PL) ---")
    # This checks if internally generated education levels (from _select_education_for_age)
    # are correctly handled (they are usually PL keys)
    persona_auto_pl = manager_pl.generate_persona(DemographicProfile(age_min=30, age_max=30))
    print(f"Auto-generated PL education: '{persona_auto_pl.education}'")
    # Should be a Polish string
    if persona_auto_pl.education in ["podstawowe", "zasadnicze zawodowe", "średnie", "policealne", "wyższe"]:
         print("✅ SUCCESS: Valid Polish education level")
    else:
         print(f"❌ FAILED: Invalid or non-Polish education level")

    # Test 4: Auto-generation normalization (EN)
    print("\n--- Test 4: Auto-generation normalization (EN) ---")
    # _select_education_for_age returns PL keys by default, so Manager(EN) must translate them
    persona_auto_en = manager_en.generate_persona(DemographicProfile(age_min=30, age_max=30))
    print(f"Auto-generated EN education: '{persona_auto_en.education}'")
    # Should be an English string
    if persona_auto_en.education in ["primary", "vocational", "secondary", "post-secondary", "higher"]:
         print("✅ SUCCESS: Valid English education level (Translated from internal PL logic)")
    else:
         print(f"❌ FAILED: Invalid or non-English education level")

if __name__ == "__main__":
    test_education_normalization()
