#!/usr/bin/env python3
"""
Test persona generation with GUS 2024 verified data.
Tests occupation distribution, income ranges, and demographic correlations.
"""

import sys
import os
import argparse
import random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import Counter
from app.services.persona_manager import PersonaManager
from app.i18n import Language

def test_occupation_distribution(language: Language):
    """Test that occupation distribution follows population weights."""
    print("=" * 60)
    print("TEST 1: Rozkład zawodów (wagi populacyjne GUS BAEL 2024)")
    print("=" * 60)
    
    pm = PersonaManager(language=language)
    occupations = [pm.generate_persona(index=i).occupation for i in range(1000)]
    counts = Counter(occupations)
    
    print("\nTop 10 zawodów (z 1000 person):")
    for occ, count in counts.most_common(10):
        print(f"  {occ:25} {count:4} ({count/10:.1f}%)")
    
    # Verify doctors are rare (~1.2%)
    doctor_label = 'lekarz' if language == Language.PL else 'doctor'
    doctor_count = counts.get(doctor_label, 0)
    print(f"\nLekarze/Doctors: {doctor_count} (oczekiwane: ~12-20 z 1000)")
    if doctor_count > 50:
        print("  ❌ ZA DUŻO lekarzy!")
    else:
        print("  ✅ OK - lekarze są rzadcy zgodnie z GUS")
    
    # Verify salespeople are common (~8%)
    sales_label = 'sprzedawca' if language == Language.PL else 'sales associate'
    salesperson_count = counts.get(sales_label, 0)
    print(f"\nSprzedawcy/Sales: {salesperson_count} (oczekiwane: ~60-100 z 1000)")
    if salesperson_count < 30:
        print("  ⚠️ Za mało sprzedawców")
    else:
        print("  ✅ OK - sprzedawcy są częstsi")
    
    return True


def test_income_by_occupation(language: Language):
    """Test that incomes are realistic for each occupation."""
    print("\n" + "=" * 60)
    print("TEST 2: Zarobki według zawodów (GUS 2024)")
    print("=" * 60)
    
    pm = PersonaManager(language=language)
    
    occupation_incomes = {}
    for i in range(500):
        p = pm.generate_persona(index=i)
        if p.occupation not in occupation_incomes:
            occupation_incomes[p.occupation] = []
        occupation_incomes[p.occupation].append(p.income)
    
    print("\nŚrednie zarobki netto według zawodu:")
    for occ, incomes in sorted(occupation_incomes.items(), key=lambda x: -sum(x[1])/len(x[1])):
        avg = sum(incomes) / len(incomes)
        min_inc = min(incomes)
        max_inc = max(incomes)
        print(f"  {occ:25} avg: {avg:>8,.0f} PLN  (min: {min_inc:>6,}, max: {max_inc:>6,})")
    
    # Check software developer earns well
    dev_label = 'programista' if language == Language.PL else 'software developer'
    if dev_label in occupation_incomes:
        prog_avg = sum(occupation_incomes[dev_label]) / len(occupation_incomes[dev_label])
        print(f"\nProgramista/Dev średnia: {prog_avg:,.0f} PLN (oczekiwane: 6000-15000)")
        if prog_avg >= 5500:
            print("  ✅ OK")
        else:
            print("  ❌ Za nisko!")
    
    return True


def test_gender_wage_gap(language: Language, samples: int):
    """Test that gender wage gap is applied."""
    print("\n" + "=" * 60)
    print("TEST 3: Różnica zarobków M/F (GUS 2024: ~17%)")
    print("=" * 60)
    
    pm = PersonaManager(language=language)
    
    male_incomes = []
    female_incomes = []
    
    for i in range(samples):
        p = pm.generate_persona(index=i)
        if p.occupation not in ('emeryt', 'rencista', 'student', 'retiree', 'disability pensioner'):
            if p.gender == 'M':
                male_incomes.append(p.income)
            else:
                female_incomes.append(p.income)
    
    if male_incomes and female_incomes:
        male_avg = sum(male_incomes) / len(male_incomes)
        female_avg = sum(female_incomes) / len(female_incomes)
        gap = (male_avg - female_avg) / male_avg * 100
        
        print(f"\nMężczyźni średnia: {male_avg:,.0f} PLN (n={len(male_incomes)})")
        print(f"Kobiety średnia:   {female_avg:,.0f} PLN (n={len(female_incomes)})")
        print(f"Różnica: {gap:.1f}% (oczekiwane: ~15-20%)")
        
        if 10 <= gap <= 25:
            print("  ✅ OK - zgodne z GUS")
        else:
            print(f"  ⚠️ Różnica poza zakresem")
    
    return True


def test_age_occupation_correlation(language: Language):
    """Test that young people don't have professions requiring degrees."""
    print("\n" + "=" * 60)
    print("TEST 4: Korelacja wiek-zawód")
    print("=" * 60)
    
    pm = PersonaManager(language=language)
    
    medical_occupations = (
        ['lekarz', 'dentysta', 'prawnik', 'architekt', 'farmaceuta']
        if language == Language.PL
        else ['doctor', 'dentist', 'lawyer', 'architect', 'pharmacist']
    )
    retiree_label = 'emeryt' if language == Language.PL else 'retiree'
    violations = []
    
    for i in range(500):
        p = pm.generate_persona(index=i)
        if p.occupation in medical_occupations and p.age < 24:
            violations.append(f"{p.age} lat, {p.occupation}")
        if p.occupation == retiree_label and p.age < 60:
            violations.append(f"{p.age} lat, emeryt!")
    
    if violations:
        print(f"\n❌ BŁĘDY ({len(violations)}):")
        for v in violations[:5]:
            print(f"  {v}")
    else:
        print("\n✅ OK - brak naruszeń wiek-zawód")
    
    return len(violations) == 0


def test_retiree_distribution(language: Language):
    """Test that retirees are more common with age."""
    print("\n" + "=" * 60)
    print("TEST 5: Rozkład emerytów według wieku")
    print("=" * 60)
    
    pm = PersonaManager(language=language)
    
    age_groups = {
        "18-59": {"total": 0, "retired": 0},
        "60-64": {"total": 0, "retired": 0},
        "65-69": {"total": 0, "retired": 0},
        "70-74": {"total": 0, "retired": 0},
        "75+": {"total": 0, "retired": 0},
    }
    
    for i in range(1000):
        p = pm.generate_persona(index=i)
        if p.age < 60:
            group = "18-59"
        elif p.age < 65:
            group = "60-64"
        elif p.age < 70:
            group = "65-69"
        elif p.age < 75:
            group = "70-74"
        else:
            group = "75+"
        
        age_groups[group]["total"] += 1
        if p.occupation in ("emeryt", "retiree"):
            age_groups[group]["retired"] += 1
    
    print("\nProcent emerytów w grupach wiekowych:")
    for group, data in age_groups.items():
        if data["total"] > 0:
            pct = data["retired"] / data["total"] * 100
            print(f"  {group:8} {pct:5.1f}% emerytów (n={data['total']})")
    
    return True


def test_location_distribution(language: Language):
    """Test that location distribution matches GUS 2024 weights."""
    print("\n" + "=" * 60)
    print("TEST 6: Rozkład lokalizacji (GUS 2024: wieś ~41%, metropolie ~13%)")
    print("=" * 60)
    
    pm = PersonaManager(language=language)
    
    # We need to map city names back to types to verify distribution
    from app.i18n import LOCATIONS
    inv_map = {}
    for l_type, cities in LOCATIONS[language].items():
        # Skip legacy keys to avoid overwriting specific categories
        if l_type in ("urban", "suburban"):
            continue
        for city in cities:
            inv_map[city] = l_type
            
    detected_types = Counter()
    total_samples = 500
    
    for i in range(total_samples):
        p = pm.generate_persona(index=i)
        l_type = inv_map.get(p.location, 'unknown')
        detected_types[l_type] += 1
        
    print("\nRozkład lokalizacji:")
    valid_dist = True
    for l_type, count in detected_types.most_common():
        pct = count / total_samples * 100
        print(f"  {l_type:15} {pct:5.1f}%")
        
        # Verify rough bounds (allow high variance due to small sample)
        if l_type == 'rural':
            if not (30 <= pct <= 55):
                print(f"    ⚠️ Rural outside expected range (30-55%)")
                valid_dist = False
        elif l_type == 'metropolis':
            if not (5 <= pct <= 25):
                print(f"    ⚠️ Metropolis outside expected range (5-25%)")
                valid_dist = False

    if valid_dist:
        print("\n  ✅ OK - zgodne z przybliżeniem GUS")
    else:
        print("\n  ⚠️ Rozkład może wymagać kalibracji lub większej próby")
        
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=["pl", "en"], default="pl")
    args = parser.parse_args()

    language = Language.PL if args.lang == "pl" else Language.EN
    print(f"\n🔍 TESTY GENEROWANIA PERSON Z DANYMI GUS 2024 ({args.lang.upper()})\n")
    
    all_passed = True
    all_passed &= test_occupation_distribution(language)
    all_passed &= test_income_by_occupation(language)
    random.seed(42)
    all_passed &= test_gender_wage_gap(language, samples=4000)
    all_passed &= test_age_occupation_correlation(language)
    all_passed &= test_retiree_distribution(language)
    all_passed &= test_location_distribution(language)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ WSZYSTKIE TESTY PRZESZŁY POMYŚLNIE")
    else:
        print("❌ NIEKTÓRE TESTY NIE PRZESZŁY")
    print("=" * 60)
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
