"""
Reference data for persona generation based on official Polish statistics.

Sources:
- GUS Struktura wynagrodzeń według zawodów (październik 2024) - opubl. 14.10.2025
- GUS Rozkład wynagrodzeń w gospodarce narodowej (grudzień 2024)
- GUS BAEL 2024 - Aktywność ekonomiczna ludności Polski
- ZUS Struktura świadczeń 2024
- Sedlak & Sedlak Raport płacowy jesień/zima 2024

Last updated: 2026-01-31
"""

from typing import Dict, Any

# =============================================================================
# WYNAGRODZENIA WEDŁUG WOJEWÓDZTW (GUS 2024)
# Źródło: stat.gov.pl - Obwieszczenie Prezesa GUS o przeciętnym miesięcznym
# wynagrodzeniu brutto w gospodarce narodowej w województwach w 2024 r.
# Średnia krajowa: 8182 PLN brutto
# =============================================================================

REGIONAL_WAGE_INDEX: Dict[str, float] = {
    "mazowieckie": 1.16,      # 9489 PLN - najwyższe w kraju
    "dolnośląskie": 1.02,     # 8360 PLN
    "małopolskie": 1.01,      # 8249 PLN
    "śląskie": 0.99,          # 8096 PLN
    "pomorskie": 0.99,        # 8067 PLN
    "wielkopolskie": 0.98,    # ~8018 PLN
    "łódzkie": 0.94,          # ~7691 PLN
    "zachodniopomorskie": 0.94,
    "kujawsko-pomorskie": 0.92,
    "lubuskie": 0.91,
    "opolskie": 0.90,
    "świętokrzyskie": 0.89,
    "lubelskie": 0.88,
    "warmińsko-mazurskie": 0.87,
    "podkarpackie": 0.86,     # najniższe
    "podlaskie": 0.86,
}

# Mapowanie lokalizacji miejskich na województwa (dla uproszczenia)
CITY_TO_REGION: Dict[str, str] = {
    "Warszawa": "mazowieckie",
    "Kraków": "małopolskie",
    "Wrocław": "dolnośląskie",
    "Poznań": "wielkopolskie",
    "Łódź": "łódzkie",
    "Gdańsk": "pomorskie",
    "Szczecin": "zachodniopomorskie",
    "Lublin": "lubelskie",
    "Katowice": "śląskie",
    "Białystok": "podlaskie",
}

# =============================================================================
# WSPÓŁCZYNNIK PŁCI (GUS 2024)
# Źródło: GUS Struktura wynagrodzeń - różnica ok. 17% na niekorzyść kobiet
# =============================================================================

GENDER_WAGE_GAP: Dict[str, float] = {
    "M": 1.085,   # mężczyźni ~8.5% powyżej mediany ogólnej
    "F": 0.915,   # kobiety ~8.5% poniżej mediany ogólnej
}

# =============================================================================
# LOKALIZACJA - KLASYFIKACJA WEDŁUG WIELKOŚCI (GUS 2024)
# Źródło: GUS "Powierzchnia i ludność w przekroju terytorialnym" 2024
# Ludność: 37.5 mln, miasta: 59%, wieś: 41%
# =============================================================================

# Rozkład populacji według wielkości miejscowości (GUS 2024)
LOCATION_POPULATION_WEIGHTS: Dict[str, float] = {
    "rural": 0.41,              # Wieś - 41% populacji (~15.4 mln)
    "small_city": 0.12,         # Małe miasta do 20 tys. (~4.5 mln)
    "medium_city": 0.18,        # Średnie miasta 20-100 tys. (~6.7 mln)
    "large_city": 0.16,         # Duże miasta 100-500 tys. (~6 mln) - 31 miast
    "metropolis": 0.13,         # Metropolie 500 tys.+ (~4.9 mln) - 5 miast
}

# Współczynnik wynagrodzeń według wielkości miejscowości (GUS 2024)
LOCATION_WAGE_INDEX: Dict[str, float] = {
    "rural": 0.85,              # Wieś - ~15% niższe zarobki
    "small_city": 0.92,         # Małe miasta - ~8% niższe
    "medium_city": 1.00,        # Średnie miasta - poziom bazowy
    "large_city": 1.10,         # Duże miasta - ~10% wyższe
    "metropolis": 1.22,         # Metropolie - ~22% wyższe (Warszawa, Kraków itp.)
}

# Lista metropolii (miasta 500 tys.+)
METROPOLISES = ["Warszawa", "Kraków", "Łódź", "Wrocław", "Poznań"]

# Lista dużych miast (100-500 tys.)
LARGE_CITIES = [
    "Gdańsk", "Szczecin", "Bydgoszcz", "Lublin", "Białystok", "Katowice",
    "Gdynia", "Częstochowa", "Radom", "Sosnowiec", "Toruń", "Kielce",
    "Rzeszów", "Gliwice", "Zabrze", "Olsztyn", "Bielsko-Biała", "Bytom",
    "Zielona Góra", "Rybnik", "Ruda Śląska", "Tychy", "Opole", "Gorzów Wielkopolski",
    "Dąbrowa Górnicza", "Płock", "Elbląg", "Wałbrzych", "Włocławek", "Tarnów", "Chorzów",
]

# =============================================================================
# EMERYTURY WEDŁUG PŁCI (ZUS 2024) - NETTO
# Źródło: ZUS Struktura wysokości świadczeń wypłacanych przez ZUS
# po waloryzacji w marcu 2024 r.
# Przeliczone na netto (×0.85 - niższy podatek dla emerytów)
# Brutto: M=4675, F=3209 → Netto: M≈3974, F≈2728
# =============================================================================

PENSION_BY_GENDER: Dict[str, Dict[str, int]] = {
    "M": {"median": 3975, "std": 1000},  # mężczyźni - wyższa emerytura netto
    "F": {"median": 2730, "std": 750},   # kobiety - niższa netto (o ~31%)
}

# Średni wiek przejścia na emeryturę (ZUS 2024)
AVERAGE_RETIREMENT_AGE = 62.7

# =============================================================================
# ROZKŁAD PŁCI WEDŁUG WIEKU (GUS 2024)
# Źródło: GUS Rocznik Demograficzny 2024, tabl. 15
# Kobiety żyją dłużej - proporcja zmienia się z wiekiem
# =============================================================================

GENDER_BY_AGE: Dict[str, Dict[str, float]] = {
    # Przy urodzeniu więcej chłopców (105:100), ale wyrównuje się z wiekiem
    "18-29": {"M": 0.51, "F": 0.49},   # Młodzi - więcej mężczyzn
    "30-39": {"M": 0.50, "F": 0.50},   # Równowaga
    "40-49": {"M": 0.50, "F": 0.50},   # Równowaga
    "50-59": {"M": 0.48, "F": 0.52},   # Kobiety zaczynają dominować
    "60-64": {"M": 0.46, "F": 0.54},   # Wyraźna różnica
    "65-69": {"M": 0.43, "F": 0.57},   # Kobiety znacząco dominują
    "70-74": {"M": 0.40, "F": 0.60},   # Duża różnica
    "75-79": {"M": 0.36, "F": 0.64},   # Kobiety żyją dłużej
    "80+": {"M": 0.30, "F": 0.70},     # Zdecydowana przewaga kobiet
}


def get_gender_distribution_for_age(age: int) -> Dict[str, float]:
    """Get gender distribution probability for a given age."""
    if age < 30:
        return GENDER_BY_AGE["18-29"]
    elif age < 40:
        return GENDER_BY_AGE["30-39"]
    elif age < 50:
        return GENDER_BY_AGE["40-49"]
    elif age < 60:
        return GENDER_BY_AGE["50-59"]
    elif age < 65:
        return GENDER_BY_AGE["60-64"]
    elif age < 70:
        return GENDER_BY_AGE["65-69"]
    elif age < 75:
        return GENDER_BY_AGE["70-74"]
    elif age < 80:
        return GENDER_BY_AGE["75-79"]
    else:
        return GENDER_BY_AGE["80+"]


# =============================================================================
# OGÓLNE STATYSTYKI WYNAGRODZEŃ (GUS grudzień 2024) - NETTO
# Wszystkie dane w tym pliku są w PLN netto miesięcznie
# =============================================================================

NATIONAL_WAGE_STATS_2024 = {
    "mean_gross": 8182,      # przeciętne brutto (dla referencji)
    "median_gross": 7267,    # mediana brutto (dla referencji)
    "mean_net": 5950,        # przeciętne netto (~72.7%)
    "median_net": 5280,      # mediana netto (~72.7%)
}

# =============================================================================
# STRUKTURA ZATRUDNIENIA WEDŁUG WIELKICH GRUP ZAWODOWYCH (GUS BAEL 2024)
# Źródło: GUS Aktywność ekonomiczna ludności Polski 2024
# Szacunki procentowe dla populacji 17.25 mln pracujących
# =============================================================================

# Wagi populacyjne dla zawodów - suma = 1.0 (tylko aktywni zawodowo)
# Na podstawie struktury ISCO-08 i danych BAEL
OCCUPATION_POPULATION_WEIGHTS: Dict[str, float] = {
    # ISCO 1: Przedstawiciele władz, kierownicy (~5%)
    "menedżer": 0.05,
    
    # ISCO 2: Specjaliści (~22% populacji pracującej)
    "programista": 0.04,          # IT mocno reprezentowane
    "lekarz": 0.012,              # ~200 tys. lekarzy na 17 mln
    "dentysta": 0.003,            # ~50 tys.
    "prawnik": 0.006,             # ~100 tys.
    "architekt": 0.002,           # ~35 tys.
    "farmaceuta": 0.003,          # ~50 tys.
    "inżynier": 0.03,             # różne branże
    "nauczyciel": 0.04,           # duża grupa
    "pielęgniarka": 0.02,         # ~340 tys.
    
    # ISCO 3: Technicy i średni personel (~12%)
    "księgowy": 0.025,
    "grafik": 0.01,
    
    # ISCO 4: Pracownicy biurowi (~10%)
    "pracownik biurowy": 0.10,
    
    # ISCO 5: Pracownicy usług i sprzedawcy (~16%)
    "sprzedawca": 0.08,           # duża grupa - handel
    "fryzjer": 0.012,
    "kelner": 0.015,
    "kucharz": 0.02,
    "policjant": 0.006,           # ~100 tys.
    "strażak": 0.003,             # ~50 tys.
    
    # ISCO 7: Robotnicy przemysłowi i rzemieślnicy (~14%)
    "mechanik": 0.035,
    "elektryk": 0.02,
    "pracownik budowlany": 0.05,
    
    # ISCO 8: Operatorzy maszyn i kierowcy (~8%)
    "kierowca": 0.05,
    
    # ISCO 9: Pracownicy przy pracach prostych (~7%)
    # (rozproszeni w innych kategoriach)
}

# Wagi dla osób nieaktywnych zawodowo/w szczególnych sytuacjach
# Te są stosowane zależnie od wieku
SPECIAL_STATUS_WEIGHTS: Dict[str, Dict[str, Any]] = {
    "student": {
        "age_range": (18, 27),
        "weight_in_age_range": 0.35,  # 35% osób 18-27 to studenci
    },
    "emeryt": {
        "age_range": (60, 100),
        "weight_by_age": {
            60: 0.20,   # 20% 60-latków to emeryci
            65: 0.60,   # 60% 65-latków
            70: 0.85,   # 85% 70-latków
            75: 0.95,   # 95% 75+
        }
    },
    "rencista": {
        "age_range": (35, 100),
        "weight_overall": 0.02,  # ~2% populacji 35+
    },
}

# =============================================================================
# WYNAGRODZENIA WEDŁUG ZAWODÓW (PLN netto miesięcznie)
# Źródła: GUS Struktura wynagrodzeń 2024, Sedlak & Sedlak 2024,
# raporty branżowe (money.pl, wynagrodzenia.pl)
# =============================================================================

OCCUPATION_INCOME_DATA: Dict[str, Dict[str, int]] = {
    # === Zawody wymagające studiów wyższych ===
    
    "lekarz": {
        # Źródło: Ustawa o wynagrodzeniach w podmiotach leczniczych 2024
        # Lekarz specjalista min. 10375 brutto, mediana 12000 brutto
        "median": 8500,    # netto po odliczeniach
        "p25": 6500,       # młody lekarz bez specjalizacji
        "p75": 14000,      # specjalista z doświadczeniem
        "min_age": 26,
        "max_age": 70,
    },
    "dentysta": {
        # Często prywatna praktyka - wyższe zarobki
        "median": 10000,
        "p25": 7000,
        "p75": 18000,
        "min_age": 26,
        "max_age": 70,
    },
    "prawnik": {
        # Zróżnicowane - od aplikanta do partnera kancelarii
        "median": 7500,
        "p25": 5000,
        "p75": 15000,
        "min_age": 24,
        "max_age": 70,
    },
    "architekt": {
        "median": 7000,
        "p25": 5000,
        "p75": 12000,
        "min_age": 26,
        "max_age": 70,
    },
    "farmaceuta": {
        "median": 6500,
        "p25": 5000,
        "p75": 9000,
        "min_age": 25,
        "max_age": 70,
    },
    "programista": {
        # Źródło: Just Join IT, Sedlak 2024
        # Mediana UoP: 8400 netto, B2B: 20000 netto
        "median": 9000,    # uśrednione UoP + B2B
        "p25": 5500,       # junior
        "p75": 16000,      # senior
        "min_age": 22,
        "max_age": 65,
    },
    "inżynier": {
        "median": 7000,
        "p25": 5000,
        "p75": 11000,
        "min_age": 23,
        "max_age": 70,
    },
    "nauczyciel": {
        # Źródło: money.pl - nauczyciel dyplomowany ~9500 brutto
        "median": 5500,
        "p25": 4200,       # początkujący
        "p75": 7000,       # dyplomowany
        "min_age": 23,
        "max_age": 67,
    },
    "pielęgniarka": {
        # Ustawa o wynagrodzeniach w podmiotach leczniczych
        "median": 5500,
        "p25": 4500,
        "p75": 7000,
        "min_age": 22,
        "max_age": 67,
    },
    
    # === Technicy i średni personel ===
    
    "księgowy": {
        "median": 5500,
        "p25": 4000,
        "p75": 8000,
        "min_age": 23,
        "max_age": 70,
    },
    "grafik": {
        "median": 5500,
        "p25": 3800,
        "p75": 9000,
        "min_age": 21,
        "max_age": 65,
    },
    
    # === Pracownicy biurowi ===
    
    "pracownik biurowy": {
        "median": 4800,
        "p25": 3800,
        "p75": 6500,
        "min_age": 19,
        "max_age": 67,
    },
    
    # === Pracownicy usług i sprzedawcy ===
    
    "sprzedawca": {
        # Źródło: Sedlak 2024 - kasjer mediana 4170 brutto
        "median": 3200,    # netto
        "p25": 2800,
        "p75": 4200,
        "min_age": 18,
        "max_age": 65,
    },
    "fryzjer": {
        "median": 3500,
        "p25": 2800,
        "p75": 5000,
        "min_age": 18,
        "max_age": 65,
    },
    "kelner": {
        # Z napiwkami może być więcej
        "median": 3200,
        "p25": 2600,
        "p75": 4500,
        "min_age": 18,
        "max_age": 55,
    },
    "kucharz": {
        "median": 4000,
        "p25": 3200,
        "p75": 6000,
        "min_age": 18,
        "max_age": 65,
    },
    "policjant": {
        "median": 5500,
        "p25": 4500,
        "p75": 7500,
        "min_age": 21,
        "max_age": 60,
    },
    "strażak": {
        "median": 5200,
        "p25": 4300,
        "p75": 7000,
        "min_age": 21,
        "max_age": 55,
    },
    
    # === Robotnicy przemysłowi i rzemieślnicy ===
    
    "mechanik": {
        "median": 4500,
        "p25": 3500,
        "p75": 6500,
        "min_age": 18,
        "max_age": 65,
    },
    "elektryk": {
        "median": 5000,
        "p25": 4000,
        "p75": 7000,
        "min_age": 18,
        "max_age": 65,
    },
    "pracownik budowlany": {
        # Często praca sezonowa, różne stawki
        "median": 5000,
        "p25": 4000,
        "p75": 7500,
        "min_age": 18,
        "max_age": 60,
    },
    
    # === Operatorzy i kierowcy ===
    
    "kierowca": {
        # TIR zarabia więcej niż kierowca dostawczy
        "median": 5000,
        "p25": 3800,
        "p75": 7500,
        "min_age": 21,
        "max_age": 67,
    },
    
    # === Statusy specjalne ===
    
    "student": {
        # Praca dorywcza, staże
        "median": 1200,
        "p25": 0,
        "p75": 2500,
        "min_age": 18,
        "max_age": 27,
    },
    "emeryt": {
        # Emerytura - obsługiwane osobno przez PENSION_BY_GENDER
        "median": 3000,   # netto średnia
        "p25": 2200,
        "p75": 3800,
        "min_age": 60,
        "max_age": 100,
    },
    "rencista": {
        "median": 2400,
        "p25": 1800,
        "p75": 3000,
        "min_age": 35,
        "max_age": 100,
    },
    
    # === Menedżer (kierownik) ===
    
    "menedżer": {
        # Bardzo zróżnicowane - od kierownika zmiany do dyrektora
        "median": 10000,
        "p25": 6500,
        "p75": 18000,
        "min_age": 28,
        "max_age": 65,
    },
}

# =============================================================================
# FUNKCJE POMOCNICZE
# =============================================================================

def get_regional_multiplier(city: str | None, region: str | None) -> float:
    """Get regional wage multiplier based on city or region name."""
    if city and city in CITY_TO_REGION:
        region = CITY_TO_REGION[city]
    if region and region.lower() in REGIONAL_WAGE_INDEX:
        return REGIONAL_WAGE_INDEX[region.lower()]
    return 1.0  # default - national average


def get_occupation_data(occupation_name: str) -> Dict[str, int]:
    """Get income data for a specific occupation."""
    return OCCUPATION_INCOME_DATA.get(
        occupation_name,
        {"median": 5000, "p25": 3500, "p75": 7000, "min_age": 18, "max_age": 70}
    )


def get_pension_for_gender(gender: str) -> Dict[str, int]:
    """Get pension data based on gender."""
    return PENSION_BY_GENDER.get(gender, PENSION_BY_GENDER["M"])
