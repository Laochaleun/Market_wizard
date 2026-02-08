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

import json
from pathlib import Path
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

GENDER_WAGE_GAP_DEFAULT: Dict[str, float] = {
    "M": 1.085,   # mężczyźni ~8.5% powyżej mediany ogólnej
    "F": 0.915,   # kobiety ~8.5% poniżej mediany ogólnej
}
GENDER_WAGE_GAP: Dict[str, float] = dict(GENDER_WAGE_GAP_DEFAULT)


def _load_gender_gap_calibration() -> None:
    path = Path(__file__).resolve().with_name("gender_gap_calibration.json")
    if not path.exists():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        m_val = float(payload.get("M", GENDER_WAGE_GAP["M"]))
        f_val = float(payload.get("F", GENDER_WAGE_GAP["F"]))
        if m_val > 0 and f_val > 0:
            GENDER_WAGE_GAP["M"] = m_val
            GENDER_WAGE_GAP["F"] = f_val
    except Exception:
        return


_load_gender_gap_calibration()

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
    # ISCO 1: Kierownicy (~6% populacji pracującej)
    "menedżer": 0.025,
    "dyrektor": 0.01,
    "przedsiębiorca": 0.025,       # ~21% samozatrudnionych = ~3.5 mln, część to przedsiębiorcy
    
    # ISCO 2: Specjaliści (~22% populacji pracującej)
    "programista": 0.04,          # IT mocno reprezentowane
    "lekarz": 0.012,              # ~200 tys. lekarzy na 17 mln
    "dentysta": 0.003,            # ~50 tys.
    "prawnik": 0.006,             # ~100 tys.
    "architekt": 0.002,           # ~35 tys.
    "farmaceuta": 0.003,          # ~50 tys.
    "inżynier": 0.03,             # różne branże
    "nauczyciel": 0.04,           # duża grupa
    
    # ISCO 3: Technicy i średni personel (~14%)
    "pielęgniarka": 0.02,         # ~340 tys.
    "księgowy": 0.025,
    "grafik": 0.01,
    "technik": 0.08,              # różne specjalizacje
    
    # ISCO 4: Pracownicy biurowi (~8%)
    "pracownik biurowy": 0.06,
    "sekretarka": 0.02,
    
    # ISCO 5: Pracownicy usług i sprzedawcy (~16%)
    "sprzedawca": 0.08,           # duża grupa - handel
    "fryzjer": 0.012,
    "kelner": 0.015,
    "kucharz": 0.02,
    "policjant": 0.006,           # ~100 tys.
    "strażak": 0.003,             # ~50 tys.
    "ochroniarz": 0.015,          # duża branża
    
    # ISCO 6: Rolnicy (~8%)
    "rolnik": 0.08,               # ~8% populacji pracującej
    
    # ISCO 7: Robotnicy przemysłowi i rzemieślnicy (~12%)
    "mechanik": 0.025,
    "elektryk": 0.02,
    "pracownik budowlany": 0.04,
    "stolarz": 0.01,
    "spawacz": 0.015,
    
    # ISCO 8: Operatorzy maszyn i kierowcy (~8%)
    "kierowca": 0.05,
    "operator produkcji": 0.03,
    
    # ISCO 9: Pracownicy przy pracach prostych (~6%)
    "magazynier": 0.03,
    "sprzątaczka": 0.025,
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
# ROZKŁAD PŁCI W ZAWODACH (GUS BAEL 2024, MEN 2024, NIL 2024)
# Źródło: GUS Aktywność ekonomiczna ludności 2024, branżowe raporty
# =============================================================================

OCCUPATION_GENDER_WEIGHTS: Dict[str, Dict[str, float]] = {
    # Zawody sfeminizowane
    "pielęgniarka": {"F": 0.97, "M": 0.03},
    "nauczyciel": {"F": 0.84, "M": 0.16},
    "fryzjer": {"F": 0.85, "M": 0.15},
    "sprzedawca": {"F": 0.70, "M": 0.30},
    "księgowy": {"F": 0.75, "M": 0.25},
    "pracownik biurowy": {"F": 0.65, "M": 0.35},
    "sekretarka": {"F": 0.95, "M": 0.05},
    "farmaceuta": {"F": 0.80, "M": 0.20},
    "kelner": {"F": 0.65, "M": 0.35},
    "dentysta": {"F": 0.75, "M": 0.25},
    "sprzątaczka": {"F": 0.90, "M": 0.10},
    
    # Zawody zrównoważone
    "lekarz": {"F": 0.58, "M": 0.42},
    "prawnik": {"F": 0.55, "M": 0.45},
    "kucharz": {"F": 0.55, "M": 0.45},
    "grafik": {"F": 0.45, "M": 0.55},
    "architekt": {"F": 0.45, "M": 0.55},
    "menedżer": {"F": 0.42, "M": 0.58},
    "dyrektor": {"F": 0.35, "M": 0.65},
    "technik": {"F": 0.30, "M": 0.70},
    "magazynier": {"F": 0.35, "M": 0.65},
    
    # Zawody smaskulinizowane
    "programista": {"F": 0.25, "M": 0.75},
    "przedsiębiorca": {"F": 0.38, "M": 0.62},  # GUS: 38% kobiet wśród samozatrudnionych
    "inżynier": {"F": 0.15, "M": 0.85},
    "policjant": {"F": 0.15, "M": 0.85},
    "ochroniarz": {"F": 0.10, "M": 0.90},
    "kierowca": {"F": 0.05, "M": 0.95},
    "strażak": {"F": 0.05, "M": 0.95},
    "mechanik": {"F": 0.03, "M": 0.97},
    "elektryk": {"F": 0.02, "M": 0.98},
    "pracownik budowlany": {"F": 0.02, "M": 0.98},
    "stolarz": {"F": 0.03, "M": 0.97},
    "spawacz": {"F": 0.02, "M": 0.98},
    "operator produkcji": {"F": 0.30, "M": 0.70},
    "rolnik": {"F": 0.35, "M": 0.65},  # GUS: rolnictwo ~35% kobiet
    
    # Statusy specjalne - neutralne
    "student": {"F": 0.55, "M": 0.45},
    "emeryt": {"F": 0.60, "M": 0.40},  # Kobiety żyją dłużej
    "rencista": {"F": 0.45, "M": 0.55},
    "bezrobotny": {"F": 0.48, "M": 0.52},  # GUS: nieco więcej mężczyzn
}

# =============================================================================
# WYKSZTAŁCENIE WEDŁUG WIEKU (GUS NSP 2021, CBOS 2024)
# Rozkład w % dla każdej grupy wiekowej
# =============================================================================

EDUCATION_BY_AGE: Dict[str, Dict[str, float]] = {
    "18-24": {
        "podstawowe": 0.20,
        "zasadnicze zawodowe": 0.15,
        "średnie": 0.50,
        "policealne": 0.05,
        "wyższe": 0.10,
    },
    "25-34": {
        "podstawowe": 0.05,
        "zasadnicze zawodowe": 0.12,
        "średnie": 0.30,
        "policealne": 0.05,
        "wyższe": 0.48,
    },
    "35-44": {
        "podstawowe": 0.08,
        "zasadnicze zawodowe": 0.25,
        "średnie": 0.25,
        "policealne": 0.05,
        "wyższe": 0.37,
    },
    "45-54": {
        "podstawowe": 0.12,
        "zasadnicze zawodowe": 0.35,
        "średnie": 0.25,
        "policealne": 0.05,
        "wyższe": 0.23,
    },
    "55-64": {
        "podstawowe": 0.18,
        "zasadnicze zawodowe": 0.32,
        "średnie": 0.28,
        "policealne": 0.05,
        "wyższe": 0.17,
    },
    "65-74": {
        "podstawowe": 0.25,
        "zasadnicze zawodowe": 0.30,
        "średnie": 0.26,
        "policealne": 0.05,
        "wyższe": 0.14,
    },
    "75+": {
        "podstawowe": 0.32,
        "zasadnicze zawodowe": 0.28,
        "średnie": 0.24,
        "policealne": 0.04,
        "wyższe": 0.12,
    },
}

# =============================================================================
# MINIMALNE WYKSZTAŁCENIE DLA ZAWODU
# =============================================================================

OCCUPATION_MIN_EDUCATION: Dict[str, str] = {
    # Zawody regulowane - wymagane wyższe
    "lekarz": "wyższe",
    "dentysta": "wyższe",
    "farmaceuta": "wyższe",
    "pielęgniarka": "wyższe",
    "nauczyciel": "wyższe",
    "prawnik": "wyższe",
    "architekt": "wyższe",
    "inżynier": "wyższe",

    # Specjaliści i technicy
    "programista": "średnie",
    "księgowy": "średnie",
    "grafik": "średnie",
    "technik": "średnie",

    # Kierownictwo i administracja
    "menedżer": "średnie",
    "dyrektor": "wyższe",
    "przedsiębiorca": "średnie",
    "pracownik biurowy": "średnie",
    "sekretarka": "średnie",

    # Służby i usługi (część wymaga średniego)
    "policjant": "średnie",
    "strażak": "średnie",
    "ochroniarz": "podstawowe",
    "sprzedawca": "podstawowe",
    "kelner": "podstawowe",
    "fryzjer": "zasadnicze zawodowe",
    "kucharz": "zasadnicze zawodowe",

    # Rolnictwo i prace proste
    "rolnik": "podstawowe",
    "magazynier": "podstawowe",
    "sprzątaczka": "podstawowe",

    # Rzemiosło i produkcja
    "mechanik": "zasadnicze zawodowe",
    "elektryk": "zasadnicze zawodowe",
    "pracownik budowlany": "podstawowe",
    "stolarz": "zasadnicze zawodowe",
    "spawacz": "zasadnicze zawodowe",
    "kierowca": "podstawowe",
    "operator produkcji": "podstawowe",

    # Statusy specjalne
    "student": "średnie",
    "emeryt": "podstawowe",
    "rencista": "podstawowe",
    "bezrobotny": "podstawowe",
}

# Współczynnik nadkwalifikacji (GUS 2024: 20% pracuje poniżej kwalifikacji)
OVERQUALIFICATION_RATE = 0.20

# =============================================================================
# STAN CYWILNY WEDŁUG WIEKU (GUS NSP 2021, Tablica 3.8)
# =============================================================================

MARITAL_STATUS_BY_AGE: Dict[str, Dict[str, float]] = {
    "18-24": {
        "kawaler/panna": 0.92,
        "małżeństwo": 0.07,
        "rozwiedziony": 0.01,
        "wdowiec/wdowa": 0.00,
    },
    "25-29": {
        "kawaler/panna": 0.60,
        "małżeństwo": 0.36,
        "rozwiedziony": 0.04,
        "wdowiec/wdowa": 0.00,
    },
    "30-34": {
        "kawaler/panna": 0.30,
        "małżeństwo": 0.62,
        "rozwiedziony": 0.07,
        "wdowiec/wdowa": 0.01,
    },
    "35-44": {
        "kawaler/panna": 0.15,
        "małżeństwo": 0.70,
        "rozwiedziony": 0.13,
        "wdowiec/wdowa": 0.02,
    },
    "45-54": {
        "kawaler/panna": 0.08,
        "małżeństwo": 0.72,
        "rozwiedziony": 0.15,
        "wdowiec/wdowa": 0.05,
    },
    "55-64": {
        "kawaler/panna": 0.06,
        "małżeństwo": 0.70,
        "rozwiedziony": 0.13,
        "wdowiec/wdowa": 0.11,
    },
    "65-74": {
        "kawaler/panna": 0.04,
        "małżeństwo": 0.60,
        "rozwiedziony": 0.10,
        "wdowiec/wdowa": 0.26,
    },
    "75+": {
        "kawaler/panna": 0.03,
        "małżeństwo": 0.35,
        "rozwiedziony": 0.07,
        "wdowiec/wdowa": 0.55,
    },
}

# =============================================================================
# POSIADANIE DZIECI WEDŁUG WIEKU (GUS 2024, szacunki)
# =============================================================================

HAS_CHILDREN_BY_AGE: Dict[str, float] = {
    "18-24": 0.08,
    "25-29": 0.35,
    "30-34": 0.55,
    "35-44": 0.78,
    "45-54": 0.85,
    "55-64": 0.82,
    "65-74": 0.80,
    "75+": 0.78,
}

# Średni wiek urodzenia pierwszego dziecka (GUS 2023)
FIRST_CHILD_AVG_AGE = {"F": 29.0, "M": 33.0}

# =============================================================================
# WIEK EMERYTALNY WEDŁUG PŁCI (ZUS 2024)
# =============================================================================

RETIREMENT_AGE_BY_GENDER: Dict[str, float] = {
    "M": 65.1,  # Mężczyźni
    "F": 60.6,  # Kobiety
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


def get_education_distribution_for_age(age: int) -> Dict[str, float]:
    """Get education level distribution probability for a given age."""
    if age < 25:
        return EDUCATION_BY_AGE["18-24"]
    elif age < 35:
        return EDUCATION_BY_AGE["25-34"]
    elif age < 45:
        return EDUCATION_BY_AGE["35-44"]
    elif age < 55:
        return EDUCATION_BY_AGE["45-54"]
    elif age < 65:
        return EDUCATION_BY_AGE["55-64"]
    elif age < 75:
        return EDUCATION_BY_AGE["65-74"]
    else:
        return EDUCATION_BY_AGE["75+"]


def get_marital_status_distribution_for_age(age: int) -> Dict[str, float]:
    """Get marital status distribution probability for a given age."""
    if age < 25:
        return MARITAL_STATUS_BY_AGE["18-24"]
    elif age < 30:
        return MARITAL_STATUS_BY_AGE["25-29"]
    elif age < 35:
        return MARITAL_STATUS_BY_AGE["30-34"]
    elif age < 45:
        return MARITAL_STATUS_BY_AGE["35-44"]
    elif age < 55:
        return MARITAL_STATUS_BY_AGE["45-54"]
    elif age < 65:
        return MARITAL_STATUS_BY_AGE["55-64"]
    elif age < 75:
        return MARITAL_STATUS_BY_AGE["65-74"]
    else:
        return MARITAL_STATUS_BY_AGE["75+"]


def get_has_children_probability(age: int) -> float:
    """Get probability of having children for a given age."""
    if age < 25:
        return HAS_CHILDREN_BY_AGE["18-24"]
    elif age < 30:
        return HAS_CHILDREN_BY_AGE["25-29"]
    elif age < 35:
        return HAS_CHILDREN_BY_AGE["30-34"]
    elif age < 45:
        return HAS_CHILDREN_BY_AGE["35-44"]
    elif age < 55:
        return HAS_CHILDREN_BY_AGE["45-54"]
    elif age < 65:
        return HAS_CHILDREN_BY_AGE["55-64"]
    elif age < 75:
        return HAS_CHILDREN_BY_AGE["65-74"]
    else:
        return HAS_CHILDREN_BY_AGE["75+"]


def get_occupation_gender_weight(occupation: str, gender: str) -> float:
    """Get gender weight for occupation selection."""
    if occupation in OCCUPATION_GENDER_WEIGHTS:
        return OCCUPATION_GENDER_WEIGHTS[occupation].get(gender, 0.5)
    return 0.5  # default 50/50


def get_min_education_for_occupation(occupation: str) -> str:
    """Get minimum required education level for an occupation."""
    return OCCUPATION_MIN_EDUCATION.get(occupation, "podstawowe")
