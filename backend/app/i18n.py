"""
Internationalization (i18n) module for Market Wizard.

Supports Polish (PL) and English (EN) languages.
"""

from enum import Enum
from typing import Dict, List


class Language(str, Enum):
    """Supported languages."""
    PL = "pl"
    EN = "en"


# =============================================================================
# ANCHOR STATEMENTS (SSR Engine)
# =============================================================================

ANCHOR_SETS: Dict[Language, List[Dict[int, str]]] = {
    Language.PL: [
        {
            1: "Zdecydowanie nie kupię tego produktu",
            2: "Raczej nie kupię tego produktu",
            3: "Nie jestem pewien czy kupiłbym ten produkt",
            4: "Prawdopodobnie kupię ten produkt",
            5: "Zdecydowanie kupię ten produkt",
        },
        {
            1: "Ten produkt mnie w ogóle nie interesuje",
            2: "Raczej nie zdecyduję się na zakup",
            3: "Mogę rozważyć zakup tego produktu",
            4: "Jestem bardzo zainteresowany zakupem",
            5: "Na pewno kupię ten produkt",
        },
        {
            1: "W żadnym wypadku tego nie kupię",
            2: "Wątpię czy zdecyduję się na zakup",
            3: "Mogę kupić, ale mogę też nie kupić",
            4: "Jest duża szansa że to kupię",
            5: "Absolutnie to kupię",
        },
        {
            1: "To zdecydowanie nie jest dla mnie",
            2: "Nie sądzę, żebym potrzebował tego produktu",
            3: "Jestem neutralny wobec zakupu",
            4: "To wygląda na coś, co bym kupił",
            5: "To dokładnie to, czego szukałem",
        },
        {
            1: "Nie mam żadnego zainteresowania tym produktem",
            2: "Jestem sceptyczny wobec zakupu",
            3: "Może kupię, może nie",
            4: "Skłaniam się ku zakupowi",
            5: "Nie mogę się doczekać zakupu",
        },
        {
            1: "To mnie zupełnie nie przyciąga",
            2: "Wolałbym tego nie kupować",
            3: "Jeszcze się nie zdecydowałem czy to kupię",
            4: "Jest dość prawdopodobne że to kupię",
            5: "Bardzo chętnie kupię ten produkt",
        },
    ],
    Language.EN: [
        {
            1: "I definitely won't buy this product",
            2: "I probably won't buy this product",
            3: "I'm not sure if I would buy this product",
            4: "I would probably buy this product",
            5: "I would definitely buy this product",
        },
        {
            1: "This product doesn't interest me at all",
            2: "I'm unlikely to purchase this",
            3: "I might consider buying this",
            4: "I'm quite interested in buying this",
            5: "I will certainly buy this product",
        },
        {
            1: "No way I would ever buy this",
            2: "It's doubtful that I would purchase this",
            3: "I could go either way on buying this",
            4: "There's a good chance I'll buy this",
            5: "I'm absolutely going to buy this",
        },
        {
            1: "This is not for me at all",
            2: "I don't think I need this product",
            3: "I'm neutral about purchasing this",
            4: "This seems like something I would buy",
            5: "This is exactly what I've been looking for",
        },
        {
            1: "I have zero interest in this product",
            2: "I'm skeptical about buying this",
            3: "Maybe I would buy this, maybe not",
            4: "I'm leaning towards buying this",
            5: "I can't wait to buy this product",
        },
        {
            1: "This doesn't appeal to me whatsoever",
            2: "I would rather not buy this",
            3: "I haven't decided if I would buy this",
            4: "I'm fairly likely to purchase this",
            5: "I'm very eager to buy this product",
        },
    ],
}


# =============================================================================
# PERSONA NAMES AND LOCATIONS
# =============================================================================

FIRST_NAMES: Dict[Language, Dict[str, List[str]]] = {
    Language.PL: {
        "M": [
            "Adam", "Piotr", "Tomasz", "Marcin", "Paweł", "Michał", "Krzysztof",
            "Andrzej", "Jan", "Stanisław", "Jakub", "Mateusz", "Łukasz", "Rafał",
            "Sebastian", "Damian", "Kamil", "Bartosz", "Wojciech", "Grzegorz",
        ],
        "F": [
            "Anna", "Maria", "Katarzyna", "Małgorzata", "Agnieszka", "Barbara",
            "Ewa", "Krystyna", "Magdalena", "Monika", "Joanna", "Aleksandra",
            "Dorota", "Natalia", "Karolina", "Sylwia", "Kinga", "Dominika",
            "Beata", "Justyna",
        ],
    },
    Language.EN: {
        "M": [
            "James", "John", "Michael", "David", "Robert", "William", "Richard",
            "Christopher", "Daniel", "Matthew", "Andrew", "Joseph", "Thomas",
            "Charles", "Steven", "Brian", "Kevin", "Jason", "Mark", "Peter",
        ],
        "F": [
            "Mary", "Patricia", "Jennifer", "Elizabeth", "Linda", "Barbara",
            "Susan", "Jessica", "Sarah", "Karen", "Nancy", "Lisa", "Margaret",
            "Betty", "Sandra", "Ashley", "Dorothy", "Kimberly", "Emily", "Donna",
        ],
    },
}

LOCATIONS: Dict[Language, Dict[str, List[str]]] = {
    Language.PL: {
        "urban": [
            "Warszawa", "Kraków", "Wrocław", "Poznań", "Łódź", "Gdańsk",
            "Szczecin", "Lublin", "Katowice", "Białystok",
        ],
        "suburban": [
            "Wieliczka", "Piaseczno", "Pruszków", "Legionowo", "Zabierzów",
            "Marki", "Ząbki", "Sopot", "Rumia", "Reda",
        ],
        "rural": [
            "wieś na Mazurach", "wieś w Wielkopolsce", "wieś na Podlasiu",
            "wieś na Śląsku", "wieś w Małopolsce", "wieś na Kaszubach",
        ],
    },
    Language.EN: {
        "urban": [
            "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
            "Philadelphia", "San Diego", "Dallas", "San Jose", "Austin",
        ],
        "suburban": [
            "Naperville", "Plano", "Irvine", "Frisco", "Cary",
            "Chandler", "Gilbert", "Scottsdale", "Arlington", "Stamford",
        ],
        "rural": [
            "a rural town in Montana", "a small town in Kansas",
            "a farming community in Iowa", "a village in Vermont",
            "a rural area in North Carolina", "a small town in Nebraska",
        ],
    },
}

OCCUPATIONS: Dict[Language, List[str]] = {
    Language.PL: [
        "programista", "nauczyciel", "lekarz", "prawnik", "inżynier",
        "sprzedawca", "kierowca", "pielęgniarka", "fryzjer", "kelner",
        "mechanik", "elektryk", "księgowy", "menedżer", "grafik",
        "architekt", "dentysta", "farmaceuta", "policjant", "strażak",
    ],
    Language.EN: [
        "software developer", "teacher", "doctor", "lawyer", "engineer",
        "sales associate", "driver", "nurse", "hairdresser", "waiter",
        "mechanic", "electrician", "accountant", "manager", "graphic designer",
        "architect", "dentist", "pharmacist", "police officer", "firefighter",
    ],
}


# =============================================================================
# LLM PROMPTS
# =============================================================================

def get_persona_prompt(
    language: Language,
    name: str,
    age: int,
    gender: str,
    location: str,
    income: int,
    occupation: str | None,
    product_description: str,
) -> str:
    """
    Build SSR-compliant prompt for synthetic consumer.
    
    Following the methodology from arxiv:2510.08338:
    - Condition LLM on demographic attributes (persona)
    - Ask for textual purchase intent expression
    - Do NOT ask for arguments or reasoning (that biases responses)
    """
    if language == Language.PL:
        gender_word = "kobieta" if gender == "F" else "mężczyzna"
        occupation_line = f"\nPracujesz jako {occupation}." if occupation else ""
        
        return f"""Jesteś {name}, {age}-letni {gender_word} mieszkający w {location}.
Twój miesięczny dochód to około {income} PLN.{occupation_line}

Rozważ następujący produkt:
{product_description}

Jak bardzo jesteś skłonny/a kupić ten produkt? Odpowiedz naturalnie, tak jak odpowiedziałbyś/odpowiedziałabyś na to pytanie w rozmowie."""

    else:  # EN
        gender_word = "woman" if gender == "F" else "man"
        occupation_line = f"\nYou work as a {occupation}." if occupation else ""
        
        return f"""You are {name}, a {age}-year-old {gender_word} living in {location}.
Your monthly income is about ${income}.{occupation_line}

Consider the following product:
{product_description}

How likely are you to purchase this product? Answer naturally, as you would in a conversation."""


# =============================================================================
# UI LABELS
# =============================================================================

UI_LABELS: Dict[Language, Dict[str, str]] = {
    Language.PL: {
        "app_title": "🔮 Market Wizard",
        "app_subtitle": "Analizator Rynku oparty na metodologii SSR",
        "tab_simulation": "📊 Symulacja Podstawowa",
        "tab_ab_test": "🔬 Test A/B",
        "tab_price": "💰 Analiza Cenowa",
        "tab_about": "ℹ️ O metodologii",
        "product_label": "Opis produktu",
        "product_placeholder": "Np. Pasta do zębów z węglem aktywnym, 75ml, cena 24.99 PLN",
        "target_group": "Grupa docelowa",
        "age_min": "Wiek min",
        "age_max": "Wiek max",
        "gender": "Płeć",
        "gender_all": "Wszystkie",
        "income": "Dochód",
        "income_all": "Wszystkie",
        "income_low": "Niski",
        "income_medium": "Średni",
        "income_high": "Wysoki",
        "location": "Lokalizacja",
        "location_all": "Wszystkie",
        "location_urban": "Miasto",
        "location_suburban": "Przedmieścia",
        "location_rural": "Wieś",
        "n_agents": "Liczba agentów",
        "run_simulation": "🚀 Uruchom symulację",
        "run_ab_test": "🔬 Uruchom test A/B",
        "run_price_analysis": "💰 Analizuj wrażliwość cenową",
        "results_title": "📊 Wyniki Symulacji",
        "mean_purchase_intent": "Średnia intencja zakupu",
        "n_agents_result": "Liczba agentów",
        "distribution": "Rozkład odpowiedzi",
        "scale_1": "Zdecydowanie NIE",
        "scale_2": "Raczej nie",
        "scale_3": "Ani tak, ani nie",
        "scale_4": "Raczej tak",
        "scale_5": "Zdecydowanie TAK",
        "opinions_title": "📝 Przykładowe opinie agentów",
        "variant_a": "Wariant A",
        "variant_b": "Wariant B",
        "price_min": "Cena min (PLN)",
        "price_max": "Cena max (PLN)",
        "price_points": "Punkty cenowe",
        "demand_curve": "Krzywa popytu",
        "optimal_price": "Optymalna cena",
        "elasticity": "Elastyczność cenowa",
        "winner": "Zwycięzca",
        "lift": "Lift",
        "error_no_product": "❌ Wprowadź opis produktu",
        "error_no_variants": "❌ Wprowadź opisy obu wariantów",
        "success": "✅ Symulacja zakończona pomyślnie",
    },
    Language.EN: {
        "app_title": "🔮 Market Wizard",
        "app_subtitle": "Market Analyzer based on SSR methodology",
        "tab_simulation": "📊 Basic Simulation",
        "tab_ab_test": "🔬 A/B Test",
        "tab_price": "💰 Price Analysis",
        "tab_about": "ℹ️ About",
        "product_label": "Product description",
        "product_placeholder": "E.g. Activated charcoal toothpaste, 75ml, price $9.99",
        "target_group": "Target audience",
        "age_min": "Age min",
        "age_max": "Age max",
        "gender": "Gender",
        "gender_all": "All",
        "income": "Income",
        "income_all": "All",
        "income_low": "Low",
        "income_medium": "Medium",
        "income_high": "High",
        "location": "Location",
        "location_all": "All",
        "location_urban": "Urban",
        "location_suburban": "Suburban",
        "location_rural": "Rural",
        "n_agents": "Number of agents",
        "run_simulation": "🚀 Run simulation",
        "run_ab_test": "🔬 Run A/B test",
        "run_price_analysis": "💰 Analyze price sensitivity",
        "results_title": "📊 Simulation Results",
        "mean_purchase_intent": "Mean purchase intent",
        "n_agents_result": "Number of agents",
        "distribution": "Response distribution",
        "scale_1": "Definitely NOT",
        "scale_2": "Probably not",
        "scale_3": "Neutral",
        "scale_4": "Probably yes",
        "scale_5": "Definitely YES",
        "opinions_title": "📝 Sample agent opinions",
        "variant_a": "Variant A",
        "variant_b": "Variant B",
        "price_min": "Price min ($)",
        "price_max": "Price max ($)",
        "price_points": "Price points",
        "demand_curve": "Demand curve",
        "optimal_price": "Optimal price",
        "elasticity": "Price elasticity",
        "winner": "Winner",
        "lift": "Lift",
        "error_no_product": "❌ Please enter a product description",
        "error_no_variants": "❌ Please enter descriptions for both variants",
        "success": "✅ Simulation completed successfully",
    },
}


def get_label(language: Language, key: str) -> str:
    """Get UI label for given language and key."""
    return UI_LABELS.get(language, UI_LABELS[Language.EN]).get(key, key)


def get_anchor_sets(language: Language) -> List[Dict[int, str]]:
    """Get anchor statements for given language."""
    return ANCHOR_SETS.get(language, ANCHOR_SETS[Language.EN])
