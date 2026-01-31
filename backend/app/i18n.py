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
        "metropolis": [
            "Warszawa", "Kraków", "Łódź", "Wrocław", "Poznań",
        ],
        "large_city": [
            "Gdańsk", "Szczecin", "Bydgoszcz", "Lublin", "Białystok", "Katowice",
            "Gdynia", "Częstochowa", "Radom", "Toruń", "Kielce", "Rzeszów",
        ],
        "medium_city": [
            "Tychy", "Opole", "Gorzów Wielkopolski", "Płock", "Elbląg",
            "Wałbrzych", "Włocławek", "Tarnów", "Chorzów", "Koszalin",
            "Słupsk", "Legnica", "Suwałki", "Jelenia Góra", "Siedlce",
        ],
        "small_city": [
            "Wieliczka", "Piaseczno", "Pruszków", "Ząbki", "Rumia",
            "Zakopane", "Sopot", "Augustów", "Kołobrzeg", "Sandomierz",
            "Ciechanów", "Kwidzyn", "Żywiec", "Nysa", "Bochnia",
        ],
        "rural": [
            "wieś na Mazowszu", "wieś w Małopolsce", "wieś na Podkarpaciu",
            "wieś na Śląsku", "wieś w Wielkopolsce", "wieś na Pomorzu",
            "wieś na Warmii", "wieś na Podlasiu", "wieś w Świętokrzyskiem",
            "wieś na Lubelszczyźnie", "wieś w Łódzkiem", "wieś na Dolnym Śląsku",
        ],
        # Legacy fallback
        "urban": ["Warszawa", "Kraków", "Gdańsk"],
        "suburban": ["Piaseczno", "Sopot", "Wieliczka"],
    },
    Language.EN: {
        "metropolis": [
            "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
        ],
        "large_city": [
            "Seattle", "Denver", "Boston", "Nashville", "Portland", "Las Vegas",
        ],
        "medium_city": [
            "Salt Lake City", "Boise", "Tucson", "Fresno", "Spokane",
        ],
        "small_city": [
            "Santa Fe", "Boulder", "Ann Arbor", "Asheville", "Key West",
        ],
        "rural": [
            "rural Texas", "rural Ohio", "rural Iowa", "rural Oregon",
            "rural Alabama", "rural Montana", "rural Vermont",
        ],
        # Legacy fallback
        "urban": ["New York", "Chicago"],
        "suburban": ["Naperville", "Pasadena"],
    },
}

OCCUPATIONS: Dict[Language, List[Dict[str, any]]] = {
    Language.PL: [
        # Zawody wymagające studiów wyższych (min. 5-6 lat)
        {"name": "lekarz", "min_age": 26, "max_age": 70, "income_min": 6000, "income_max": 20000},
        {"name": "dentysta", "min_age": 26, "max_age": 70, "income_min": 7000, "income_max": 25000},
        {"name": "prawnik", "min_age": 24, "max_age": 70, "income_min": 5000, "income_max": 18000},
        {"name": "architekt", "min_age": 26, "max_age": 70, "income_min": 5500, "income_max": 15000},
        {"name": "farmaceuta", "min_age": 25, "max_age": 70, "income_min": 5500, "income_max": 12000},
        {"name": "programista", "min_age": 22, "max_age": 65, "income_min": 6000, "income_max": 25000},
        {"name": "inżynier", "min_age": 23, "max_age": 70, "income_min": 5000, "income_max": 15000},
        {"name": "księgowy", "min_age": 23, "max_age": 70, "income_min": 4000, "income_max": 12000},
        {"name": "menedżer", "min_age": 28, "max_age": 65, "income_min": 7000, "income_max": 25000},
        {"name": "nauczyciel", "min_age": 23, "max_age": 67, "income_min": 4000, "income_max": 8000},
        {"name": "grafik", "min_age": 21, "max_age": 65, "income_min": 3500, "income_max": 12000},
        {"name": "pielęgniarka", "min_age": 22, "max_age": 67, "income_min": 4500, "income_max": 8000},
        
        # Zawody bez wymagań wyższego wykształcenia
        {"name": "sprzedawca", "min_age": 18, "max_age": 65, "income_min": 2800, "income_max": 5000},
        {"name": "kierowca", "min_age": 21, "max_age": 67, "income_min": 3500, "income_max": 7000},
        {"name": "fryzjer", "min_age": 18, "max_age": 65, "income_min": 2500, "income_max": 6000},
        {"name": "kelner", "min_age": 18, "max_age": 55, "income_min": 2500, "income_max": 4500},
        {"name": "mechanik", "min_age": 18, "max_age": 65, "income_min": 3000, "income_max": 7000},
        {"name": "elektryk", "min_age": 18, "max_age": 65, "income_min": 3500, "income_max": 8000},
        {"name": "policjant", "min_age": 21, "max_age": 60, "income_min": 4500, "income_max": 9000},
        {"name": "strażak", "min_age": 21, "max_age": 55, "income_min": 4500, "income_max": 8000},
        {"name": "pracownik biurowy", "min_age": 19, "max_age": 67, "income_min": 3500, "income_max": 7000},
        {"name": "pracownik budowlany", "min_age": 18, "max_age": 60, "income_min": 3500, "income_max": 8000},
        {"name": "kucharz", "min_age": 18, "max_age": 65, "income_min": 3000, "income_max": 7000},
        
        # Statusy specjalne (wiek/sytuacja życiowa)
        {"name": "student", "min_age": 18, "max_age": 27, "income_min": 0, "income_max": 2500},
        {"name": "emeryt", "min_age": 60, "max_age": 100, "income_min": 2000, "income_max": 4500},
        {"name": "rencista", "min_age": 35, "max_age": 100, "income_min": 1800, "income_max": 3500},
    ],
    Language.EN: [
        # Professions requiring higher education
        {"name": "doctor", "min_age": 26, "max_age": 70, "income_min": 8000, "income_max": 25000},
        {"name": "dentist", "min_age": 26, "max_age": 70, "income_min": 9000, "income_max": 30000},
        {"name": "lawyer", "min_age": 24, "max_age": 70, "income_min": 6000, "income_max": 20000},
        {"name": "architect", "min_age": 26, "max_age": 70, "income_min": 5500, "income_max": 15000},
        {"name": "pharmacist", "min_age": 25, "max_age": 70, "income_min": 6000, "income_max": 12000},
        {"name": "software developer", "min_age": 22, "max_age": 65, "income_min": 7000, "income_max": 25000},
        {"name": "engineer", "min_age": 23, "max_age": 70, "income_min": 5500, "income_max": 15000},
        {"name": "accountant", "min_age": 23, "max_age": 70, "income_min": 4500, "income_max": 12000},
        {"name": "manager", "min_age": 28, "max_age": 65, "income_min": 8000, "income_max": 25000},
        {"name": "teacher", "min_age": 23, "max_age": 67, "income_min": 4000, "income_max": 8000},
        {"name": "graphic designer", "min_age": 21, "max_age": 65, "income_min": 4000, "income_max": 12000},
        {"name": "nurse", "min_age": 22, "max_age": 67, "income_min": 5000, "income_max": 9000},
        
        # Professions without higher education requirement
        {"name": "sales associate", "min_age": 18, "max_age": 65, "income_min": 2500, "income_max": 5000},
        {"name": "driver", "min_age": 21, "max_age": 67, "income_min": 3500, "income_max": 7000},
        {"name": "hairdresser", "min_age": 18, "max_age": 65, "income_min": 2500, "income_max": 6000},
        {"name": "waiter", "min_age": 18, "max_age": 55, "income_min": 2500, "income_max": 5000},
        {"name": "mechanic", "min_age": 18, "max_age": 65, "income_min": 3500, "income_max": 8000},
        {"name": "electrician", "min_age": 18, "max_age": 65, "income_min": 4000, "income_max": 9000},
        {"name": "police officer", "min_age": 21, "max_age": 60, "income_min": 5000, "income_max": 10000},
        {"name": "firefighter", "min_age": 21, "max_age": 55, "income_min": 5000, "income_max": 9000},
        {"name": "office worker", "min_age": 19, "max_age": 67, "income_min": 3500, "income_max": 7000},
        {"name": "construction worker", "min_age": 18, "max_age": 60, "income_min": 3500, "income_max": 8000},
        {"name": "chef", "min_age": 18, "max_age": 65, "income_min": 3500, "income_max": 8000},
        
        # Special statuses (age/life situation)
        {"name": "student", "min_age": 18, "max_age": 27, "income_min": 0, "income_max": 2500},
        {"name": "retiree", "min_age": 60, "max_age": 100, "income_min": 2000, "income_max": 5000},
        {"name": "disability pensioner", "min_age": 35, "max_age": 100, "income_min": 1800, "income_max": 4000},
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
# REPORT ANALYSIS PROMPTS
# =============================================================================

def get_report_analysis_prompt(language: Language, payload_json: str) -> str:
    """Build prompt for narrative report analysis from simulation data."""
    if language == Language.PL:
        return (
            "Rola:\n"
            "Jesteś Ekspertem Strategy & Business Intelligence. Twoim zadaniem jest \"wycisniecie\" z raportu SSR "
            "wnioskow, ktorych nie widac na pierwszy rzut oka. Nie streszczaj - interpretuj.\n\n"
            "Zadanie:\n"
            "Przeanalizuj dane i zwroc wynik w formacie JSON.\n\n"
            "Wynik:\n"
            "Zwroc WYLACZNIE poprawny obiekt JSON (bez blokow kodu i bez backtickow). Schemat:\n"
            '{'
            '"narrative":"Glowna analiza (Markdown)",'
            '"agent_summary":"Lista kluczowych faktow (String z lista punktowana)",'
            '"recommendations":"Konkretne dzialania biznesowe (String z lista punktowana)"'
            "}\n\n"
            "Wytyczne do tresci (Instrukcja \"Jak myslec\"):\n\n"
            "1. SEKCJA \"narrative\" (To jest serce raportu, ok. 3000-4500 znakow):\n"
            "   Pisz stylem eseju biznesowego, ale neutralnym i profesjonalnym. "
            "Uzywaj jezyka literalnego i faktow; unikaj metafor, personifikacji i porownan. "
            "Jesli pojawia sie figura stylistyczna, zastap ja opisem zachowania lub wniosku. "
            "Nie tworz nazw segmentow ani etykiet - segmenty opisuj przez cechy "
            "(zawod + dochod + lokalizacja + postawa). "
            "Unikaj pytan retorycznych; uzywaj zdan deklaratywnych. "
            "Kazdy akapit musi zawierac co najmniej jeden element weryfikowalny z danych "
            "(np. liczba, zawod, lokalizacja, dochod). "
            "Podziel na sekcje (uzyj Markdown **Naglowek** i \\n\\n):\n\n"
            "   A. **Psychologia Odbioru i Sentyment:**\n"
            "      - Nie pisz tylko \"jest pozytywnie\". Wyjasnij mechanizmy stojace za ocena "
            "(np. \"fair deal\" lub \"jakosc materialu\" - tylko jesli takie watki rzeczywiscie wystepuja w danych).\n"
            "      - Zanalizuj funkcje produktu: czy to dekoracja, narzedzie spoleczne, prezent, czy cos innego "
            "- ale tylko jesli wynika to z danych.\n"
            "      - Wskaz grupy zawodowe, dla ktorych produkt pelni role \"wentylu bezpieczenstwa\", "
            "jesli takie wskazania sie pojawiaja.\n\n"
            "   B. **Anomalie i Segmentacja:**\n"
            "      - Znajdz sprzecznosci i odchylenia: np. roznice miedzy zawodami estetycznymi a "
            "pragmatycznymi, jesli sa obecne w danych.\n"
            "      - Kto waha sie najbardziej i dlaczego?\n\n"
            "   C. **Strategia i Pozycjonowanie:**\n"
            "      - Zrekonstruuj obecna strategie na podstawie danych (np. jesli pojawiaja sie slowa-klucze "
            "dotyczace jakosci materialu lub funkcji prezentowej).\n"
            "      - Zdefiniuj grupe docelowa psychograficznie (np. dystans do siebie vs \"sztywniacy\"), "
            "jesli mozna to wyczytac z wypowiedzi.\n\n"
            "   D. **Wnioski Koncowe:** Synteza prowadzaca do rekomendacji.\n\n"
            "2. SEKCJA \"agent_summary\" (Konkrety):\n"
            "   - Wypunktuj twarde fakty i powtarzalne wzorce z danych "
            "(np. wzmianki o konkurencji, akceptowane poziomy cen, recurring phrases).\n"
            "   - Nie wymyslaj: jesli jakiegos typu danych brak, pomin.\n\n"
            "3. SEKCJA \"recommendations\" (Strategia):\n"
            "   - Nie dawaj porad typu \"zrob reklamy\". Badz precyzyjny.\n"
            "   - Wymien: potencjal cross-sellingu (co dokupic? zestawy?), sugestie targetowania "
            "(jakie zawody/grupy?), argumenty sprzedazowe (co uwypuklic na Landing Page?).\n"
            "   - Kazda rekomendacja MUSI zakonczyc sie jednym z dopiskow: "
            "\"(wsparte danymi)\" albo \"(sygnal do weryfikacji w szerszej probie)\".\n"
            "   - Uzyj \"(wsparte danymi)\" tylko wtedy, gdy masz wyrazne wsparcie w danych "
            "(np. >=3 niezalezne przyklady). W przeciwnym razie uzyj "
            "\"(sygnal do weryfikacji w szerszej probie)\".\n\n"
            "Zasady techniczne:\n"
            "- Jezyk: Polski.\n"
            "- Formatowanie: Markdown wewnatrz stringow JSON.\n"
            "- Wiernosc: Opieraj sie tylko na dostarczonych danych wejsciowych (JSON SSR).\n\n"
            f"DANE:\n{payload_json}"
        )

    return (
        "Role:\n"
        "You are a Strategy & Business Intelligence Expert. Your task is to \"squeeze\" insights out of the provided "
        "SSR market research data that are not immediately obvious. Do not just summarize - interpret.\n\n"
        "Task:\n"
        "Analyze the input data and return the result in JSON format.\n\n"
        "Output:\n"
        "Return ONLY a valid JSON object (no code blocks, no backticks). Schema:\n"
        '{'
        '"narrative":"Main analysis (Markdown format)",'
        '"agent_summary":"List of key facts (String with a bulleted list)",'
        '"recommendations":"Concrete business actions (String with a bulleted list)"'
        "}\n\n"
        "Content Guidelines (\"How to think\"):\n\n"
        "1. SECTION \"narrative\" (The heart of the report, approx. 3000-4500 characters):\n"
        "   Write in a business-essay style, but neutral and professional. "
        "Use literal language and facts; avoid metaphors, personification, and comparisons. "
        "If a figure of speech appears, replace it with a literal behavioral description or conclusion. "
        "Do not create segment names or labels - describe segments by attributes "
        "(occupation + income + location + attitude). "
        "Avoid rhetorical questions; use declarative sentences. "
        "Each paragraph must include at least one verifiable data element "
        "(e.g., number, occupation, location, income). "
        "Divide into sections (use Markdown **Header** and \\n\\n):\n\n"
        "   A. **Psychology of Perception & Sentiment:**\n"
        "      - Don't just say \"positive.\" Explain the mechanisms behind the evaluation "
        "(e.g., \"fair deal\" or \"material quality\" - only if such themes actually appear in the data).\n"
        "      - Analyze the product function: decor vs social tool vs gift - but only if supported by the responses.\n"
        "      - Identify professional groups for whom the product acts as a \"safety valve,\" if such signals are present.\n\n"
        "   B. **Anomalies & Segmentation:**\n"
        "      - Find contradictions and deviations: e.g., differences between design-sensitive vs pragmatic professions, "
        "if present in the data.\n"
        "      - Who hesitates the most and why?\n\n"
        "   C. **Strategy & Positioning:**\n"
        "      - Reconstruct the current strategy from the data (e.g., if keyword signals about material quality "
        "or gift value appear).\n"
        "      - Define the target audience psychographically (e.g., self-distance vs \"stiff\" corporate types), "
        "only if evidenced in responses.\n\n"
        "   D. **Strategic Synthesis:** Summary leading into recommendations.\n\n"
        "2. SECTION \"agent_summary\" (Hard Facts):\n"
        "   - Bullet point hard facts and recurring patterns from the data "
        "(e.g., competitor mentions, acceptable price levels, recurring phrases).\n"
        "   - Do not invent; if a data type is missing, omit it.\n\n"
        "3. SECTION \"recommendations\" (Strategy):\n"
        "   - Do not give generic advice like \"run ads.\" Be precise.\n"
        "   - List: cross-selling potential (what to bundle?), targeting suggestions (which professions/groups?), "
        "sales arguments (what to highlight on the landing page?).\n"
        "   - Every recommendation MUST end with one of these suffixes: "
        "\"(supported by data)\" or \"(signal to validate with a broader sample)\".\n"
        "   - Use \"(supported by data)\" only when there is clear support in the data "
        "(e.g., >=3 independent examples). Otherwise use "
        "\"(signal to validate with a broader sample)\".\n\n"
        "Technical Rules:\n"
        "- Language: English.\n"
        "- Formatting: Markdown inside JSON strings.\n"
        "- Fidelity: Use only the provided SSR input data (JSON).\n\n"
        f"DATA:\n{payload_json}"
    )


def get_report_analysis_sanitize_prompt(language: Language, analysis_json: str) -> str:
    """Build prompt to sanitize report analysis into a literal, neutral style."""
    if language == Language.PL:
        return (
            "Zadanie: Oczyść styl tekstu analizy bez zmiany faktów.\n"
            "Wejście: JSON z polami narrative, agent_summary, recommendations.\n"
            "Wyjście: ZWRÓĆ WYŁĄCZNIE poprawny JSON o tym samym schemacie.\n\n"
            "Zasady:\n"
            "- Nie dodawaj nowych informacji ani wniosków.\n"
            "- Nie usuwaj faktów, liczb, zawodów, lokalizacji ani cytowanych przykładów.\n"
            "- Zamień metafory, personifikacje i storytelling na opis literalny.\n"
            "- Nie używaj etykiet dla grup (np. \"segment X\"). Opisuj je przez cechy.\n"
            "- Unikaj pytań retorycznych. Używaj zdań deklaratywnych.\n"
            "- Zachowaj strukturę akapitów i list.\n"
            "- Nie zmieniaj ani nie usuwaj suffixów w rekomendacjach "
            "(np. \"(wsparte danymi)\" / \"(sygnal do weryfikacji w szerszej probie)\").\n\n"
            f"JSON:\n{analysis_json}"
        )
    return (
        "Task: Sanitize the analysis style without changing facts.\n"
        "Input: JSON with fields narrative, agent_summary, recommendations.\n"
        "Output: RETURN ONLY valid JSON with the same schema.\n\n"
        "Rules:\n"
        "- Do not add new information or conclusions.\n"
        "- Do not remove facts, numbers, occupations, locations, or cited examples.\n"
        "- Replace metaphors, personification, and storytelling with literal descriptions.\n"
        "- Replace playful labels or nicknames (e.g., \"Neighbor War\", \"Office Prank\") with neutral descriptions.\n"
        "- Do not use group labels (e.g., \"segment X\"). Describe by attributes.\n"
        "- Avoid rhetorical questions. Use declarative sentences.\n"
        "- Preserve paragraph and list structure.\n"
        "- Do not change or remove recommendation suffixes "
        "(e.g., \"(supported by data)\" / \"(signal to validate with a broader sample)\").\n\n"
        f"JSON:\n{analysis_json}"
    )

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
        "extract_url": "🔗 Pobierz z URL",
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
        "extract_url": "🔗 Fetch from URL",
    },
}


def get_label(language: Language, key: str) -> str:
    """Get UI label for given language and key."""
    return UI_LABELS.get(language, UI_LABELS[Language.EN]).get(key, key)


def get_anchor_sets(language: Language) -> List[Dict[int, str]]:
    """Get anchor statements for given language."""
    return ANCHOR_SETS.get(language, ANCHOR_SETS[Language.EN])
