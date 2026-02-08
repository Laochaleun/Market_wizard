---
title: Market Wizard
emoji: 🔮
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# Market Wizard 🔮

Analizator rynku i produktu oparty na metodologii **SSR (Semantic Similarity Rating)** z badania [arxiv:2510.08338](https://arxiv.org/abs/2510.08338).

> Lokalna kopia referencyjnego papera (opcjonalna, poza gitem):
> `/Users/pawel/Market_wizard/.local_context/papers/2510.08338v3.pdf`

## 🎯 Co to robi?

Market Wizard symuluje reakcje konsumentów na produkt **bez przeprowadzania rzeczywistych badań ankietowych**:

1. **Generuje syntetycznych konsumentów** - persony z realistycznymi profilami demograficznymi (GUS)
2. **Zbiera opinie** - każda persona ocenia produkt używając AI (Gemini)
3. **Mapuje na skalę Likerta** - odpowiedzi tekstowe → oceny 1-5 przez podobieństwo semantyczne
4. **Agreguje wyniki** - rozkład statystyczny "intencji zakupu" (Purchase Intent)
5. **Generuje raporty** - pełne raporty HTML z wykresami i wszystkimi odpowiedziami

**Kluczowa przewaga SSR:** 90% korelacji z rzeczywistymi decyzjami zakupowymi (vs 80% dla bezpośrednich pytań o liczby).

## 🚀 Szybki start

### 1. Wymagania

- Python 3.11+
- Klucz API Google (Gemini)

### 2. Instalacja

```bash
# Sklonuj repozytorium i przejdź do katalogu
cd Market_wizard

# Utwórz wirtualne środowisko (opcjonalne)
python -m venv venv
source venv/bin/activate

# Zainstaluj projekt
cd backend
pip install -e .
```

### 3. Konfiguracja

```bash
cp backend/.env.example backend/.env
```

Edytuj `backend/.env`:
```env
GOOGLE_API_KEY=your-gemini-api-key-here
LLM_MODEL=gemini-2.0-flash-001
```

### 4. Uruchomienie

```bash
./run.sh gradio
```

Otwórz: **http://localhost:7860**

### 5. API (opcjonalnie)

```bash
./run.sh api
```

Dokumentacja: **http://localhost:8000/docs**

#### Przykładowe zapytania curl (projekty)

```bash
# Lista projektów
curl http://localhost:8000/api/v1/projects

# Utworzenie projektu
curl -X POST http://localhost:8000/api/v1/projects \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Testowy projekt",
    "product_description": "Pasta z węglem aktywnym 75ml, cena 24.99 PLN",
    "target_audience": {
      "age_min": 25,
      "age_max": 45,
      "gender": "F",
      "income_level": "medium",
      "location_type": "urban"
    },
    "research": {}
  }'

# Pobranie projektu
curl http://localhost:8000/api/v1/projects/<ID>

# Aktualizacja projektu
curl -X PUT http://localhost:8000/api/v1/projects/<ID> \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Projekt po aktualizacji",
    "product_description": "Zaktualizowany opis produktu",
    "target_audience": {
      "age_min": 30,
      "age_max": 55,
      "gender": null,
      "income_level": "high",
      "location_type": "suburban"
    },
    "research": {}
  }'

# Usunięcie projektu
curl -X DELETE http://localhost:8000/api/v1/projects/<ID>
```

## 📊 Funkcjonalności

| Funkcja | Opis |
|---------|------|
| 🌐 **Dwujęzyczność** | Polski i angielski (przełącznik w UI) |
| 📊 **Symulacja SSR** | Estymacja intencji zakupu dla grupy docelowej |
| 🔬 **Test A/B** | Porównanie dwóch wariantów produktu |
| 💰 **Analiza cenowa** | Krzywa popytu i elastyczność cenowa |
| 🎯 **Focus Group** | Wirtualne grupy fokusowe z dyskusją multi-agent |
| 👥 **Dane GUS** | Realistyczne rozkłady demograficzne Polski |
| 🗺️ **Filtr regionu (województwo)** | Targetowanie respondentów wg województwa (16 regionów GUS) |
| 📄 **Raporty HTML** | Pełne raporty z wykresami i wszystkimi odpowiedziami |

## 🗂️ Struktura projektu

```
Market_wizard/
├── backend/
│   ├── app/
│   │   ├── config.py                # Konfiguracja
│   │   ├── i18n.py                  # Dwujęzyczność (PL/EN)
│   │   ├── models/                  # Pydantic schemas
│   │   └── services/
│   │       ├── ssr_engine.py        # Algorytm SSR
│   │       ├── llm_client.py        # Gemini LLM
│   │       ├── embedding_client.py  # BGE-M3 embeddingi
│   │       ├── persona_manager.py   # Generator person + GUS
│   │       ├── simulation_engine.py # Orchestrator
│   │       ├── focus_group_engine.py # Focus Groups
│   │       └── report_generator.py  # Raporty HTML
│   ├── .env                         # Zmienne środowiskowe
│   └── pyproject.toml
├── frontend/
│   └── main.py                      # Gradio UI
└── run.sh                           # Skrypt uruchamiania
```

## 🔧 Konfiguracja zaawansowana

### Zmienne środowiskowe

| Zmienna | Opis | Domyślna |
|---------|------|----------|
| `GOOGLE_API_KEY` | **Wymagane** - klucz API Google | - |
| `LLM_MODEL` | Model Gemini | `gemini-2.0-flash-001` |
| `RESEARCH_LLM_MODEL` | Model do groundingu (wyszukiwania źródeł) | `gemini-2.5-flash-lite` |
| `RESEARCH_INTERPRETATION_MODEL` | Model do interpretacji treści źródeł | `gemini-3-flash-preview` |
| `EMBEDDING_MODEL` | Model embeddingów | `BAAI/bge-m3` (lokalny) |
| `EMBEDDING_WARMUP` | Warmup modelu lokalnego (pobranie przy starcie) | `true` |
| `SSR_TEMPERATURE` | Temperatura SSR (zgodna z treningiem kalibratora) | `1.0` |
| `SSR_EPSILON` | Regularizacja epsilon w mapowaniu PMF | `0.0` |
| `SSR_CALIBRATION_ENABLED` | Włączenie kalibracji post-SSR | `true` |
| `SSR_CALIBRATION_ARTIFACT_PATH` | Ścieżka do globalnego kalibratora (`isotonic_v1`) | `backend/app/data/ssr_calibrator_default.json` |
| `SSR_CALIBRATION_POLICY_PATH` | Ścieżka do polityki domenowej (`domain_calibration_v1`) | `backend/app/data/ssr_calibration_policy_default.json` |
| `GUS_API_KEY` | Opcjonalny - dla API GUS | - |

### Modele embeddingów (lokalne)

Wspierane modele:
- `BAAI/bge-m3` (domyślny)
- `all-MiniLM-L6-v2` (opcjonalny, zgodny z SSR tool)

Model lokalny jest automatycznie pobierany przy starcie aplikacji.
> **Uwaga:** Model embeddingów ma istotny wpływ na rozkłady SSR (np. przesunięcie masy w stronę 4–5). Porównuj wyniki tylko przy stałym embeddingu. Szczegóły: `technical_report.md`.

### Modele LLM

Wspierane modele:
- `gemini-2.0-flash-001` (domyślny, szybki)
- `gemini-2.0-pro-001` (lepszy jakościowo)
  
Dla research (źródła i interpretacja):
- `RESEARCH_LLM_MODEL` (grounding) domyślnie `gemini-2.5-flash-lite`
- `RESEARCH_INTERPRETATION_MODEL` (interpretacja danych) domyślnie `gemini-3-flash-preview`

## 👥 Generowanie populacji

System generuje realistyczne persony syntetycznych konsumentów na podstawie oficjalnych danych statystycznych.

### Źródła danych (styczeń 2026)

| Źródło | Dane | Rok |
|--------|------|-----|
| GUS Struktura wynagrodzeń | Zarobki według zawodów, regionów | 2024 |
| GUS BAEL | Struktura zatrudnienia według grup ISCO-08 | 2024 |
| ZUS | Emerytury według płci | 2024 |
| Sedlak & Sedlak | Mediany wynagrodzeń dla zawodów | 2024 |

### Jak działa generowanie person?

Każda persona ma przypisane:
- **Wiek** (18-80 lat) - rozkład oparty na demografii Polski
- **Płeć** (M/F) - rozkład 48%/52%
- **Zawód** - wybierany z wagami populacyjnymi (GUS BAEL)
- **Dochód netto** - obliczany na podstawie zawodu z modyfikatorami
- **Lokalizacja** - miasto/wieś z wpływem na dochód
- **Region (województwo)** - opcjonalny filtr targetowania respondentów

> Ustawienia demograficzne z panelu symulacji (wiek, płeć, dochód, lokalizacja, region)
> są współdzielone przez **Symulację SSR, A/B test, analizę cenową i Focus Group**.
> Wyjątek: liczba uczestników i liczba rund Focus Group są ustawiane osobno.

### Wagi populacyjne zawodów

System nie wybiera zawodów losowo - używa wag opartych na strukturze zatrudnienia:

| Zawód | Udział w populacji | Źródło |
|-------|-------------------|--------|
| Pracownik biurowy | ~10% | GUS BAEL ISCO-4 |
| Sprzedawca | ~8% | GUS BAEL ISCO-5 |
| Kierowca | ~5% | GUS BAEL ISCO-8 |
| Programista | ~4% | GUS BAEL ISCO-2 |
| Lekarz | ~1.2% | GUS BAEL |
| Dentysta | ~0.3% | GUS BAEL |

### Obliczanie dochodu netto

Dochód jest obliczany z uwzględnieniem wielu czynników:

```
dochód = dochód_bazowy × współczynnik_doświadczenia 
         × współczynnik_płci × współczynnik_regionu 
         × współczynnik_lokalizacji ± wariacja
```

| Modyfikator | Zakres | Źródło |
|-------------|--------|--------|
| Doświadczenie | 0.0 → 1.0 (20 lat) | Model |
| Płeć | M: +8.5%, F: -8.5% | GUS 2024 |
| Region | Mazowieckie +16%, Podkarpackie -14% | GUS 2024 |
| Lokalizacja | miasto +8%, wieś -12% | GUS BAEL |

### Tryb offline (gdy GUS API niedostępne)

Gdy API GUS jest niedostępne (błąd 403, timeout, brak klucza), system używa **wbudowanych danych referencyjnych** z pliku `backend/app/data/reference_data.py`:

```python
# Przykładowe dane wbudowane
REGIONAL_WAGE_INDEX = {
    "mazowieckie": 1.16,  # +16%
    "podkarpackie": 0.86, # -14%
    # ... 16 województw
}

PENSION_BY_GENDER = {
    "M": {"median": 3975, "std": 1000},  # netto
    "F": {"median": 2730, "std": 750},   # netto
}

OCCUPATION_INCOME_DATA = {
    "programista": {"median": 9000, "p25": 5500, "p75": 16000},
    "sprzedawca": {"median": 3200, "p25": 2800, "p75": 4200},
    # ... wszystkie zawody
}
```

> **Uwaga:** Wszystkie kwoty w systemie są w **PLN netto miesięcznie**.

### Aktualizacja danych

Dane referencyjne znajdują się w:
- `backend/app/data/reference_data.py` - współczynniki i zarobki
- `backend/app/i18n.py` - lista zawodów z zakresami wiekowymi

Aby zaktualizować dane po publikacji nowych raportów GUS:
1. Edytuj `reference_data.py`
2. Zaktualizuj komentarze ze źródłami
3. Skalibruj mnożnik luki płacowej: `make calibrate`
4. Uruchom testy: `python scripts/test_personas.py`

## 📚 Metodologia SSR

Oparta na badaniu: **Maier, B.F., et al. (2025).** *"LLMs Reproduce Human Purchase Intent via Semantic Similarity Elicitation of Likert Ratings"* [arXiv:2510.08338](https://arxiv.org/abs/2510.08338)

### Jak działa?

1. **Tekstowa elicytacja** - LLM odpowiada naturalnym tekstem (nie liczbą)
2. **Anchor statements** - 6 zestawów zdań reprezentujących skalę 1-5
3. **Embedddingi** - tekst → wektor (lokalnie `BAAI/bge-m3` lub OpenAI `text-embedding-3-small`)
4. **Cosine similarity** - porównanie z kotwicami → rozkład PMF
5. **Agregacja** - średnia z wielu agentów
6. **Intent-only SSR** - do punktacji używana jest krótka deklaracja intencji zakupu; dłuższe odpowiedzi są tylko do wniosków jakościowych

### Dlaczego SSR?

| Metoda | Korelacja z rzeczywistością |
|--------|---------------------------|
| Bezpośrednie pytanie "1-5" | ~80% |
| **SSR (ta aplikacja)** | **~90%** |
| | |

### 🌡️ Temperatura (Precision)

Parametr `temperature` kontroluje "zdecydowanie" modelu w ocenach.

*   **1.0 (Domyślnie w aplikacji i w artykule)**: Wyniki są bardziej wygładzone, "bezpieczne". Model unika skrajności (1 i 5).
*   **Niższe wartości**: Wyniki bardziej "ostre", większa skłonność do skrajności.

## 🧭 Stage 1 Calibration (2026-02-07)

Pierwszy etap kalibracji został wdrożony end-to-end: od treningu kalibratorów, przez runtime, po zewnętrzną walidację produkcyjną.

### Co jest wdrożone

- **Globalna kalibracja post-SSR (isotonic)** z walidacją OOF/holdout w:
  - `backend/scripts/tune_ssr_hierarchical.py`
- **Polityka domenowa kalibracji** (`domain_calibration_v1`) z routingiem:
  - `backend/app/services/score_calibration.py`
  - `backend/app/services/ssr_engine.py`
- **Routing domeny w runtime**:
  - `SimulationEngine` używa `domain_hint="ecommerce"` dla głównego scoringu PI.
- **Artefakty fallback dla HF Spaces** (bez polegania na lokalnym `.env`):
  - `backend/app/data/ssr_calibrator_default.json`
  - `backend/app/data/ssr_calibration_policy_default.json`

### Runtime defaults (spójne z treningiem kalibratora)

- `SSR_TEMPERATURE=1.0`
- `SSR_EPSILON=0.0`
- `SSR_CALIBRATION_ENABLED=true`
- `SSR_CALIBRATION_ARTIFACT_PATH=backend/app/data/ssr_calibrator_default.json`
- `SSR_CALIBRATION_POLICY_PATH=backend/app/data/ssr_calibration_policy_default.json`

### Skrypty etapu 1

1. Trening/benchmark + raport kalibracji:

```bash
cd /Users/pawel/Market_wizard
PYTHONPATH=/Users/pawel/Market_wizard/backend python backend/scripts/tune_ssr_hierarchical.py \
  --model BAAI/bge-m3 \
  --language en \
  --anchor-language-mode auto \
  --global-calibration isotonic \
  --calibration-cv-folds 5 \
  --calibration-holdout-ratio 0.2
```

2. Budowa domenowej polityki kalibracji:

```bash
cd /Users/pawel/Market_wizard
PYTHONPATH=/Users/pawel/Market_wizard/backend python backend/scripts/build_domain_calibration_policy.py \
  --model BAAI/bge-m3 \
  --temperature 1.0 \
  --epsilon 0.0 \
  --optimize off1 \
  --out /Users/pawel/Market_wizard/backend/app/data/ssr_calibration_policy_default.json
```

3. Zewnętrzna walidacja gotowości produkcyjnej:

```bash
cd /Users/pawel/Market_wizard
PYTHONPATH=/Users/pawel/Market_wizard/backend python backend/scripts/validate_production_readiness.py \
  --model BAAI/bge-m3 \
  --temperature 1.0 \
  --epsilon 0.0 \
  --calibrator-path /Users/pawel/Market_wizard/backend/app/data/ssr_calibrator_default.json \
  --policy-path /Users/pawel/Market_wizard/backend/app/data/ssr_calibration_policy_default.json \
  --report-out /Users/pawel/Market_wizard/reports/production_readiness_validation_2026-02-07.md
```

### Status po Stage 1

- Najlepsza polityka z testowanych: `ecommerce_only_calibrated`.
- Zewnętrzne bramki produkcyjne wciąż niezaliczone (`FAIL`), głównie przez:
  - `Off-by-one < 0.92`,
  - `MAE > 0.60`.
- Wniosek: Stage 1 dostarcza infrastrukturę i realną poprawę metryk, ale nie kończy tematu „production-ready”.

## 🧪 Testy zgodności z `semantic-similarity-rating`

Poniżej minimalny zestaw kroków do potwierdzenia zgodności rdzenia SSR między:
- `Market_wizard` (`backend/app/services/ssr_engine.py`)
- `/Users/pawel/semantic-similarity-rating/semantic_similarity_rating`

### 1) Testy SSR w Market Wizard

```bash
cd /Users/pawel/Market_wizard/backend
pytest -q tests/test_ssr_engine.py
```

### 2) Testy referencyjnego repo

```bash
cd /Users/pawel/semantic-similarity-rating
PYTHONDONTWRITEBYTECODE=1 ./.venv/bin/pytest -q -p no:cacheprovider tests/test_compute.py tests/test_response_rater.py
```

### 3) Numeryczne porównanie rdzenia PMF/temperature (1:1)

Uruchom poniższy skrypt z repo `Market_wizard`:

```bash
cd /Users/pawel/Market_wizard
python - <<'PY'
import numpy as np
import importlib.util

spec = importlib.util.spec_from_file_location(
    "ssr_compute",
    "/Users/pawel/semantic-similarity-rating/semantic_similarity_rating/compute.py",
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
ref_pmf = mod.response_embeddings_to_pmf
ref_scale = mod.scale_pmf

def mw_pmf(response_embeddings, likert_embeddings, epsilon=0.0):
    M_left = response_embeddings
    M_right = likert_embeddings
    if M_left.shape[0] == 0:
        return np.empty((0, M_right.shape[1]))
    norm_right = np.linalg.norm(M_right, axis=0)
    M_right = M_right / norm_right[None, :]
    norm_left = np.linalg.norm(M_left, axis=1)
    M_left = M_left / norm_left[:, None]
    cos = (1 + M_left.dot(M_right)) / 2
    cos_min = cos.min(axis=1)[:, None]
    numerator = cos - cos_min
    if epsilon > 0:
        mins = np.argmin(cos, axis=1)
        for i, j in enumerate(mins):
            numerator[i, j] += epsilon
    den = cos.sum(axis=1)[:, None] - cos.shape[1] * cos_min + epsilon
    return numerator / den

def mw_scale(pmf, temperature):
    pmf = np.asarray(pmf, dtype=float)
    if temperature == 1.0:
        return pmf
    if temperature == 0.0:
        if np.all(pmf == pmf[0]):
            return pmf
        out = np.zeros_like(pmf)
        out[np.argmax(pmf)] = 1.0
        return out
    hist = pmf ** (1 / temperature)
    return hist / hist.sum()

rng = np.random.default_rng(123)
for eps in [0.0, 1e-6, 0.01, 0.2]:
    for _ in range(50):
        r = rng.normal(size=(6, 384))
        l = rng.normal(size=(384, 5))
        assert np.allclose(ref_pmf(r, l, epsilon=eps), mw_pmf(r, l, epsilon=eps), atol=1e-12, rtol=1e-12)

for t in [0.0, 0.1, 1.0, 2.0, 10.0]:
    for _ in range(50):
        p = rng.random(5); p = p / p.sum()
        assert np.allclose(ref_scale(p, t), mw_scale(p, t), atol=1e-12, rtol=1e-12)

print("OK: PMF and temperature scaling are numerically equivalent.")
PY
```

Oczekiwany rezultat: brak assertion error i komunikat `OK: PMF and temperature scaling are numerically equivalent.`

### 4) Benchmark na danych rzeczywistych (wpływ embeddingów)

Skrypt poniżej używa realnych danych z `amazon_reviews_multi` (Hugging Face, test split, EN)
i mierzy zgodność SSR z etykietami 1-5 (MAE, Spearman, accuracy), a także różnice
między modelami embeddingów:

```bash
cd /Users/pawel/Market_wizard/backend
python scripts/evaluate_ssr_on_real_data.py \
  --limit 1200 \
  --language en \
  --models all-MiniLM-L6-v2,BAAI/bge-m3
```

Opcjonalnie można zapisać predykcje per-próbka:

```bash
python scripts/evaluate_ssr_on_real_data.py \
  --limit 1200 \
  --language en \
  --models all-MiniLM-L6-v2,BAAI/bge-m3 \
  --csv-out /tmp/ssr_real_data_eval.csv
```

## 📄 Raporty

Po uruchomieniu symulacji możesz wygenerować pełny raport HTML zawierający:

- 📦 Opis analizowanego produktu
- 📊 Średnia intencja zakupu + odchylenie standardowe
- 📈 Wykres rozkładu intencji (słupkowy)

## ✅ TODO

- [ ] Integracja zewnętrznych źródeł danych dochodów (np. Eurostat API lub plik CSV/Excel z BAEL/GUS), aby zasilać rozkład dochodów wg wieku/regionu zamiast obecnego modelu syntetycznego.
- 👥 Profil demograficzny (wiek, dochód, płeć)
- 💰 Wykres korelacji dochód ↔ intencja
- 📝 **Wszystkie odpowiedzi agentów** z ocenami SSR

Raport można otworzyć w przeglądarce i wydrukować.

## 📄 Licencja

MIT
