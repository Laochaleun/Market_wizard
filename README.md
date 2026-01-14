# Market Wizard 🔮

Analizator rynku i produktu oparty na metodologii **SSR (Semantic Similarity Rating)** z badania [arxiv:2510.08338](https://arxiv.org/abs/2510.08338).

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

## 📊 Funkcjonalności

| Funkcja | Opis |
|---------|------|
| 🌐 **Dwujęzyczność** | Polski i angielski (przełącznik w UI) |
| 📊 **Symulacja SSR** | Estymacja intencji zakupu dla grupy docelowej |
| 🔬 **Test A/B** | Porównanie dwóch wariantów produktu |
| 💰 **Analiza cenowa** | Krzywa popytu i elastyczność cenowa |
| 👥 **Dane GUS** | Realistyczne rozkłady demograficzne Polski |
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
| `EMBEDDING_MODEL` | Model embeddingów | `BAAI/bge-m3` (lokalny) |
| `GUS_API_KEY` | Opcjonalny - dla API GUS | - |

### Modele LLM

Wspierane modele:
- `gemini-2.0-flash-001` (domyślny, szybki)
- `gemini-2.0-pro-001` (lepszy jakościowo)

## 📚 Metodologia SSR

Oparta na badaniu: **Maier, B.F., et al. (2025).** *"LLMs Reproduce Human Purchase Intent via Semantic Similarity Elicitation of Likert Ratings"* [arXiv:2510.08338](https://arxiv.org/abs/2510.08338)

### Jak działa?

1. **Tekstowa elicytacja** - LLM odpowiada naturalnym tekstem (nie liczbą)
2. **Anchor statements** - 6 zestawów zdań reprezentujących skalę 1-5
3. **Embedddingi** - tekst → wektor (BGE-M3)
4. **Cosine similarity** - porównanie z kotwicami → rozkład PMF
5. **Agregacja** - średnia z wielu agentów

### Dlaczego SSR?

| Metoda | Korelacja z rzeczywistością |
|--------|---------------------------|
| Bezpośrednie pytanie "1-5" | ~80% |
| **SSR (ta aplikacja)** | **~90%** |

## 📄 Raporty

Po uruchomieniu symulacji możesz wygenerować pełny raport HTML zawierający:

- 📦 Opis analizowanego produktu
- 📊 Średnia intencja zakupu + odchylenie standardowe
- 📈 Wykres rozkładu intencji (słupkowy)
- 👥 Profil demograficzny (wiek, dochód, płeć)
- 💰 Wykres korelacji dochód ↔ intencja
- 📝 **Wszystkie odpowiedzi agentów** z ocenami SSR

Raport można otworzyć w przeglądarce i wydrukować.

## 📄 Licencja

MIT
