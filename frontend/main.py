"""
Market Wizard - Gradio Frontend

Interactive dashboard for running SSR-based market research simulations.
Supports Polish (PL) and English (EN) languages.
"""

import asyncio
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import gradio as gr
import numpy as np
import pandas as pd

# Add backend to path
import sys
from pathlib import Path as PathlibPath

# Insert backend path at position 0 to prioritize it
backend_path = str(PathlibPath(__file__).parent.parent / "backend")
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

# Project data lives at repo root regardless of import resolution.
PROJECT_BASE_DIR = PathlibPath(__file__).resolve().parents[1]

# Now import from backend app package
from app.models import DemographicProfile, SimulationResult
from app.services import (
    SimulationEngine,
    ABTestEngine,
    PriceSensitivityEngine,
    FocusGroupEngine,
    ProjectStore,
)
from app.services.report_generator import generate_html_report, save_report
from app.i18n import Language, get_label


# Store last simulation result for report generation
_last_simulation_result = None
_last_product_description = None
_last_simulation_inputs = None

_last_ab_test_result = None
_last_ab_test_inputs = None

_last_price_analysis_result = None
_last_price_analysis_inputs = None

_last_focus_group_inputs = None


# === Helper Functions ===


def get_lang(lang_code: str) -> Language:
    """Convert language code string to Language enum."""
    return Language.EN if lang_code == "English" else Language.PL


def create_histogram_data(distribution: dict, lang: Language) -> pd.DataFrame:
    """Convert distribution dict to DataFrame for plotting."""
    if lang == Language.PL:
        labels = ["1-Nie", "2-Raczej nie", "3-Neutralny", "4-Raczej tak", "5-Tak"]
        col_x, col_y = "Odpowiedź", "Procent"
    else:
        labels = ["1-No", "2-Probably not", "3-Neutral", "4-Probably yes", "5-Yes"]
        col_x, col_y = "Response", "Percent"
    
    values = [
        distribution.get("scale_1", 0) * 100,
        distribution.get("scale_2", 0) * 100,
        distribution.get("scale_3", 0) * 100,
        distribution.get("scale_4", 0) * 100,
        distribution.get("scale_5", 0) * 100,
    ]
    return pd.DataFrame({col_x: labels, col_y: values})


def format_opinion(agent_response, lang: Language) -> str:
    """Format a single agent response for display."""
    persona = agent_response.persona
    if lang == Language.PL:
        return (
            f"**{persona.name}** ({persona.age} lat, {persona.gender}, {persona.location})\n"
            f"*Dochód: ~{persona.income} PLN/mies.*\n\n"
            f"> {agent_response.text_response}\n\n"
            f"📊 Ocena: **{agent_response.likert_score:.2f}/5**\n"
            f"---"
        )
    else:
        return (
            f"**{persona.name}** ({persona.age} y.o., {persona.gender}, {persona.location})\n"
            f"*Income: ~${persona.income}/month*\n\n"
            f"> {agent_response.text_response}\n\n"
            f"📊 Score: **{agent_response.likert_score:.2f}/5**\n"
            f"---"
        )


def build_simulation_summary(result: SimulationResult, lang: Language) -> str:
    """Build summary markdown for a simulation result."""
    dist = result.aggregate_distribution.model_dump()
    if lang == Language.PL:
        return (
            f"## 📊 Wyniki Symulacji\n\n"
            f"**Średnia intencja zakupu:** {result.mean_purchase_intent:.2f}/5\n\n"
            f"**Liczba agentów:** {result.n_agents}\n\n"
            f"### Rozkład odpowiedzi:\n"
            f"- Zdecydowanie NIE: {dist['scale_1']*100:.1f}%\n"
            f"- Raczej nie: {dist['scale_2']*100:.1f}%\n"
            f"- Neutralny: {dist['scale_3']*100:.1f}%\n"
            f"- Raczej tak: {dist['scale_4']*100:.1f}%\n"
            f"- Zdecydowanie TAK: {dist['scale_5']*100:.1f}%\n"
        )
    return (
        f"## 📊 Simulation Results\n\n"
        f"**Mean purchase intent:** {result.mean_purchase_intent:.2f}/5\n\n"
        f"**Number of agents:** {result.n_agents}\n\n"
        f"### Response distribution:\n"
        f"- Definitely NO: {dist['scale_1']*100:.1f}%\n"
        f"- Probably not: {dist['scale_2']*100:.1f}%\n"
        f"- Neutral: {dist['scale_3']*100:.1f}%\n"
        f"- Probably yes: {dist['scale_4']*100:.1f}%\n"
        f"- Definitely YES: {dist['scale_5']*100:.1f}%\n"
    )


def build_simulation_opinions(result: SimulationResult, lang: Language) -> str:
    """Build opinions markdown from a simulation result."""
    return "\n\n".join(format_opinion(r, lang) for r in result.agent_responses[:5])


def build_ab_test_summary(result: dict, lang: Language) -> str:
    """Build markdown summary for an A/B test result."""
    va = result["variant_a"]
    vb = result["variant_b"]
    comp = result["comparison"]
    if lang == Language.PL:
        return (
            f"## 🔬 Wyniki Testu A/B\n\n"
            f"### Wariant A\n"
            f"- Intencja zakupu: **{va['mean_purchase_intent']:.2f}/5**\n\n"
            f"### Wariant B\n"
            f"- Intencja zakupu: **{vb['mean_purchase_intent']:.2f}/5**\n\n"
            f"### Porównanie\n"
            f"- **Zwycięzca:** Wariant {comp['winner']}\n"
            f"- **Lift:** {comp['lift_percent']:+.1f}%\n"
            f"- Agentów na wariant: {comp['n_agents_per_variant']}\n"
        )
    return (
        f"## 🔬 A/B Test Results\n\n"
        f"### Variant A\n"
        f"- Purchase intent: **{va['mean_purchase_intent']:.2f}/5**\n\n"
        f"### Variant B\n"
        f"- Purchase intent: **{vb['mean_purchase_intent']:.2f}/5**\n\n"
        f"### Comparison\n"
        f"- **Winner:** Variant {comp['winner']}\n"
        f"- **Lift:** {comp['lift_percent']:+.1f}%\n"
        f"- Agents per variant: {comp['n_agents_per_variant']}\n"
    )


def build_price_analysis_outputs(result: dict, lang: Language) -> tuple[str, pd.DataFrame]:
    """Build markdown and chart data for a price analysis result."""
    demand_raw = result["demand_curve"]
    demand: dict[float, dict] = {}
    for key, value in demand_raw.items():
        try:
            price = float(key)
        except (TypeError, ValueError):
            continue
        demand[price] = value
    currency = "$" if lang == Language.EN else "PLN"

    if lang == Language.PL:
        curve_text = "### Krzywa popytu\n\n| Cena (PLN) | Intencja zakupu |\n|------------|----------------|\n"
    else:
        curve_text = "### Demand Curve\n\n| Price ($) | Purchase Intent |\n|-----------|----------------|\n"

    for price in sorted(demand.keys()):
        pi = demand[price]["mean_purchase_intent"]
        curve_text += f"| {price:.2f} | {pi:.2f}/5 |\n"

    optimal_price = result.get("optimal_price")
    try:
        optimal_price = float(optimal_price)
    except (TypeError, ValueError):
        optimal_price = None

    if optimal_price is not None:
        if lang == Language.PL:
            curve_text += f"\n\n**Optymalna cena:** {optimal_price:.2f} {currency}"
        else:
            curve_text += f"\n\n**Optimal price:** {currency}{optimal_price:.2f}"

    if result.get("elasticities"):
        if lang == Language.PL:
            curve_text += "\n\n### Elastyczność cenowa\n"
        else:
            curve_text += "\n\n### Price Elasticity\n"
        for e in result["elasticities"]:
            curve_text += f"- {e['price_range']}: {e['elasticity']:.2f}\n"

    if lang == Language.PL:
        chart_df = pd.DataFrame(
            {
                "Cena": sorted(demand.keys()),
                "Intencja": [demand[p]["mean_purchase_intent"] for p in sorted(demand.keys())],
            }
        )
    else:
        chart_df = pd.DataFrame(
            {
                "Price": sorted(demand.keys()),
                "Intent": [demand[p]["mean_purchase_intent"] for p in sorted(demand.keys())],
            }
        )

    return curve_text, chart_df


def build_project_info(project: dict, lang: Language) -> str:
    """Build markdown with core project metadata."""
    name = project.get("name", "Untitled")
    product = project.get("product_description", "")
    created_at = project.get("created_at", "-")
    updated_at = project.get("updated_at", "-")
    audience = project.get("target_audience") or {}

    gender = audience.get("gender")
    income_level = audience.get("income_level")
    location_type = audience.get("location_type")

    if lang == Language.PL:
        gender_label = "Wszystkie" if not gender else gender
        income_map = {"low": "Niski", "medium": "Średni", "high": "Wysoki"}
        location_map = {"urban": "Miasto", "suburban": "Przedmieścia", "rural": "Wieś"}
        income_label = income_map.get(income_level, "Wszystkie")
        location_label = location_map.get(location_type, "Wszystkie")
        age_label = f"{audience.get('age_min', '-')}-{audience.get('age_max', '-')}"
        research = project.get("research") or {}
        research_items = []
        if "simulation" in research:
            research_items.append("Symulacja SSR")
        if "ab_test" in research:
            research_items.append("Test A/B")
        if "price_analysis" in research:
            research_items.append("Analiza cenowa")
        if "focus_group" in research:
            research_items.append("Grupa fokusowa")
        research_label = ", ".join(research_items) if research_items else "Brak"
        return (
            f"## 📁 {name}\n\n"
            f"**Produkt:** {product}\n\n"
            f"**Utworzono:** {created_at}\n\n"
            f"**Zaktualizowano:** {updated_at}\n\n"
            f"**Grupa docelowa:** wiek {age_label}, płeć {gender_label}, "
            f"dochód {income_label}, lokalizacja {location_label}\n\n"
            f"**Zapisane badania:** {research_label}\n"
        )

    gender_label = "All" if not gender else gender
    income_map = {"low": "Low", "medium": "Medium", "high": "High"}
    location_map = {"urban": "Urban", "suburban": "Suburban", "rural": "Rural"}
    income_label = income_map.get(income_level, "All")
    location_label = location_map.get(location_type, "All")
    age_label = f"{audience.get('age_min', '-')}-{audience.get('age_max', '-')}"
    research = project.get("research") or {}
    research_items = []
    if "simulation" in research:
        research_items.append("SSR Simulation")
    if "ab_test" in research:
        research_items.append("A/B Test")
    if "price_analysis" in research:
        research_items.append("Price Analysis")
    if "focus_group" in research:
        research_items.append("Focus Group")
    research_label = ", ".join(research_items) if research_items else "None"
    return (
        f"## 📁 {name}\n\n"
        f"**Product:** {product}\n\n"
        f"**Created:** {created_at}\n\n"
        f"**Updated:** {updated_at}\n\n"
        f"**Target audience:** age {age_label}, gender {gender_label}, "
        f"income {income_label}, location {location_label}\n\n"
        f"**Saved research:** {research_label}\n"
    )


def get_project_store() -> ProjectStore:
    """Return project store pinned to repo root."""
    return ProjectStore(base_dir=PROJECT_BASE_DIR)


def mark_dirty(
    product_description: str,
    age_min: int,
    age_max: int,
    gender: str,
    income_level: str,
    location_type: str,
    n_agents: int,
    enable_web_search: bool,
    temperature: float,
    variant_a_input: str,
    variant_b_input: str,
    ab_n_agents: int,
    price_product: str,
    price_min: float,
    price_max: float,
    price_points: int,
    price_n_agents: int,
    fg_product: str,
    fg_participants: int,
    fg_rounds: int,
    suppress_dirty: bool,
) -> bool:
    """Set project dirty flag only when inputs differ from defaults."""
    if suppress_dirty:
        return False
    defaults = {
        "product_description": "",
        "age_min": 25,
        "age_max": 45,
        "gender": "Wszystkie",
        "income_level": "Wszystkie",
        "location_type": "Wszystkie",
        "n_agents": 20,
        "enable_web_search": False,
        "temperature": 0.01,
        "variant_a_input": "",
        "variant_b_input": "",
        "ab_n_agents": 30,
        "price_product": "",
        "price_min": 19.99,
        "price_max": 59.99,
        "price_points": 5,
        "price_n_agents": 20,
        "fg_product": "",
        "fg_participants": 6,
        "fg_rounds": 3,
    }
    current = {
        "product_description": (product_description or "").strip(),
        "age_min": int(age_min),
        "age_max": int(age_max),
        "gender": gender,
        "income_level": income_level,
        "location_type": location_type,
        "n_agents": int(n_agents),
        "enable_web_search": bool(enable_web_search),
        "temperature": float(temperature),
        "variant_a_input": (variant_a_input or "").strip(),
        "variant_b_input": (variant_b_input or "").strip(),
        "ab_n_agents": int(ab_n_agents),
        "price_product": (price_product or "").strip(),
        "price_min": float(price_min),
        "price_max": float(price_max),
        "price_points": int(price_points),
        "price_n_agents": int(price_n_agents),
        "fg_product": (fg_product or "").strip(),
        "fg_participants": int(fg_participants),
        "fg_rounds": int(fg_rounds),
    }
    return any(current[key] != defaults[key] for key in defaults)


def clear_suppress_dirty() -> bool:
    """Clear dirty suppression after programmatic updates."""
    return False


def normalize_target_audience(
    lang: Language,
    age_min: int,
    age_max: int,
    gender: str,
    income_level: str,
    location_type: str,
) -> dict:
    """Normalize UI inputs to stored demographic fields."""
    all_values = {"All", "Wszystkie"}
    income_map = {
        "Low": "low",
        "Medium": "medium",
        "High": "high",
        "Niski": "low",
        "Średni": "medium",
        "Wysoki": "high",
    }
    location_map = {
        "Urban": "urban",
        "Suburban": "suburban",
        "Rural": "rural",
        "Miasto": "urban",
        "Przedmieścia": "suburban",
        "Wieś": "rural",
    }

    income = income_map.get(income_level) if income_level not in all_values else None
    location = location_map.get(location_type) if location_type not in all_values else None

    if gender in ["M", "F"]:
        gender_value = gender
    elif gender in all_values:
        gender_value = None
    else:
        gender_value = None

    return {
        "age_min": age_min,
        "age_max": age_max,
        "gender": gender_value,
        "income_level": income,
        "location_type": location,
    }


def autosave_simulation_project(
    project_id: str | None,
    result: SimulationResult,
    inputs: dict,
    lang: Language,
) -> str:
    """Autosave simulation results into an existing project."""
    if not project_id:
        return ""
    store = get_project_store()
    try:
        project = store.load_project(project_id)
        research = project.get("research", {})
        research["simulation"] = {
            "inputs": inputs,
            "result": result.model_dump(mode="json"),
        }
        project["research"] = research

        product_description = inputs.get("product_description") or project.get("product_description", "")
        project["product_description"] = product_description

        target_audience = inputs.get("target_audience") or project.get("target_audience")
        if target_audience:
            project["target_audience"] = target_audience

        saved = store.save_project(project)
    except FileNotFoundError:
        return (
            "⚠️ Autozapis nieudany (brak projektu)"
            if lang == Language.PL
            else "⚠️ Autosave failed (project missing)"
        )
    except Exception:
        return (
            "⚠️ Autozapis nieudany"
            if lang == Language.PL
            else "⚠️ Autosave failed"
        )

    return (
        f"💾 Autozapis: {saved['name']}"
        if lang == Language.PL
        else f"💾 Autosaved: {saved['name']}"
    )


def autosave_focus_group_project(
    project_id: str | None,
    product: str,
    transcript: str,
    summary: str,
    inputs: dict,
    lang: Language,
) -> str:
    """Autosave focus group results into an existing project."""
    if not project_id:
        return ""
    store = get_project_store()
    try:
        project = store.load_project(project_id)
        research = project.get("research", {})
        research["focus_group"] = {
            "inputs": inputs,
            "result": {
                "product": product,
                "transcript": transcript,
                "summary": summary,
            },
        }
        project["research"] = research

        if product:
            project["product_description"] = product

        saved = store.save_project(project)
    except FileNotFoundError:
        return (
            "⚠️ Autozapis nieudany (brak projektu)"
            if lang == Language.PL
            else "⚠️ Autosave failed (project missing)"
        )
    except Exception:
        return (
            "⚠️ Autozapis nieudany"
            if lang == Language.PL
            else "⚠️ Autosave failed"
        )

    return (
        f"💾 Autozapis: {saved['name']}"
        if lang == Language.PL
        else f"💾 Autosaved: {saved['name']}"
    )


def target_audience_to_ui(
    lang: Language,
    target_audience: dict | None,
) -> tuple[int, int, str, str, str]:
    """Map stored target audience fields to UI labels."""
    default_age_min = 25
    default_age_max = 45
    # UI choices are fixed in Polish in this layout.
    default_gender = "Wszystkie"
    default_income = "Wszystkie"
    default_location = "Wszystkie"
    income_map = {"low": "Niski", "medium": "Średni", "high": "Wysoki"}
    location_map = {"urban": "Miasto", "suburban": "Przedmieścia", "rural": "Wieś"}

    if not target_audience:
        return (
            default_age_min,
            default_age_max,
            default_gender,
            default_income,
            default_location,
        )

    age_min = int(target_audience.get("age_min", default_age_min))
    age_max = int(target_audience.get("age_max", default_age_max))

    gender_raw = target_audience.get("gender")
    if gender_raw in ["M", "F"]:
        gender = gender_raw
    else:
        gender = default_gender

    income_level = income_map.get(target_audience.get("income_level"), default_income)
    location_type = location_map.get(target_audience.get("location_type"), default_location)

    return age_min, age_max, gender, income_level, location_type


# === URL Detection and Product Extraction ===


import re

def is_url(text: str) -> bool:
    """Check if text is a URL."""
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain
        r'localhost|'  # localhost
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # IP
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return bool(url_pattern.match(text.strip()))


async def process_product_input(
    product_input: str,
    language: Language,
) -> tuple[str, str]:
    """Process product input - extract from URL if needed.
    
    Returns:
        (product_description, status_message)
    """
    product_input = product_input.strip()
    
    def _looks_like_full_extraction(text: str) -> bool:
        text = text.strip()
        if not text:
            return False
        if len(text) < 60:
            return False
        if len(text.split()) < 8:
            return False
        if not any(ch.isdigit() for ch in text):
            return False
        lowered = text.lower()
        currency_markers = ["pln", "zł", "zl", "usd", "eur", "$", "€", "cena", "price"]
        if not any(marker in lowered for marker in currency_markers):
            return False
        if "." not in text and "," not in text:
            return False
        return True

    def _attach_source(text: str, url: str) -> str:
        label = "Źródło" if language == Language.PL else "Source"
        text = text.strip()
        if not text:
            return f"{label}: {url}"
        if f"{label}: {url}" in text:
            return text
        return f"{text} {label}: {url}"

    def _fallback_from_url(url: str) -> str:
        from urllib.parse import urlparse, unquote

        parsed = urlparse(url)
        slug = parsed.path.rstrip("/").split("/")[-1]
        slug = unquote(slug)
        slug = slug.replace("-", " ").replace("_", " ").strip()
        if not slug:
            slug = "produkt z podanego URL" if language == Language.PL else "product from provided URL"
        if language == Language.PL:
            return f"{slug}. Opis na podstawie URL (brak pełnej ekstrakcji)."
        return f"{slug}. Description based on URL (full extraction unavailable)."

    if is_url(product_input):
        # Extract product from URL using Gemini
        from app.services.llm_client import get_llm_client
        
        try:
            client = get_llm_client()
            if hasattr(client, 'extract_product_from_url'):
                status = "🔗 Extracting product from URL..." if language == Language.EN else "🔗 Pobieranie produktu z URL..."
                product_description = await client.extract_product_from_url(product_input, language)
                if _looks_like_full_extraction(product_description):
                    product_description = _attach_source(product_description, product_input)
                    return product_description, f"✅ Extracted from: {product_input}"
                fallback = _fallback_from_url(product_input)
                fallback = _attach_source(fallback, product_input)
                warn = (
                    "⚠️ Extraction too short; using URL fallback."
                    if language == Language.EN
                    else "⚠️ Ekstrakcja zbyt krótka — używam opisu na podstawie URL."
                )
                return fallback, warn
            fallback = _attach_source(_fallback_from_url(product_input), product_input)
            warn = (
                "⚠️ URL extraction unavailable; using URL fallback."
                if language == Language.EN
                else "⚠️ Ekstrakcja z URL niedostępna — używam opisu na podstawie URL."
            )
            return fallback, warn
        except Exception as e:
            # Fallback to using URL as-is
            fallback = _fallback_from_url(product_input)
            fallback = _attach_source(fallback, product_input)
            return fallback, f"⚠️ Could not fetch URL: {e}"
    
    return product_input, ""


# === Main Simulation Tab ===


async def run_simulation_async(
    lang_code: str,
    product_description: str,
    n_agents: int,
    age_min: int,
    age_max: int,
    gender: Optional[str],
    income_level: Optional[str],
    location_type: Optional[str],
    enable_web_search: bool = False,
    temperature: float = 0.01,
    project_id: str | None = None,
    progress=gr.Progress(),
):
    """Run SSR simulation and return results."""
    lang = get_lang(lang_code)
    
    if not product_description.strip():
        err = get_label(lang, "error_no_product")
        return None, err, "", err, gr.update(), ""

    progress(0, desc="Initializing..." if lang == Language.EN else "Inicjalizacja...")
    url_status_msg = ""
    extracted_preview = ""

    try:
        # Process product input - extract from URL if needed
        if is_url(product_description.strip()):
            progress(0.05, desc="🔗 Fetching product from URL..." if lang == Language.EN else "🔗 Pobieranie produktu z URL...")
            product_description, url_status = await process_product_input(product_description, lang)
            url_status_msg = url_status or ""
            extracted_preview = product_description
            if not product_description:
                return None, "Could not extract product from URL", "", "❌", gr.update(), ""
        # Handle language-specific "All" values
        all_value = "All" if lang == Language.EN else "Wszystkie"
        
        # Map income level
        income_map = {"Low": "low", "Medium": "medium", "High": "high", 
                      "Niski": "low", "Średni": "medium", "Wysoki": "high"}
        income = income_map.get(income_level) if income_level != all_value else None
        
        # Map location type
        loc_map = {"Urban": "urban", "Suburban": "suburban", "Rural": "rural",
                   "Miasto": "urban", "Przedmieścia": "suburban", "Wieś": "rural"}
        location = loc_map.get(location_type) if location_type != all_value else None
        
        # Build demographic profile
        profile = DemographicProfile(
            age_min=age_min,
            age_max=age_max,
            gender=gender if gender not in [all_value, "M", "F"] else (gender if gender in ["M", "F"] else None),
            income_level=income,
            location_type=location,
        )
        # Fix gender handling
        if gender in ["M", "F"]:
            profile.gender = gender
        elif gender != all_value:
            profile.gender = None
        else:
            profile.gender = None

        search_msg = " + Google Search" if enable_web_search else ""
        progress(0.1, desc=f"Generating personas{search_msg}..." if lang == Language.EN else f"Generowanie person{search_msg}...")

        # Run simulation with language parameter
        engine = SimulationEngine(language=lang, temperature=temperature)
        from uuid import uuid4
        result = await engine.run_simulation(
            project_id=uuid4(),
            product_description=product_description,
            target_audience=profile,
            n_agents=n_agents,
            enable_web_search=enable_web_search,
        )

        progress(0.9, desc="Processing results..." if lang == Language.EN else "Przetwarzanie wyników...")

        # Prepare histogram
        dist = result.aggregate_distribution.model_dump()
        chart_df = create_histogram_data(dist, lang)

        summary = build_simulation_summary(result, lang)
        opinions = build_simulation_opinions(result, lang)

        progress(1.0, desc="Done!" if lang == Language.EN else "Gotowe!")

        # Store result for report generation and project saving
        global _last_simulation_result, _last_product_description, _last_simulation_inputs
        _last_simulation_result = result
        _last_product_description = product_description
        sim_inputs = {
            "product_description": product_description,
            "n_agents": n_agents,
            "target_audience": profile.model_dump(),
            "enable_web_search": enable_web_search,
            "temperature": temperature,
        }
        _last_simulation_inputs = sim_inputs

        autosave_msg = autosave_simulation_project(project_id, result, sim_inputs, lang)
        status_msg = get_label(lang, "success")
        dirty_value = True
        if autosave_msg:
            status_msg = f"{status_msg} • {autosave_msg}"
            dirty_value = False
        if url_status_msg:
            status_msg = f"{status_msg} • {url_status_msg}"

        if extracted_preview:
            if lang == Language.PL:
                extracted_preview = f"**Wyciągnięte dane:** {extracted_preview}"
            else:
                extracted_preview = f"**Extracted data:** {extracted_preview}"

        return chart_df, summary, opinions, status_msg, dirty_value, extracted_preview

    except Exception as e:
        import traceback
        traceback.print_exc()
        return (
            None,
            f"❌ Error: {str(e)}" if lang == Language.EN else f"❌ Błąd: {str(e)}",
            "",
            f"❌ {str(e)}",
            gr.update(),
            "",
        )


def run_simulation(*args):
    """Wrapper to run async function."""
    return asyncio.run(run_simulation_async(*args))


def generate_report(lang_code: str):
    """Generate HTML report preview from last simulation results."""
    global _last_simulation_result, _last_product_description
    lang = get_lang(lang_code)
    
    if _last_simulation_result is None:
        if lang == Language.EN:
            return "", "❌ No simulation results. Run a simulation first."
        else:
            return "", "❌ Brak wyników symulacji. Najpierw uruchom symulację."
    
    try:
        # Generate HTML report content
        html_content = generate_html_report(
            result=_last_simulation_result,
            product_description=_last_product_description,
            lang=lang,
        )
        
        # Store for export (raw HTML)
        global _last_report_html
        _last_report_html = html_content
        
        # Wrap in iframe to isolate styles from Gradio
        # Escape quotes for srcdoc attribute
        escaped_html = html_content.replace('"', '&quot;')
        iframe_html = f'''<iframe 
            srcdoc="{escaped_html}" 
            style="width: 100%; height: 800px; border: 1px solid #ccc; border-radius: 8px;"
            sandbox="allow-same-origin">
        </iframe>'''
        
        if lang == Language.EN:
            return iframe_html, "✅ Report ready! Choose export format below."
        else:
            return iframe_html, "✅ Raport gotowy! Wybierz format eksportu poniżej."
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return "", f"❌ Error: {str(e)}" if lang == Language.EN else f"❌ Błąd: {str(e)}"


# Store last report HTML for export
_last_report_html = None


def export_report(lang_code: str, export_format: str):
    """Export report to HTML or PDF file."""
    global _last_report_html
    lang = get_lang(lang_code)
    
    if _last_report_html is None:
        if lang == Language.EN:
            return None, "❌ Generate report first."
        else:
            return None, "❌ Najpierw wygeneruj raport."
    
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = PathlibPath(tempfile.gettempdir()) / "market_wizard_reports"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if export_format == "PDF":
            # Export as PDF using weasyprint if available, otherwise use pdfkit
            try:
                from weasyprint import HTML
                output_path = output_dir / f"ssr_report_{timestamp}.pdf"
                HTML(string=_last_report_html).write_pdf(str(output_path))
            except ImportError:
                # Fallback: save as HTML with PDF instruction
                output_path = output_dir / f"ssr_report_{timestamp}.html"
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(_last_report_html)
                if lang == Language.EN:
                    return str(output_path), "⚠️ PDF export requires weasyprint. Saved as HTML - use browser Print to PDF."
                else:
                    return str(output_path), "⚠️ Eksport PDF wymaga weasyprint. Zapisano HTML - użyj Drukuj do PDF w przeglądarce."
        else:
            # Export as HTML
            output_path = output_dir / f"ssr_report_{timestamp}.html"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(_last_report_html)
        
        if lang == Language.EN:
            return str(output_path), f"✅ Exported: {output_path.name}"
        else:
            return str(output_path), f"✅ Wyeksportowano: {output_path.name}"
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"❌ Error: {str(e)}" if lang == Language.EN else f"❌ Błąd: {str(e)}"


# === A/B Test Tab ===


async def run_ab_test_async(
    lang_code: str,
    variant_a: str,
    variant_b: str,
    n_agents: int,
    progress=gr.Progress(),
):
    """Run A/B test comparison."""
    lang = get_lang(lang_code)
    
    if not variant_a.strip() or not variant_b.strip():
        return get_label(lang, "error_no_variants"), "❌"

    progress(0, desc="Running A/B test..." if lang == Language.EN else "Uruchamianie testu A/B...")

    try:
        engine = ABTestEngine(language=lang)
        from uuid import uuid4
        result = await engine.run_ab_test(
            project_id=uuid4(),
            variant_a=variant_a,
            variant_b=variant_b,
            n_agents=n_agents,
        )

        progress(1.0)

        global _last_ab_test_result, _last_ab_test_inputs
        _last_ab_test_result = result
        _last_ab_test_inputs = {
            "variant_a": variant_a,
            "variant_b": variant_b,
            "n_agents": n_agents,
        }

        return build_ab_test_summary(result, lang), "✅", True

    except Exception as e:
        import traceback
        traceback.print_exc()
        return (
            f"❌ Error: {str(e)}" if lang == Language.EN else f"❌ Błąd: {str(e)}",
            "❌",
            gr.update(),
        )


def run_ab_test(*args):
    """Wrapper for async A/B test."""
    return asyncio.run(run_ab_test_async(*args))


# === Price Sensitivity Tab ===


async def run_price_analysis_async(
    lang_code: str,
    product_description: str,
    price_min: float,
    price_max: float,
    n_points: int,
    n_agents: int,
    progress=gr.Progress(),
):
    """Run price sensitivity analysis."""
    lang = get_lang(lang_code)
    
    if not product_description.strip():
        return get_label(lang, "error_no_product"), None, "❌"

    progress(0, desc="Analyzing price sensitivity..." if lang == Language.EN else "Analizowanie wrażliwości cenowej...")

    try:
        # Generate price points
        price_points = list(np.linspace(price_min, price_max, int(n_points)))

        engine = PriceSensitivityEngine(language=lang)
        from uuid import uuid4
        result = await engine.analyze_price_sensitivity(
            project_id=uuid4(),
            base_product_description=product_description,
            price_points=price_points,
            n_agents=n_agents,
        )

        progress(1.0)

        global _last_price_analysis_result, _last_price_analysis_inputs
        _last_price_analysis_result = result
        _last_price_analysis_inputs = {
            "base_product_description": product_description,
            "price_min": price_min,
            "price_max": price_max,
            "price_points": n_points,
            "n_agents": n_agents,
        }

        curve_text, chart_df = build_price_analysis_outputs(result, lang)
        return curve_text, chart_df, "✅", True

    except Exception as e:
        import traceback
        traceback.print_exc()
        return (
            f"❌ Error: {str(e)}" if lang == Language.EN else f"❌ Błąd: {str(e)}",
            None,
            "❌",
            gr.update(),
        )


def run_price_analysis(*args):
    """Wrapper for async price analysis."""
    return asyncio.run(run_price_analysis_async(*args))


# === Focus Group Functions ===


async def run_focus_group_async(
    lang_code: str,
    product: str,
    n_participants: int,
    n_rounds: int,
    project_id: str | None = None,
) -> tuple[str, str, str, object]:
    """Run virtual focus group discussion."""
    lang = get_lang(lang_code)
    
    if not product or not product.strip():
        error_msg = "❌ Enter product description" if lang == Language.EN else "❌ Wprowadź opis produktu"
        return "", "", error_msg
    
    try:
        # Process product input - extract from URL if needed
        if is_url(product.strip()):
            product, _ = await process_product_input(product, lang)
            if not product:
                return "", "", "❌ Could not extract product from URL"
        
        engine = FocusGroupEngine(language=lang)
        
        # Run focus group
        result = await engine.run_focus_group(
            product_description=product.strip(),
            n_participants=int(n_participants),
            n_rounds=int(n_rounds),
        )
        
        # Format discussion transcript
        transcript = ""
        current_round = 0
        for msg in result.discussion:
            if msg.round != current_round:
                current_round = msg.round
                if lang == Language.PL:
                    transcript += f"\n### 📢 Runda {current_round}\n\n"
                else:
                    transcript += f"\n### 📢 Round {current_round}\n\n"
            
            transcript += f"**{msg.persona_name}** ({msg.persona_demographics}):\n"
            transcript += f"> {msg.content}\n\n"
        
        # Format summary
        summary = f"## 📋 {'Moderator Summary' if lang == Language.EN else 'Podsumowanie moderatora'}\n\n"
        summary += result.summary
        
        if lang == Language.PL:
            status = f"✅ Zakończono! {len(result.participants)} uczestników, {n_rounds} rund"
        else:
            status = f"✅ Complete! {len(result.participants)} participants, {n_rounds} rounds"

        autosave_msg = autosave_focus_group_project(
            project_id,
            product.strip(),
            transcript,
            summary,
            {
                "product_description": product.strip(),
                "n_participants": int(n_participants),
                "n_rounds": int(n_rounds),
            },
            lang,
        )
        dirty_value = True
        if autosave_msg:
            status = f"{status} • {autosave_msg}"
            dirty_value = False
        
        return transcript, summary, status, dirty_value
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        error_msg = f"❌ Error: {str(e)}" if lang == Language.EN else f"❌ Błąd: {str(e)}"
        return "", "", error_msg, gr.update()


# Store last focus group result for export  
_last_fg_transcript = None
_last_fg_summary = None
_last_fg_product = None


def run_focus_group(*args):
    """Wrapper for async focus group."""
    global _last_fg_transcript, _last_fg_summary, _last_fg_product, _last_focus_group_inputs
    result = asyncio.run(run_focus_group_async(*args))
    _last_fg_transcript = result[0]
    _last_fg_summary = result[1]
    _last_fg_product = args[1] if len(args) > 1 else ""
    _last_focus_group_inputs = {
        "product_description": _last_fg_product,
        "n_participants": args[2] if len(args) > 2 else None,
        "n_rounds": args[3] if len(args) > 3 else None,
    }
    return result


def export_focus_group(lang_code: str, export_format: str) -> tuple[str | None, str]:
    """Export focus group results to HTML or PDF file."""
    global _last_fg_transcript, _last_fg_summary, _last_fg_product
    lang = get_lang(lang_code)
    
    if not _last_fg_transcript:
        msg = "❌ Run Focus Group first" if lang == Language.EN else "❌ Najpierw uruchom Focus Group"
        return None, msg
    
    # Generate HTML
    html_content = f"""<!DOCTYPE html>
<html lang="{'en' if lang == Language.EN else 'pl'}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Focus Group Report - Market Wizard</title>
    <style>
        body {{ font-family: 'Segoe UI', system-ui, sans-serif; max-width: 900px; margin: 0 auto; padding: 2rem; background: #f8fafc; color: #1f2937; }}
        .header {{ background: linear-gradient(135deg, #1e3a5f, #2d5a87); color: white; padding: 2rem; border-radius: 1rem; margin-bottom: 2rem; }}
        .header h1 {{ margin: 0; }}
        .section {{ background: white; border-radius: 1rem; padding: 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .section h2 {{ color: #1e3a5f; border-bottom: 2px solid #1e3a5f; padding-bottom: 0.5rem; }}
        blockquote {{ background: #f1f5f9; padding: 1rem; border-left: 3px solid #1e3a5f; margin: 0.5rem 0; border-radius: 0.25rem; }}
        .footer {{ text-align: center; color: #6b7280; font-size: 0.875rem; margin-top: 2rem; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 {'Focus Group Report' if lang == Language.EN else 'Raport Grupy Fokusowej'}</h1>
        <p>{'Product' if lang == Language.EN else 'Produkt'}: {_last_fg_product}</p>
    </div>
    
    <div class="section">
        <h2>💬 {'Discussion Transcript' if lang == Language.EN else 'Transkrypcja dyskusji'}</h2>
        {_last_fg_transcript.replace('###', '<h3>').replace('**', '<strong>').replace('> ', '<blockquote>').replace(chr(10)+chr(10), '</blockquote><br>')}
    </div>
    
    <div class="section">
        <h2>📋 {'Moderator Summary' if lang == Language.EN else 'Podsumowanie moderatora'}</h2>
        <div>{_last_fg_summary.replace(chr(10), '<br>')}</div>
    </div>
    
    <div class="footer">
        <p>{'Generated by Market Wizard' if lang == Language.EN else 'Wygenerowano przez Market Wizard'}</p>
    </div>
</body>
</html>"""
    
    # Save to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if export_format == "PDF":
        try:
            from weasyprint import HTML
            filename = f"focus_group_{timestamp}.pdf"
            filepath = Path(tempfile.gettempdir()) / filename
            HTML(string=html_content).write_pdf(str(filepath))
        except ImportError:
            return None, "❌ PDF export requires weasyprint. Install with: pip install weasyprint"
    else:
        filename = f"focus_group_{timestamp}.html"
        filepath = Path(tempfile.gettempdir()) / filename
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)
    
    msg = f"✅ Exported to {filename}" if lang == Language.EN else f"✅ Wyeksportowano do {filename}"
    return str(filepath), msg


# === Project Management ===


def _empty_histogram_df(lang: Language) -> pd.DataFrame:
    return create_histogram_data(
        {
            "scale_1": 0,
            "scale_2": 0,
            "scale_3": 0,
            "scale_4": 0,
            "scale_5": 0,
        },
        lang,
    )


def _empty_price_df(lang: Language) -> pd.DataFrame:
    if lang == Language.PL:
        return pd.DataFrame({"Cena": [], "Intencja": []})
    return pd.DataFrame({"Price": [], "Intent": []})


def _project_dropdown_choices(
    projects: list[dict], selected_id: str | None = None
) -> tuple[list[tuple[str, str]], str | None]:
    choices: list[tuple[str, str]] = []
    for project in projects:
        name = project.get("name", "Untitled")
        timestamp = project.get("updated_at") or project.get("created_at") or ""
        label = f"{name} • {timestamp}"
        project_id = project.get("id")
        if project_id:
            choices.append((label, project_id))
    if selected_id and any(pid == selected_id for _, pid in choices):
        return choices, selected_id
    return choices, None


def list_saved_projects(lang_code: str):
    store = get_project_store()
    projects = store.list_projects()
    choices, selected = _project_dropdown_choices(projects)
    return gr.update(choices=choices, value=selected), selected, gr.update()


def save_project(
    lang_code: str,
    project_name: str,
    selected_project_id: str | None,
    product_description: str,
    age_min: int,
    age_max: int,
    gender: str,
    income_level: str,
    location_type: str,
    n_agents: int,
    enable_web_search: bool,
    temperature: float,
    variant_a_input: str,
    variant_b_input: str,
    ab_n_agents: int,
    price_product: str,
    price_min: float,
    price_max: float,
    price_points: int,
    price_n_agents: int,
    fg_product: str,
    fg_participants: int,
    fg_rounds: int,
):
    lang = get_lang(lang_code)
    store = get_project_store()

    existing: dict = {}
    create_new = False
    if selected_project_id:
        try:
            existing = store.load_project(selected_project_id)
        except FileNotFoundError:
            existing = {}
        else:
            if (project_name or "").strip() and existing.get("name") and existing.get("name") != project_name.strip():
                create_new = True

    has_new_data = any(
        [
            _last_simulation_result is not None,
            _last_ab_test_result is not None,
            _last_price_analysis_result is not None,
            bool(_last_fg_transcript),
        ]
    )

    name = (project_name or "").strip() or existing.get("name")
    if not name:
        msg = (
            "❌ Podaj nazwę projektu."
            if lang == Language.PL
            else "❌ Please provide a project name."
        )
        dropdown_update, selected, dirty_update = list_saved_projects(lang_code)
        return msg, dropdown_update, selected, dirty_update

    input_product = (product_description or "").strip()
    ab_product = (variant_a_input or "").strip()
    ab_product_b = (variant_b_input or "").strip()
    price_product_input = (price_product or "").strip()
    fg_product_input = (fg_product or "").strip()

    product_description = input_product or _last_product_description or ""
    if not product_description and _last_focus_group_inputs:
        product_description = _last_focus_group_inputs.get("product_description", "")
    if not product_description and _last_price_analysis_inputs:
        product_description = _last_price_analysis_inputs.get("base_product_description", "")
    if not product_description and _last_ab_test_inputs:
        product_description = _last_ab_test_inputs.get("variant_a", "")
    if not product_description:
        product_description = fg_product_input or price_product_input or ab_product
    if not product_description:
        product_description = existing.get("product_description", "")

    target_audience = existing.get("target_audience")
    if _last_simulation_inputs:
        target_audience = _last_simulation_inputs.get("target_audience")
    elif input_product or not existing:
        target_audience = normalize_target_audience(
            lang,
            age_min,
            age_max,
            gender,
            income_level,
            location_type,
        )

    research = existing.get("research", {})

    if _last_simulation_result is not None or _last_simulation_inputs is not None:
        sim_payload = research.get("simulation", {})
        if _last_simulation_inputs is not None:
            sim_payload["inputs"] = _last_simulation_inputs
        else:
            sim_payload["inputs"] = {
                "product_description": input_product or product_description,
                "n_agents": n_agents,
                "target_audience": target_audience,
                "enable_web_search": enable_web_search,
                "temperature": temperature,
            }
        if _last_simulation_result is not None:
            sim_payload["result"] = _last_simulation_result.model_dump(mode="json")
        research["simulation"] = sim_payload
    elif "simulation" not in research:
        research["simulation"] = {
            "inputs": {
                "product_description": input_product or product_description,
                "n_agents": n_agents,
                "target_audience": target_audience,
                "enable_web_search": enable_web_search,
                "temperature": temperature,
            }
        }

    ab_inputs_present = bool(ab_product or ab_product_b)
    if ab_inputs_present or _last_ab_test_result is not None or _last_ab_test_inputs is not None:
        ab_payload = research.get("ab_test", {})
        if ab_inputs_present:
            ab_payload["inputs"] = {
                "variant_a": variant_a_input,
                "variant_b": variant_b_input,
                "n_agents": ab_n_agents,
            }
        elif _last_ab_test_inputs is not None:
            ab_payload["inputs"] = _last_ab_test_inputs
        if _last_ab_test_result is not None:
            ab_payload["result"] = _last_ab_test_result
        research["ab_test"] = ab_payload

    price_inputs_present = bool(price_product_input)
    if price_inputs_present or _last_price_analysis_result is not None or _last_price_analysis_inputs is not None:
        price_payload = research.get("price_analysis", {})
        if price_inputs_present:
            price_payload["inputs"] = {
                "base_product_description": price_product,
                "price_min": price_min,
                "price_max": price_max,
                "price_points": price_points,
                "n_agents": price_n_agents,
            }
        elif _last_price_analysis_inputs is not None:
            price_payload["inputs"] = _last_price_analysis_inputs
        if _last_price_analysis_result is not None:
            price_payload["result"] = _last_price_analysis_result
        research["price_analysis"] = price_payload

    fg_inputs_present = bool(fg_product_input)
    if fg_inputs_present or _last_fg_transcript or _last_fg_summary or _last_focus_group_inputs:
        fg_payload = research.get("focus_group", {})
        if fg_inputs_present:
            fg_payload["inputs"] = {
                "product_description": fg_product,
                "n_participants": fg_participants,
                "n_rounds": fg_rounds,
            }
        elif _last_focus_group_inputs is not None:
            fg_payload["inputs"] = _last_focus_group_inputs
        fg_payload["result"] = {
            "product": _last_fg_product or fg_product_input or existing.get("product_description", ""),
            "transcript": _last_fg_transcript or fg_payload.get("result", {}).get("transcript", ""),
            "summary": _last_fg_summary or fg_payload.get("result", {}).get("summary", ""),
        }
        research["focus_group"] = fg_payload

    project = {
        "id": None if create_new else (selected_project_id or existing.get("id")),
        "name": name,
        "product_description": product_description,
        "target_audience": target_audience,
        "research": research,
    }

    saved = store.save_project(project)
    projects = store.list_projects()
    choices, selected = _project_dropdown_choices(projects, saved["id"])

    if not has_new_data:
        msg = (
            f"✅ Zapisano projekt (bez wyników badań): {saved['name']}"
            if lang == Language.PL
            else f"✅ Saved project (no research yet): {saved['name']}"
        )
    else:
        msg = (
            f"✅ Zapisano projekt: {saved['name']}"
            if lang == Language.PL
            else f"✅ Saved project: {saved['name']}"
        )
    return msg, gr.update(choices=choices, value=selected), selected, False


def load_project(
    lang_code: str,
    project_id: str | None,
    allow_discard: bool,
    project_dirty: bool,
):
    lang = get_lang(lang_code)
    def _no_change_updates(count: int) -> list:
        return [gr.update() for _ in range(count)]
    total_outputs = 43
    status_index = 9

    def _confirm_message(message: str):
        updates = [gr.update() for _ in range(total_outputs)]
        updates[status_index] = message
        return tuple(updates)

    if not project_id:
        msg = "❌ Wybierz projekt." if lang == Language.PL else "❌ Select a project."
        return (
            "",
            "",
            _empty_histogram_df(lang),
            "",
            "",
            "",
            _empty_price_df(lang),
            "",
            "",
            msg,
            gr.update(visible=False),
            gr.update(visible=False),
            *_no_change_updates(31),
        )

    if project_dirty and not allow_discard:
        msg = (
            "⚠️ Masz niezapisane zmiany. Czy chcesz zapisać bieżący projekt? Użyj przycisków poniżej."
            if lang == Language.PL
            else "⚠️ You have unsaved changes. Do you want to save the current project? Use the buttons below."
        )
        updates = list(_confirm_message(msg))
        updates[10] = gr.update(visible=True)
        updates[11] = gr.update(visible=True)
        return tuple(updates)

    store = get_project_store()
    try:
        project = store.load_project(project_id)
    except FileNotFoundError:
        msg = "❌ Nie znaleziono projektu." if lang == Language.PL else "❌ Project not found."
        return (
            "",
            "",
            _empty_histogram_df(lang),
            "",
            "",
            "",
            _empty_price_df(lang),
            "",
            "",
            msg,
            gr.update(visible=False),
            gr.update(visible=False),
            *_no_change_updates(31),
        )

    info = build_project_info(project, lang)

    research = project.get("research", {})

    sim_summary = ""
    sim_opinions = ""
    sim_chart = _empty_histogram_df(lang)
    sim_payload = research.get("simulation", {})
    if sim_payload.get("result"):
        sim_result = SimulationResult.model_validate(sim_payload["result"])
        sim_summary = build_simulation_summary(sim_result, lang)
        sim_opinions = build_simulation_opinions(sim_result, lang)
        sim_chart = create_histogram_data(sim_result.aggregate_distribution.model_dump(), lang)
        # Restore globals for report generation
        global _last_simulation_result, _last_product_description, _last_simulation_inputs
        _last_simulation_result = sim_result
        _last_product_description = (
            sim_payload.get("inputs", {}).get("product_description")
            or project.get("product_description", "")
        )
        _last_simulation_inputs = sim_payload.get("inputs")
    else:
        sim_summary = (
            "Brak zapisanych wyników symulacji."
            if lang == Language.PL
            else "No saved simulation results."
        )

    ab_summary = ""
    ab_payload = research.get("ab_test", {})
    if ab_payload.get("result"):
        ab_summary = build_ab_test_summary(ab_payload["result"], lang)
    else:
        ab_summary = (
            "Brak zapisanych wyników testu A/B."
            if lang == Language.PL
            else "No saved A/B test results."
        )

    price_summary = ""
    price_chart = _empty_price_df(lang)
    price_payload = research.get("price_analysis", {})
    if price_payload.get("result"):
        price_summary, price_chart = build_price_analysis_outputs(price_payload["result"], lang)
    else:
        price_summary = (
            "Brak zapisanych wyników analizy cenowej."
            if lang == Language.PL
            else "No saved price analysis results."
        )

    fg_transcript = ""
    fg_summary = ""
    fg_payload = research.get("focus_group", {})
    if fg_payload.get("result"):
        fg_transcript = fg_payload["result"].get("transcript", "")
        fg_summary = fg_payload["result"].get("summary", "")
        global _last_fg_transcript, _last_fg_summary, _last_fg_product, _last_focus_group_inputs
        _last_fg_transcript = fg_transcript
        _last_fg_summary = fg_summary
        _last_fg_product = fg_payload["result"].get("product", "")
        _last_focus_group_inputs = fg_payload.get("inputs")
    else:
        fg_transcript = (
            "Brak zapisanej transkrypcji."
            if lang == Language.PL
            else "No saved transcript."
        )
        fg_summary = (
            "Brak zapisanych wniosków."
            if lang == Language.PL
            else "No saved summary."
        )

    sim_inputs = sim_payload.get("inputs", {}) if sim_payload else {}
    ab_inputs = ab_payload.get("inputs", {}) if ab_payload else {}
    price_inputs = price_payload.get("inputs", {}) if price_payload else {}
    fg_inputs = fg_payload.get("inputs", {}) if fg_payload else {}

    project_name_value = project.get("name", "")
    product_value = (
        sim_inputs.get("product_description")
        or project.get("product_description", "")
    )

    target_audience = sim_inputs.get("target_audience") or project.get("target_audience")
    age_min_value, age_max_value, gender_value, income_value, location_value = (
        target_audience_to_ui(lang, target_audience)
    )

    n_agents_value = int(sim_inputs.get("n_agents") or 20)
    enable_web_search_value = bool(sim_inputs.get("enable_web_search", False))
    temperature_value = float(sim_inputs.get("temperature") or 0.01)

    variant_a_value = ab_inputs.get("variant_a") or project.get("product_description", "")
    variant_b_value = ab_inputs.get("variant_b") or ""
    ab_n_agents_value = int(ab_inputs.get("n_agents") or 30)

    price_product_value = (
        price_inputs.get("base_product_description")
        or project.get("product_description", "")
    )
    price_min_value = float(price_inputs.get("price_min") or 19.99)
    price_max_value = float(price_inputs.get("price_max") or 59.99)
    price_points_value = int(price_inputs.get("price_points") or 5)
    price_n_agents_value = int(price_inputs.get("n_agents") or 20)

    fg_product_value = fg_inputs.get("product_description") or project.get("product_description", "")
    fg_participants_value = int(fg_inputs.get("n_participants") or 6)
    fg_rounds_value = int(fg_inputs.get("n_rounds") or 3)

    msg = "✅ Wczytano projekt." if lang == Language.PL else "✅ Project loaded."

    return (
        info,
        sim_summary,
        sim_chart,
        sim_opinions,
        ab_summary,
        price_summary,
        price_chart,
        fg_transcript,
        fg_summary,
        msg,
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(value=project_name_value),
        gr.update(value=product_value),
        gr.update(value=age_min_value),
        gr.update(value=age_max_value),
        gr.update(value=gender_value),
        gr.update(value=income_value),
        gr.update(value=location_value),
        gr.update(value=n_agents_value),
        gr.update(value=enable_web_search_value),
        gr.update(value=temperature_value),
        gr.update(value=variant_a_value),
        gr.update(value=variant_b_value),
        gr.update(value=ab_n_agents_value),
        gr.update(value=price_product_value),
        gr.update(value=price_min_value),
        gr.update(value=price_max_value),
        gr.update(value=price_points_value),
        gr.update(value=price_n_agents_value),
        gr.update(value=fg_product_value),
        gr.update(value=fg_participants_value),
        gr.update(value=fg_rounds_value),
        sim_summary,
        sim_chart,
        sim_opinions,
        ab_summary,
        price_summary,
        price_chart,
        fg_transcript,
        fg_summary,
        False,
        True,
    )


def maybe_autoload_project(
    lang_code: str,
    project_id: str | None,
    allow_discard: bool,
    auto_load: bool,
    project_dirty: bool,
):
    """Auto-load project on selection if enabled."""
    if not auto_load:
        return tuple(gr.update() for _ in range(43))
    return load_project(lang_code, project_id, allow_discard, project_dirty)


def delete_project(lang_code: str, project_id: str | None):
    lang = get_lang(lang_code)
    if not project_id:
        msg = "❌ Wybierz projekt." if lang == Language.PL else "❌ Select a project."
        dropdown_update, selected, dirty_update = list_saved_projects(lang_code)
        return msg, dropdown_update, selected, dirty_update

    store = get_project_store()
    store.delete_project(project_id)

    projects = store.list_projects()
    choices, selected = _project_dropdown_choices(projects)

    msg = "✅ Usunięto projekt." if lang == Language.PL else "✅ Project deleted."
    return msg, gr.update(choices=choices, value=selected), selected, False


# === Build Gradio Interface ===


def create_interface():
    """Create the Gradio interface with language selection."""

    with gr.Blocks(
        title="Market Wizard - Market Analyzer",
    ) as demo:
        # Language selector at the top
        with gr.Row():
            gr.Markdown("# 🔮 Market Wizard")
            language_select = gr.Dropdown(
                choices=["Polski", "English"],
                value="Polski",
                label="🌐 Language / Język",
                scale=0,
                min_width=200,
            )

        project_state = gr.State(value=None)
        project_dirty = gr.State(value=False)
        suppress_dirty = gr.State(value=False)
        allow_discard_state = gr.State(value=False)
        allow_discard_true = gr.State(value=True)
        
        gr.Markdown("*SSR-based purchase intent simulation using AI*")
        gr.Markdown("---")

        with gr.Tabs():
            # === Tab 1: Basic Simulation ===
            with gr.TabItem("📊 Simulation / Symulacja"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Product / Produkt")
                        product_input = gr.Textbox(
                            label="Product description / Opis produktu",
                            placeholder="E.g. Activated charcoal toothpaste, 75ml, price $9.99",
                            lines=5,
                        )
                        extracted_product_preview = gr.Markdown("")

                        gr.Markdown("### Target Audience / Grupa docelowa")
                        with gr.Row():
                            age_min = gr.Slider(18, 80, value=25, step=1, label="Age min / Wiek min")
                            age_max = gr.Slider(18, 80, value=45, step=1, label="Age max / Wiek max")

                        with gr.Row():
                            gender = gr.Dropdown(
                                choices=["Wszystkie", "M", "F"],
                                value="Wszystkie",
                                label="Gender / Płeć",
                            )
                            income = gr.Dropdown(
                                choices=["Wszystkie", "Niski", "Średni", "Wysoki"],
                                value="Wszystkie",
                                label="Income / Dochód",
                            )
                            location = gr.Dropdown(
                                choices=["Wszystkie", "Miasto", "Przedmieścia", "Wieś"],
                                value="Wszystkie",
                                label="Location / Lokalizacja",
                            )

                        n_agents = gr.Slider(5, 100, value=20, step=5, label="Number of agents / Liczba agentów")

                        with gr.Accordion("🔍 Web Search (RAG)", open=False):
                            enable_web_search = gr.Checkbox(
                                label="Enable Google Search / Włącz wyszukiwanie",
                                value=False,
                                info="Agents will search the web for market info / Agenci wyszukają informacje o rynku"
                            )

                        with gr.Accordion("⚙️ Advanced Settings / Zaawansowane", open=False):
                            temperature = gr.Slider(
                                minimum=0.01, 
                                maximum=2.0, 
                                value=0.01, 
                                step=0.01, 
                                label="SSR Temperature / Temperatura",
                                info="Lower = more decisive (default 0.01), Higher = smoother (paper 1.0) / Niższa = bardziej zdecydowane"
                            )

                        run_btn = gr.Button("🚀 Run Simulation / Uruchom", variant="primary")
                        status = gr.Markdown("")

                    with gr.Column(scale=1):
                        summary_output = gr.Markdown(label="Summary / Podsumowanie")
                        chart_output = gr.BarPlot(
                            x="Odpowiedź",
                            y="Procent",
                            title="Purchase Intent Distribution / Rozkład intencji zakupu",
                            height=300,
                        )

                with gr.Accordion("📝 Sample Agent Opinions / Przykładowe opinie", open=False):
                    opinions_output = gr.Markdown()

                # Report generation section
                gr.Markdown("### 📄 Report / Raport")
                with gr.Row():
                    report_btn = gr.Button("👁️ Generate Preview / Generuj podgląd", variant="secondary")
                    report_status = gr.Markdown("")
                
                # Report preview (HTML iframe)
                report_preview = gr.HTML(
                    label="Report Preview / Podgląd raportu",
                    value="<div style='text-align: center; padding: 40px; color: #666; border: 2px dashed #ccc; border-radius: 8px;'>Generate preview first / Najpierw wygeneruj podgląd</div>",
                )
                
                # Export section
                with gr.Row():
                    export_format = gr.Dropdown(
                        choices=["HTML", "PDF"],
                        value="HTML",
                        label="📁 Export format / Format eksportu",
                        scale=1,
                    )
                    export_btn = gr.Button("💾 Export / Eksportuj", variant="primary", scale=1)
                export_status = gr.Markdown("")
                export_file = gr.File(label="📥 Download / Pobierz", visible=True)

                run_btn.click(
                    fn=run_simulation,
                    inputs=[
                        language_select,
                        product_input,
                        n_agents,
                        age_min,
                        age_max,
                        gender,
                        income,
                        location,
                        enable_web_search,
                        temperature,
                        project_state,
                    ],
                    outputs=[
                        chart_output,
                        summary_output,
                        opinions_output,
                        status,
                        project_dirty,
                        extracted_product_preview,
                    ],
                )

                report_btn.click(
                    fn=generate_report,
                    inputs=[language_select],
                    outputs=[report_preview, report_status],
                )

                export_btn.click(
                    fn=export_report,
                    inputs=[language_select, export_format],
                    outputs=[export_file, export_status],
                )

            # === Tab 2: A/B Testing ===
            with gr.TabItem("🔬 A/B Test"):
                gr.Markdown("### Compare Two Product Variants / Porównaj dwie wersje produktu")

                with gr.Row():
                    variant_a_input = gr.Textbox(
                        label="Variant A / Wariant A",
                        placeholder="Description of first product variant...",
                        lines=4,
                    )
                    variant_b_input = gr.Textbox(
                        label="Variant B / Wariant B",
                        placeholder="Description of second product variant...",
                        lines=4,
                    )

                ab_n_agents = gr.Slider(10, 100, value=30, step=10, label="Agents per variant / Agentów na wariant")
                ab_run_btn = gr.Button("🔬 Run A/B Test / Uruchom test", variant="primary")
                ab_status = gr.Markdown("")
                ab_result = gr.Markdown()

                ab_run_btn.click(
                    fn=run_ab_test,
                    inputs=[language_select, variant_a_input, variant_b_input, ab_n_agents],
                    outputs=[ab_result, ab_status, project_dirty],
                )

            # === Tab 3: Price Sensitivity ===
            with gr.TabItem("💰 Price Analysis / Analiza Cenowa"):
                gr.Markdown("### Price Sensitivity Analysis / Analiza wrażliwości cenowej")

                price_product = gr.Textbox(
                    label="Product description (without price) / Opis produktu (bez ceny)",
                    placeholder="Describe the product without price information...",
                    lines=3,
                )

                with gr.Row():
                    price_min = gr.Number(value=19.99, label="Price min / Cena min")
                    price_max = gr.Number(value=59.99, label="Price max / Cena max")
                    price_points = gr.Slider(3, 7, value=5, step=1, label="Price points / Punkty cenowe")

                price_n_agents = gr.Slider(10, 50, value=20, step=5, label="Agents per price / Agentów na cenę")
                price_run_btn = gr.Button("💰 Analyze / Analizuj", variant="primary")
                price_status = gr.Markdown("")

                with gr.Row():
                    price_result = gr.Markdown()
                    price_chart = gr.LinePlot(
                        x="Cena",
                        y="Intencja",
                        title="Demand Curve / Krzywa popytu",
                        height=300,
                    )

                price_run_btn.click(
                    fn=run_price_analysis,
                    inputs=[language_select, price_product, price_min, price_max, price_points, price_n_agents],
                    outputs=[price_result, price_chart, price_status, project_dirty],
                )

            # === Tab 4: Focus Group ===
            with gr.TabItem("🎯 Focus Group / Grupa fokusowa"):
                gr.Markdown("### Virtual Focus Group / Wirtualna grupa fokusowa")
                gr.Markdown("*Multi-agent discussion about your product / Dyskusja wielu agentów o Twoim produkcie*")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        fg_product = gr.Textbox(
                            label="Product description / Opis produktu",
                            placeholder="Describe your product...",
                            lines=3,
                        )
                    with gr.Column(scale=1):
                        fg_participants = gr.Slider(
                            label="Participants / Uczestnicy",
                            minimum=4,
                            maximum=8,
                            value=6,
                            step=1,
                        )
                        fg_rounds = gr.Slider(
                            label="Discussion rounds / Rundy dyskusji",
                            minimum=2,
                            maximum=4,
                            value=3,
                            step=1,
                        )
                
                fg_status = gr.Markdown("*Ready / Gotowe*")
                fg_run_btn = gr.Button("🎯 Start Focus Group / Rozpocznij dyskusję", variant="primary")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 💬 Discussion Transcript / Transkrypcja")
                        fg_transcript = gr.Markdown(
                            value="",
                            label="Transcript",
                        )
                    with gr.Column():
                        gr.Markdown("### 📋 Moderator Summary / Podsumowanie")
                        fg_summary = gr.Markdown(
                            value="",
                            label="Summary",
                        )
                
                fg_run_btn.click(
                    fn=run_focus_group,
                    inputs=[language_select, fg_product, fg_participants, fg_rounds, project_state],
                    outputs=[fg_transcript, fg_summary, fg_status, project_dirty],
                )
                
                # Export section
                with gr.Row():
                    fg_export_format = gr.Dropdown(
                        choices=["HTML", "PDF"],
                        value="HTML",
                        label="📄 Export format / Format eksportu",
                        scale=0,
                        min_width=120,
                    )
                    fg_export_btn = gr.Button("📥 Export", variant="secondary")
                    fg_export_file = gr.File(label="Download", visible=True)
                    fg_export_status = gr.Markdown("")
                
                fg_export_btn.click(
                    fn=export_focus_group,
                    inputs=[language_select, fg_export_format],
                    outputs=[fg_export_file, fg_export_status],
                )

            # === Tab 5: Projects ===
            with gr.TabItem("📁 Projects / Projekty"):
                gr.Markdown("### Project Manager / Menedżer projektów")

                with gr.Row():
                    project_name = gr.Textbox(
                        label="Project name / Nazwa projektu",
                        placeholder="E.g. Charcoal Toothpaste Research",
                    )
                    save_project_btn = gr.Button("💾 Save / Zapisz", variant="primary")

                save_project_status = gr.Markdown("")

                with gr.Row():
                    initial_projects = get_project_store().list_projects()
                    initial_choices, _ = _project_dropdown_choices(initial_projects)
                    project_select = gr.Dropdown(
                        label="Saved projects / Zapisane projekty",
                        choices=initial_choices,
                        value=None,
                    )
                    refresh_projects_btn = gr.Button("🔄 Refresh / Odśwież", variant="secondary")

                project_select.change(
                    fn=lambda value: value,
                    inputs=[project_select],
                    outputs=[project_state],
                )

                gr.Markdown(
                    "*Select a project to update it; leave empty to create a new one. / "
                    "Wybierz projekt, aby go zaktualizować; pozostaw puste, aby utworzyć nowy.*"
                )

                with gr.Row():
                    load_project_btn = gr.Button("📂 Load / Wczytaj", variant="primary")
                    delete_project_btn = gr.Button("🗑️ Delete / Usuń", variant="secondary")

                auto_load = gr.Checkbox(
                    label="🔁 Auto-wczytanie po wyborze / Auto-load on select",
                    value=False,
                )

                project_action_status = gr.Markdown("")
                with gr.Row():
                    save_then_load_btn = gr.Button(
                        "✅ Tak, zapisz i wczytaj / Yes, save and load",
                        variant="secondary",
                        visible=False,
                    )
                    load_without_saving_btn = gr.Button(
                        "⚠️ Nie, wczytaj bez zapisu / No, load without saving",
                        variant="secondary",
                        visible=False,
                    )

                project_info = gr.Markdown()

                with gr.Accordion("📊 Simulation / Symulacja", open=False):
                    saved_sim_summary = gr.Markdown()
                    saved_sim_chart = gr.BarPlot(
                        x="Odpowiedź",
                        y="Procent",
                        title="Purchase Intent Distribution / Rozkład intencji zakupu",
                        height=300,
                    )
                    saved_sim_opinions = gr.Markdown()

                with gr.Accordion("🔬 A/B Test", open=False):
                    saved_ab_summary = gr.Markdown()

                with gr.Accordion("💰 Price Analysis / Analiza Cenowa", open=False):
                    saved_price_summary = gr.Markdown()
                    saved_price_chart = gr.LinePlot(
                        x="Cena",
                        y="Intencja",
                        title="Demand Curve / Krzywa popytu",
                        height=300,
                    )

                with gr.Accordion("🎯 Focus Group / Grupa fokusowa", open=False):
                    saved_fg_transcript = gr.Markdown()
                    saved_fg_summary = gr.Markdown()

                project_select.change(
                    fn=maybe_autoload_project,
                    inputs=[
                        language_select,
                        project_select,
                        allow_discard_state,
                        auto_load,
                        project_dirty,
                    ],
                    outputs=[
                        project_info,
                        saved_sim_summary,
                        saved_sim_chart,
                        saved_sim_opinions,
                        saved_ab_summary,
                        saved_price_summary,
                        saved_price_chart,
                        saved_fg_transcript,
                        saved_fg_summary,
                        project_action_status,
                        save_then_load_btn,
                        load_without_saving_btn,
                        project_name,
                        product_input,
                        age_min,
                        age_max,
                        gender,
                        income,
                        location,
                        n_agents,
                        enable_web_search,
                        temperature,
                        variant_a_input,
                        variant_b_input,
                        ab_n_agents,
                        price_product,
                        price_min,
                        price_max,
                        price_points,
                        price_n_agents,
                        fg_product,
                        fg_participants,
                        fg_rounds,
                        summary_output,
                        chart_output,
                        opinions_output,
                        ab_result,
                        price_result,
                        price_chart,
                        fg_transcript,
                        fg_summary,
                        project_dirty,
                        suppress_dirty,
                    ],
                ).then(
                    fn=clear_suppress_dirty,
                    outputs=[suppress_dirty],
                )

                save_project_btn.click(
                    fn=save_project,
                    inputs=[
                        language_select,
                        project_name,
                        project_select,
                        product_input,
                        age_min,
                        age_max,
                        gender,
                        income,
                        location,
                        n_agents,
                        enable_web_search,
                        temperature,
                        variant_a_input,
                        variant_b_input,
                        ab_n_agents,
                        price_product,
                        price_min,
                        price_max,
                        price_points,
                        price_n_agents,
                        fg_product,
                        fg_participants,
                        fg_rounds,
                    ],
                    outputs=[save_project_status, project_select, project_state, project_dirty],
                )

                refresh_projects_btn.click(
                    fn=list_saved_projects,
                    inputs=[language_select],
                    outputs=[project_select, project_state, project_dirty],
                )

                load_project_btn.click(
                    fn=load_project,
                    inputs=[language_select, project_select, allow_discard_state, project_dirty],
                    outputs=[
                        project_info,
                        saved_sim_summary,
                        saved_sim_chart,
                        saved_sim_opinions,
                        saved_ab_summary,
                        saved_price_summary,
                        saved_price_chart,
                        saved_fg_transcript,
                        saved_fg_summary,
                        project_action_status,
                        save_then_load_btn,
                        load_without_saving_btn,
                        project_name,
                        product_input,
                        age_min,
                        age_max,
                        gender,
                        income,
                        location,
                        n_agents,
                        enable_web_search,
                        temperature,
                        variant_a_input,
                        variant_b_input,
                        ab_n_agents,
                        price_product,
                        price_min,
                        price_max,
                        price_points,
                        price_n_agents,
                        fg_product,
                        fg_participants,
                        fg_rounds,
                        summary_output,
                        chart_output,
                        opinions_output,
                        ab_result,
                        price_result,
                        price_chart,
                        fg_transcript,
                        fg_summary,
                        project_dirty,
                        suppress_dirty,
                    ],
                ).then(
                    fn=clear_suppress_dirty,
                    outputs=[suppress_dirty],
                )

                save_then_load_btn.click(
                    fn=save_project,
                    inputs=[
                        language_select,
                        project_name,
                        project_select,
                        product_input,
                        age_min,
                        age_max,
                        gender,
                        income,
                        location,
                        n_agents,
                        enable_web_search,
                        temperature,
                        variant_a_input,
                        variant_b_input,
                        ab_n_agents,
                        price_product,
                        price_min,
                        price_max,
                        price_points,
                        price_n_agents,
                        fg_product,
                        fg_participants,
                        fg_rounds,
                    ],
                    outputs=[save_project_status, project_select, project_state, project_dirty],
                ).then(
                    fn=load_project,
                    inputs=[language_select, project_select, allow_discard_state, project_dirty],
                    outputs=[
                        project_info,
                        saved_sim_summary,
                        saved_sim_chart,
                        saved_sim_opinions,
                        saved_ab_summary,
                        saved_price_summary,
                        saved_price_chart,
                        saved_fg_transcript,
                        saved_fg_summary,
                        project_action_status,
                        save_then_load_btn,
                        load_without_saving_btn,
                        project_name,
                        product_input,
                        age_min,
                        age_max,
                        gender,
                        income,
                        location,
                        n_agents,
                        enable_web_search,
                        temperature,
                        variant_a_input,
                        variant_b_input,
                        ab_n_agents,
                        price_product,
                        price_min,
                        price_max,
                        price_points,
                        price_n_agents,
                        fg_product,
                        fg_participants,
                        fg_rounds,
                        summary_output,
                        chart_output,
                        opinions_output,
                        ab_result,
                        price_result,
                        price_chart,
                        fg_transcript,
                        fg_summary,
                        project_dirty,
                        suppress_dirty,
                    ],
                ).then(
                    fn=clear_suppress_dirty,
                    outputs=[suppress_dirty],
                )

                load_without_saving_btn.click(
                    fn=load_project,
                    inputs=[language_select, project_select, allow_discard_true, project_dirty],
                    outputs=[
                        project_info,
                        saved_sim_summary,
                        saved_sim_chart,
                        saved_sim_opinions,
                        saved_ab_summary,
                        saved_price_summary,
                        saved_price_chart,
                        saved_fg_transcript,
                        saved_fg_summary,
                        project_action_status,
                        save_then_load_btn,
                        load_without_saving_btn,
                        project_name,
                        product_input,
                        age_min,
                        age_max,
                        gender,
                        income,
                        location,
                        n_agents,
                        enable_web_search,
                        temperature,
                        variant_a_input,
                        variant_b_input,
                        ab_n_agents,
                        price_product,
                        price_min,
                        price_max,
                        price_points,
                        price_n_agents,
                        fg_product,
                        fg_participants,
                        fg_rounds,
                        summary_output,
                        chart_output,
                        opinions_output,
                        ab_result,
                        price_result,
                        price_chart,
                        fg_transcript,
                        fg_summary,
                        project_dirty,
                        suppress_dirty,
                    ],
                ).then(
                    fn=clear_suppress_dirty,
                    outputs=[suppress_dirty],
                )

                delete_project_btn.click(
                    fn=delete_project,
                    inputs=[language_select, project_select],
                    outputs=[project_action_status, project_select, project_state, project_dirty],
                )

            # === Tab 6: About ===
            with gr.TabItem("ℹ️ About / O metodologii"):
                gr.Markdown(
                    """
                    ## SSR Methodology (Semantic Similarity Rating)
                    
                    Market Wizard uses the SSR methodology described in:
                    
                    > **Maier, B. F., et al. (2025).** *"LLMs Reproduce Human Purchase Intent 
                    > via Semantic Similarity Elicitation of Likert Ratings"*. 
                    > [arXiv:2510.08338](https://arxiv.org/abs/2510.08338)
                    
                    ### How it works / Jak to działa?
                    
                    1. **Persona generation** - Creates synthetic consumers with realistic 
                       demographic profiles (age, income, location)
                    
                    2. **Opinion generation** - Each persona evaluates the product using AI 
                       (Gemini), generating natural text responses
                    
                    3. **SSR mapping** - Text responses are converted to Likert scale (1-5) 
                       by comparing embeddings with "anchor" statements
                    
                    4. **Aggregation** - Results from all agents are aggregated into a 
                       statistical Purchase Intent distribution
                    
                    ### Why SSR? / Dlaczego SSR?
                    
                    Traditional "rate from 1 to 5" prompts lead to:
                    - Regression to the mean (responses clustered around 3)
                    - Low correlation with real data (~80%)
                    
                    SSR achieves **90% correlation** with actual purchase decisions!
                    
                    ---
                    
                    ### Language Support / Obsługa języków
                    
                    This app supports both **Polish (PL)** and **English (EN)**. 
                    Use the language selector at the top to switch between them.
                    
                    Personas, prompts, and anchor statements are all localized.
            """
        )

        demo.load(
            fn=list_saved_projects,
            inputs=[language_select],
            outputs=[project_select, project_state, project_dirty],
        )

        dirty_inputs = [
            product_input,
            age_min,
            age_max,
            gender,
            income,
            location,
            n_agents,
            enable_web_search,
            temperature,
            variant_a_input,
            variant_b_input,
            ab_n_agents,
            price_product,
            price_min,
            price_max,
            price_points,
            price_n_agents,
            fg_product,
            fg_participants,
            fg_rounds,
        ]
        mark_dirty_inputs = dirty_inputs + [suppress_dirty]
        for comp in dirty_inputs:
            comp.change(
                fn=mark_dirty,
                inputs=mark_dirty_inputs,
                outputs=[project_dirty],
            )

        gr.Markdown(
            """
            ---
            *Market Wizard v0.2.0 | Based on arXiv:2510.08338 | PL/EN*
            """
        )

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
