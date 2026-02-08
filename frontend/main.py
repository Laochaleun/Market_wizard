"""
Market Wizard - Gradio Frontend

Interactive dashboard for running SSR-based market research simulations.
Supports Polish (PL) and English (EN) languages.
"""

import asyncio
import logging
import os
import tempfile
from urllib.parse import urljoin, urlparse, parse_qsl, urlencode, urlunparse
from datetime import datetime
from pathlib import Path
from typing import Optional

import gradio as gr
from gradio import processing_utils, utils as gr_utils
import numpy as np
import pandas as pd
import plotly.graph_objects as go

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
from app.services.llm_client import get_report_analysis_client
from app.services.report_generator import generate_html_report, save_report
from app.i18n import Language, get_label
from app.config import get_settings
from app.services.embedding_client import warmup_embeddings

settings = get_settings()

# Store last simulation result for report generation
_last_simulation_result = None
_last_product_description = None
_last_product_input_raw = None
_last_simulation_inputs = None
_last_extracted_preview = None
_last_extracted_full = None
_last_extracted_url = None

_last_ab_test_result = None
_last_ab_test_inputs = None

_last_price_analysis_result = None
_last_price_analysis_inputs = None

_last_focus_group_inputs = None
_last_report_analysis = None
_last_report_analysis_key = None


# Warm up local embeddings on startup to ensure model is downloaded (local & HF).
warmup_embeddings()


# === Helper Functions ===


def get_lang(lang_code: str) -> Language:
    """Convert language code string to Language enum."""
    return Language.EN if lang_code == "English" else Language.PL


def create_histogram_chart(distribution: dict, lang: Language) -> go.Figure:
    """Convert distribution dict to a Plotly bar chart."""
    if lang == Language.PL:
        labels = ["1-Nie", "2-Raczej nie", "3-Ani tak, ani nie", "4-Raczej tak", "5-Tak"]
        x_title, y_title = "Odpowiedź", "Procent respondentów"
        chart_title = "Purchase Intent Distribution / Rozkład intencji zakupu"
    else:
        labels = ["1-No", "2-Probably not", "3-Neutral", "4-Probably yes", "5-Yes"]
        x_title, y_title = "Response", "Percent"
        chart_title = "Purchase Intent Distribution / Rozkład intencji zakupu"
    
    values = [
        distribution.get("scale_1", 0) * 100,
        distribution.get("scale_2", 0) * 100,
        distribution.get("scale_3", 0) * 100,
        distribution.get("scale_4", 0) * 100,
        distribution.get("scale_5", 0) * 100,
    ]
    colors = ["#1e3a5f", "#2d5a87", "#4a7c9b", "#6b9db8", "#8ebfd4"]

    max_val = max(values) if values else 0
    y_max = max(5, max_val * 1.2) if max_val > 0 else 100

    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=values,
                marker_color=colors,
                text=[f"{v:.1f}%" for v in values],
                textposition="outside",
                cliponaxis=False,
                hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=chart_title,
        height=300,
        autosize=True,
        width=None,
        margin=dict(l=40, r=20, t=50, b=40),
        yaxis=dict(
            title=y_title,
            range=[0, y_max],
            ticksuffix="%",
            gridcolor="rgba(0,0,0,0.08)",
            zerolinecolor="rgba(0,0,0,0.2)",
            linecolor="rgba(0,0,0,0.4)",
        ),
        xaxis=dict(
            title=x_title,
            linecolor="rgba(0,0,0,0.4)",
        ),
        showlegend=False,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(color="#000000"),
    )
    fig.update_layout(
        dragmode=False,
        modebar=dict(
            remove=["zoom", "pan", "select", "lasso", "zoomIn", "zoomOut", "autoScale", "resetScale", "toImage"]
        ),
    )
    return fig


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


def _shorten_text(text: str, max_chars: int = 400) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _build_report_analysis_payload(
    result: SimulationResult,
    product_description: str,
    lang: Language,
) -> dict:
    scores = [r.likert_score for r in result.agent_responses]
    incomes = [r.persona.income for r in result.agent_responses]
    ages = [r.persona.age for r in result.agent_responses]
    dist = result.aggregate_distribution

    sorted_responses = sorted(result.agent_responses, key=lambda r: r.likert_score, reverse=True)
    top = sorted_responses[:6]
    bottom = sorted_responses[-6:] if len(sorted_responses) > 6 else []

    def pack_response(r):
        return {
            "score": round(r.likert_score, 2),
            "text": _shorten_text(r.text_response),
            "persona": {
                "age": r.persona.age,
                "gender": r.persona.gender,
                "location": r.persona.location,
                "income": r.persona.income,
                "occupation": r.persona.occupation,
            },
        }

    payload = {
        "language": lang.value,
        "product_description": product_description,
        "n_agents": result.n_agents,
        "mean_score": float(np.mean(scores)) if scores else None,
        "std_score": float(np.std(scores)) if scores else None,
        "min_score": float(min(scores)) if scores else None,
        "max_score": float(max(scores)) if scores else None,
        "distribution_pct": {
            "1": round(dist.scale_1 * 100, 1),
            "2": round(dist.scale_2 * 100, 1),
            "3": round(dist.scale_3 * 100, 1),
            "4": round(dist.scale_4 * 100, 1),
            "5": round(dist.scale_5 * 100, 1),
        },
        "demographics": {
            "mean_age": round(float(np.mean(ages)), 1) if ages else None,
            "mean_income": round(float(np.mean(incomes)), 0) if incomes else None,
            "gender_m": sum(1 for r in result.agent_responses if r.persona.gender == "M"),
            "gender_f": sum(1 for r in result.agent_responses if r.persona.gender == "F"),
        },
        "top_responses": [pack_response(r) for r in top],
        "bottom_responses": [pack_response(r) for r in bottom],
        "sources_count": len(result.web_sources or []),
    }
    return payload


async def _generate_report_analysis_async(
    result: SimulationResult,
    product_description: str,
    lang: Language,
) -> dict[str, str] | None:
    client = get_report_analysis_client()
    if not client:
        return None
    payload = _build_report_analysis_payload(result, product_description, lang)
    analysis_sections = await client.generate_report_analysis(payload=payload, language=lang)
    if not analysis_sections:
        return None
    # Optional sanitize pass to enforce literal, neutral style
    try:
        if hasattr(client, "sanitize_report_analysis"):
            sanitized = await client.sanitize_report_analysis(
                analysis_sections=analysis_sections,
                language=lang,
            )
            if sanitized and any((sanitized.get("narrative"), sanitized.get("agent_summary"), sanitized.get("recommendations"))):
                return sanitized
    except Exception as e:
        logging.getLogger(__name__).warning("Report analysis sanitize failed: %s", e)
    return analysis_sections


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
            f"- Ani tak, ani nie: {dist['scale_3']*100:.1f}%\n"
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

    if "seed" in result:
        if lang == Language.PL:
            curve_text += f"\n\n*Stabilizacja: stała populacja person. Seed: {result['seed']}*"
        else:
            curve_text += f"\n\n*Stabilization: fixed persona population. Seed: {result['seed']}*"

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
    region = audience.get("region")

    if lang == Language.PL:
        gender_label = "Wszystkie" if not gender else ("K" if gender == "F" else gender)
        income_map = {
            "very_low": "Najniższy (do 3500 PLN)",
            "low": "Niski (3500-5000 PLN)",
            "medium": "Średni (5000-7000 PLN)",
            "high": "Wysoki (7000-10000 PLN)",
            "very_high": "Najwyższy (10000+ PLN)",
        }
        location_map = {
            "rural": "Wieś",
            "small_city": "Małe miasto",
            "medium_city": "Średnie miasto",
            "large_city": "Duże miasto",
            "metropolis": "Metropolia",
            "urban": "Miasto",
            "suburban": "Przedmieścia",
        }
        income_label = income_map.get(income_level, "Wszystkie")
        location_label = location_map.get(location_type, "Wszystkie")
        region_label = region or "Wszystkie regiony"
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
            f"dochód {income_label}, lokalizacja {location_label}, region {region_label}\n\n"
            f"**Zapisane badania:** {research_label}\n"
        )

    gender_label = "All" if not gender else gender
    income_map = {
        "very_low": "Very Low (up to 3500 PLN)",
        "low": "Low (3500-5000 PLN)",
        "medium": "Medium (5000-7000 PLN)",
        "high": "High (7000-10000 PLN)",
        "very_high": "Very High (10000+ PLN)",
    }
    location_map = {
        "rural": "Rural",
        "small_city": "Small city",
        "medium_city": "Medium city",
        "large_city": "Large city",
        "metropolis": "Metropolis",
        "urban": "Urban",
        "suburban": "Suburban",
    }
    income_label = income_map.get(income_level, "All")
    location_label = location_map.get(location_type, "All")
    region_label = region or "All regions"
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
        f"income {income_label}, location {location_label}, region {region_label}\n\n"
        f"**Saved research:** {research_label}\n"
    )


def get_project_store() -> ProjectStore:
    """Return project store pinned to repo root."""
    return ProjectStore(base_dir=PROJECT_BASE_DIR)


INCOME_CHOICES = {
    Language.PL: [
        "Wszystkie",
        "Najniższy (do 3500 PLN)",
        "Niski (3500-5000 PLN)",
        "Średni (5000-7000 PLN)",
        "Wysoki (7000-10000 PLN)",
        "Najwyższy (10000+ PLN)",
    ],
    Language.EN: [
        "All",
        "Very low (up to 3500 PLN)",
        "Low (3500-5000 PLN)",
        "Medium (5000-7000 PLN)",
        "High (7000-10000 PLN)",
        "Very high (10000+ PLN)",
    ],
}

EDUCATION_CHOICES = {
    Language.PL: [
        "Wszystkie",
        "Podstawowe",
        "Zasadnicze zawodowe",
        "Średnie",
        "Policealne",
        "Wyższe",
    ],
    Language.EN: [
        "All",
        "Primary",
        "Vocational",
        "Secondary",
        "Post-secondary",
        "Higher",
    ],
}

LOCATION_CHOICES = {
    Language.PL: [
        "Wszystkie",
        "Wieś (41% populacji)",
        "Małe miasto do 20 tys. (12%)",
        "Średnie miasto 20-100 tys. (18%)",
        "Duże miasto 100-500 tys. (16%)",
        "Metropolia 500 tys.+ (13%)",
    ],
    Language.EN: [
        "All",
        "Rural (41% population)",
        "Small city up to 20k (12%)",
        "Medium city 20k-100k (18%)",
        "Large city 100k-500k (16%)",
        "Metropolis 500k+ (13%)",
    ],
}

REGION_CHOICES = {
    Language.PL: [
        "Wszystkie regiony",
        "dolnośląskie",
        "kujawsko-pomorskie",
        "lubelskie",
        "lubuskie",
        "łódzkie",
        "małopolskie",
        "mazowieckie",
        "opolskie",
        "podkarpackie",
        "podlaskie",
        "pomorskie",
        "śląskie",
        "świętokrzyskie",
        "warmińsko-mazurskie",
        "wielkopolskie",
        "zachodniopomorskie",
    ],
    Language.EN: [
        "All regions",
        "dolnośląskie",
        "kujawsko-pomorskie",
        "lubelskie",
        "lubuskie",
        "łódzkie",
        "małopolskie",
        "mazowieckie",
        "opolskie",
        "podkarpackie",
        "podlaskie",
        "pomorskie",
        "śląskie",
        "świętokrzyskie",
        "warmińsko-mazurskie",
        "wielkopolskie",
        "zachodniopomorskie",
    ],
}


def mark_dirty(
    product_description: str,
    age_min: int,
    age_max: int,
    gender: str,
    income_level: str,
    education_level: str,
    location_type: str,
    region: str,
    n_agents: int,
    enable_web_search: bool,
    temperature: float,
    variant_a_input: str,
    variant_b_input: str,
    ab_n_agents: int,
    ab_enable_web_search: bool,
    price_product: str,
    price_min: float,
    price_max: float,
    price_points: int,
    price_n_agents: int,
    price_enable_web_search: bool,
    fg_product: str,
    fg_participants: int,
    fg_rounds: int,
    fg_enable_web_search: bool,
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
        "education_level": "Wszystkie",
        "location_type": "Wszystkie",
        "region": None,
        "n_agents": 20,
        "enable_web_search": True,
        "temperature": 1.0,
        "variant_a_input": "",
        "variant_b_input": "",
        "ab_n_agents": 30,
        "ab_enable_web_search": True,
        "price_product": "",
        "price_min": 19.99,
        "price_max": 59.99,
        "price_points": 5,
        "price_n_agents": 50,
        "price_enable_web_search": True,
        "fg_product": "",
        "fg_participants": 6,
        "fg_rounds": 3,
        "fg_enable_web_search": True,
    }
    current = {
        "product_description": (product_description or "").strip(),
        "age_min": int(age_min),
        "age_max": int(age_max),
        "gender": gender,
        "income_level": income_level,
        "education_level": education_level,
        "location_type": location_type,
        "region": normalize_region_value(region),
        "n_agents": int(n_agents),
        "enable_web_search": bool(enable_web_search),
        "temperature": float(temperature),
        "variant_a_input": (variant_a_input or "").strip(),
        "variant_b_input": (variant_b_input or "").strip(),
        "ab_n_agents": int(ab_n_agents),
        "ab_enable_web_search": bool(ab_enable_web_search),
        "price_product": (price_product or "").strip(),
        "price_min": float(price_min),
        "price_max": float(price_max),
        "price_points": int(price_points),
        "price_n_agents": int(price_n_agents),
        "price_enable_web_search": bool(price_enable_web_search),
        "fg_product": (fg_product or "").strip(),
        "fg_participants": int(fg_participants),
        "fg_rounds": int(fg_rounds),
        "fg_enable_web_search": bool(fg_enable_web_search),
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
    region: str,
) -> dict:
    """Normalize UI inputs to stored demographic fields."""
    all_values = {"All", "Wszystkie"}
    income_map = {
        # New GUS-based categories with ranges
        "Najniższy (do 3500 PLN)": "very_low",
        "Niski (3500-5000 PLN)": "low",
        "Średni (5000-7000 PLN)": "medium",
        "Wysoki (7000-10000 PLN)": "high",
        "Najwyższy (10000+ PLN)": "very_high",
        # Legacy mappings for backwards compatibility
        "Very Low": "very_low",
        "Low": "low",
        "Medium": "medium",
        "High": "high",
        "Very High": "very_high",
        "Niski": "low",
        "Średni": "medium",
        "Wysoki": "high",
    }
    location_map = {
        # New GUS 2024 categories with population percentages
        "Wieś (41% populacji)": "rural",
        "Małe miasto do 20 tys. (12%)": "small_city",
        "Średnie miasto 20-100 tys. (18%)": "medium_city",
        "Duże miasto 100-500 tys. (16%)": "large_city",
        "Metropolia 500 tys.+ (13%)": "metropolis",
        # Legacy mappings for backwards compatibility
        "Urban": "large_city",
        "Suburban": "medium_city",
        "Rural": "rural",
        "Miasto": "large_city",
        "Przedmieścia": "medium_city",
        "Wieś": "rural",
    }

    income = income_map.get(income_level) if income_level not in all_values else None
    location = location_map.get(location_type) if location_type not in all_values else None

    gender_value = normalize_gender_value(gender)
    region_value = normalize_region_value(region)

    return {
        "age_min": age_min,
        "age_max": age_max,
        "gender": gender_value,
        "income_level": income,
        "location_type": location,
        "region": region_value,
    }


def normalize_gender_value(gender: Optional[str]) -> Optional[str]:
    """Normalize UI gender value to backend format: M/F/None."""
    value = (gender or "").strip().upper()
    if value in {"M", "F"}:
        return value
    if value == "K":
        return "F"
    if value in {"ALL", "WSZYSTKIE", "WSZYSCY", ""}:
        return None
    return None


def normalize_region_value(region: Optional[str]) -> Optional[str]:
    """Normalize UI region value to backend format: voivodeship key or None."""
    value = (region or "").strip().lower()
    if value in {"", "all", "all regions", "wszystkie", "wszystkie regiony"}:
        return None
    valid_regions = {r.lower() for r in REGION_CHOICES[Language.PL][1:]}
    if value in valid_regions:
        return value
    return None


def update_demographic_dropdowns(
    lang_code: str,
    current_gender: str,
    current_income: str,
    current_education: str,
    current_location: str,
    current_region: str,
):
    """Update demographic dropdown choices and preserve equivalent selections."""
    lang = get_lang(lang_code)
    all_gender = "All" if lang == Language.EN else "Wszystkie"
    all_region = "All regions" if lang == Language.EN else "Wszystkie regiony"

    gender_norm = normalize_gender_value(current_gender)
    gender_value = all_gender
    if gender_norm:
        if gender_norm == "F":
            gender_value = "F" if lang == Language.EN else "K"
        elif gender_norm == "M":
            gender_value = "M"

    income_level = normalize_income_value(current_income)
    income_value = income_label_from_level(income_level, lang)

    education_level = normalize_education_value(current_education)
    education_value = education_label_from_level(education_level, lang)

    location_type = normalize_location_value(current_location)
    location_value = location_label_from_type(location_type, lang)

    region_value = normalize_region_value(current_region)
    region_ui_value = region_value if region_value else all_region

    gender_choices = [all_gender, "M", "F"] if lang == Language.EN else [all_gender, "M", "K"]
    return (
        gr.update(choices=gender_choices, value=gender_value),
        gr.update(choices=INCOME_CHOICES[lang], value=income_value),
        gr.update(choices=EDUCATION_CHOICES[lang], value=education_value),
        gr.update(choices=LOCATION_CHOICES[lang], value=location_value),
        gr.update(choices=REGION_CHOICES[lang], value=region_ui_value),
    )


def normalize_income_value(income_level: Optional[str]) -> Optional[str]:
    all_values = {"All", "Wszystkie"}
    income_map = {
        "Najniższy (do 3500 PLN)": "very_low",
        "Niski (3500-5000 PLN)": "low",
        "Średni (5000-7000 PLN)": "medium",
        "Wysoki (7000-10000 PLN)": "high",
        "Najwyższy (10000+ PLN)": "very_high",
        "Very low (up to 3500 PLN)": "very_low",
        "Low (3500-5000 PLN)": "low",
        "Medium (5000-7000 PLN)": "medium",
        "High (7000-10000 PLN)": "high",
        "Very high (10000+ PLN)": "very_high",
        "Very Low": "very_low",
        "Low": "low",
        "Medium": "medium",
        "High": "high",
        "Very High": "very_high",
        "Niski": "low",
        "Średni": "medium",
        "Wysoki": "high",
    }
    if income_level in all_values or not income_level:
        return None
    return income_map.get(income_level)


def normalize_location_value(location_type: Optional[str]) -> Optional[str]:
    all_values = {"All", "Wszystkie"}
    loc_map = {
        "Wieś (41% populacji)": "rural",
        "Małe miasto do 20 tys. (12%)": "small_city",
        "Średnie miasto 20-100 tys. (18%)": "medium_city",
        "Duże miasto 100-500 tys. (16%)": "large_city",
        "Metropolia 500 tys.+ (13%)": "metropolis",
        "Rural (41% population)": "rural",
        "Small city up to 20k (12%)": "small_city",
        "Medium city 20k-100k (18%)": "medium_city",
        "Large city 100k-500k (16%)": "large_city",
        "Metropolis 500k+ (13%)": "metropolis",
        "Urban": "large_city",
        "Suburban": "medium_city",
        "Rural": "rural",
        "Miasto": "large_city",
        "Przedmieścia": "medium_city",
        "Wieś": "rural",
    }
    if location_type in all_values or not location_type:
        return None
    return loc_map.get(location_type)


def normalize_education_value(education_level: Optional[str]) -> Optional[str]:
    """Normalize education dropdown value to internal lowercase format."""
    all_values = {"All", "Wszystkie"}
    if education_level in all_values or not education_level:
        return None
    # Map UI values to internal lowercase format
    edu_map = {
        "Podstawowe": "podstawowe",
        "Zasadnicze zawodowe": "zasadnicze zawodowe",
        "Średnie": "średnie",
        "Policealne": "policealne",
        "Wyższe": "wyższe",
        "Primary": "primary",
        "Vocational": "vocational",
        "Secondary": "secondary",
        "Post-secondary": "post-secondary",
        "Higher": "higher",
    }
    return edu_map.get(education_level, education_level.lower())


def income_label_from_level(income_level: Optional[str], lang: Language) -> str:
    mapping = {
        Language.PL: {
            "very_low": "Najniższy (do 3500 PLN)",
            "low": "Niski (3500-5000 PLN)",
            "medium": "Średni (5000-7000 PLN)",
            "high": "Wysoki (7000-10000 PLN)",
            "very_high": "Najwyższy (10000+ PLN)",
        },
        Language.EN: {
            "very_low": "Very low (up to 3500 PLN)",
            "low": "Low (3500-5000 PLN)",
            "medium": "Medium (5000-7000 PLN)",
            "high": "High (7000-10000 PLN)",
            "very_high": "Very high (10000+ PLN)",
        },
    }
    return mapping[lang].get(income_level, INCOME_CHOICES[lang][0])


def location_label_from_type(location_type: Optional[str], lang: Language) -> str:
    mapping = {
        Language.PL: {
            "rural": "Wieś (41% populacji)",
            "small_city": "Małe miasto do 20 tys. (12%)",
            "medium_city": "Średnie miasto 20-100 tys. (18%)",
            "large_city": "Duże miasto 100-500 tys. (16%)",
            "metropolis": "Metropolia 500 tys.+ (13%)",
            "urban": "Duże miasto 100-500 tys. (16%)",
            "suburban": "Średnie miasto 20-100 tys. (18%)",
        },
        Language.EN: {
            "rural": "Rural (41% population)",
            "small_city": "Small city up to 20k (12%)",
            "medium_city": "Medium city 20k-100k (18%)",
            "large_city": "Large city 100k-500k (16%)",
            "metropolis": "Metropolis 500k+ (13%)",
            "urban": "Large city 100k-500k (16%)",
            "suburban": "Medium city 20k-100k (18%)",
        },
    }
    return mapping[lang].get(location_type, LOCATION_CHOICES[lang][0])


def education_label_from_level(education_level: Optional[str], lang: Language) -> str:
    # Maps internal keys (both PL and EN) to display labels
    mapping = {
        Language.PL: {
            "primary": "Podstawowe",
            "podstawowe": "Podstawowe",
            "vocational": "Zasadnicze zawodowe",
            "zasadnicze zawodowe": "Zasadnicze zawodowe",
            "secondary": "Średnie",
            "średnie": "Średnie",
            "post-secondary": "Policealne",
            "policealne": "Policealne",
            "higher": "Wyższe",
            "wyższe": "Wyższe",
        },
        Language.EN: {
            "primary": "Primary",
            "podstawowe": "Primary",
            "vocational": "Vocational",
            "zasadnicze zawodowe": "Vocational",
            "secondary": "Secondary",
            "średnie": "Secondary",
            "post-secondary": "Post-secondary",
            "policealne": "Post-secondary",
            "higher": "Higher",
            "wyższe": "Higher",
        },
    }
    # Handle normalized values (lowercase)
    norm_level = education_level.lower() if education_level else None
    return mapping[lang].get(norm_level, EDUCATION_CHOICES[lang][0])


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
        raw_input = inputs.get("product_input_raw")
        if raw_input:
            project["product_input_raw"] = raw_input
        extracted_full = inputs.get("product_extracted_full")
        if extracted_full:
            project["product_extracted_full"] = extracted_full
        extracted_preview = inputs.get("product_extracted_preview")
        if extracted_preview:
            project["product_extracted_preview"] = extracted_preview
        extracted_url = inputs.get("product_extracted_url")
        if extracted_url:
            project["product_extracted_url"] = extracted_url

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
) -> tuple[int, int, str, str, str, str]:
    """Map stored target audience fields to UI labels."""
    default_age_min = 25
    default_age_max = 45
    default_gender = "All" if lang == Language.EN else "Wszystkie"
    default_income = INCOME_CHOICES[lang][0]
    default_location = LOCATION_CHOICES[lang][0]
    default_region = REGION_CHOICES[lang][0]

    if not target_audience:
        return (
            default_age_min,
            default_age_max,
            default_gender,
            default_income,
            default_location,
            default_region,
        )

    age_min = int(target_audience.get("age_min", default_age_min))
    age_max = int(target_audience.get("age_max", default_age_max))

    gender_raw = target_audience.get("gender")
    if gender_raw == "F":
        gender = "F" if lang == Language.EN else "K"
    elif gender_raw == "M":
        gender = "M"
    else:
        gender = default_gender

    income_level = income_label_from_level(target_audience.get("income_level"), lang)
    location_type = location_label_from_type(target_audience.get("location_type"), lang)
    region_raw = (target_audience.get("region") or "").strip().lower()
    valid_regions = {r.lower() for r in REGION_CHOICES[Language.PL][1:]}
    region = region_raw if region_raw in valid_regions else default_region

    return age_min, age_max, gender, income_level, location_type, region


# === URL Detection and Product Extraction ===


import re


def normalize_url(text: str) -> str:
    """Normalize URLs by adding https:// when scheme is missing."""
    value = (text or "").strip()
    if not value:
        return value
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", value):
        return value
    if re.match(r"^[A-Z0-9.-]+\.[A-Z]{2,}(/.*)?$", value, re.IGNORECASE):
        return f"https://{value}"
    return value

def is_url(text: str) -> bool:
    """Check if text is a URL."""
    value = (text or "").strip()
    if not value:
        return False
    if re.match(r"^https?://", value, re.IGNORECASE):
        return True
    if re.match(r"^www\.", value, re.IGNORECASE):
        return True
    return False


def _shorten_extracted(text: str, lang: Language) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    # Keep a short, readable preview.
    max_chars = 320
    if len(text) <= max_chars:
        return text
    shortened = text[: max_chars - 1].rsplit(" ", 1)[0].strip()
    suffix = "…" if lang == Language.PL else "…"
    return f"{shortened}{suffix}"


async def process_product_input(
    product_input: str,
    language: Language,
) -> tuple[str, str]:
    """Process product input - extract from URL if needed.
    
    Returns:
        (product_description, status_message)
    """
    product_input = normalize_url(product_input)
    
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

    if is_url(product_input):
        # Extract product from URL using Gemini
        from app.services.llm_client import get_llm_client
        
        try:
            client = get_llm_client()
            if hasattr(client, 'extract_product_from_url'):
                status = "🔗 Extracting product from URL..." if language == Language.EN else "🔗 Pobieranie produktu z URL..."
                product_description = await client.extract_product_from_url(product_input, language)
                if product_description and _looks_like_full_extraction(product_description):
                    product_description = _attach_source(product_description, product_input)
                    return product_description, f"✅ Extracted from: {product_input}"
                warn = (
                    "❌ URL extraction incomplete; please paste full product description manually."
                    if language == Language.EN
                    else "❌ Ekstrakcja z URL niepełna — wklej pełny opis produktu ręcznie."
                )
                return "", warn
            warn = (
                "❌ URL extraction unavailable; please paste full product description manually."
                if language == Language.EN
                else "❌ Ekstrakcja z URL niedostępna — wklej pełny opis produktu ręcznie."
            )
            return "", warn
        except Exception as e:
            # Fallback to using URL as-is
            warn = (
                f"❌ Could not fetch URL: {e}"
                if language == Language.EN
                else f"❌ Nie udało się pobrać URL: {e}"
            )
            return "", warn
    
    return product_input, ""


async def _preview_extract_from_input_async(
    product_input: str,
    lang_code: str,
    last_url: str,
):
    """Extract product details from URL with a visible in-flight status."""
    lang = get_lang(lang_code)
    normalized = normalize_url(product_input)
    if not is_url(normalized):
        msg = "❌ To nie wygląda jak URL." if lang == Language.PL else "❌ This doesn't look like a URL."
        yield msg, "", "", "", last_url or ""
        return
    if normalized == (last_url or ""):
        yield gr.update(), gr.update(), gr.update(), gr.update(), last_url
        return

    logging.getLogger(__name__).info("Manual URL extraction start: %s", normalized)
    waiting = "⏳ Pobieranie danych z URL..." if lang == Language.PL else "⏳ Fetching data from URL..."
    yield waiting, "", "", "", last_url or ""

    try:
        description, status = await process_product_input(normalized, lang)
    except Exception as exc:
        logging.getLogger(__name__).exception("Manual URL extraction failed")
        msg = (
            f"❌ Nie udało się pobrać URL: {exc}"
            if lang == Language.PL
            else f"❌ Could not fetch URL: {exc}"
        )
        yield msg, "", "", "", normalized
        return
    if not description:
        logging.getLogger(__name__).warning("Manual URL extraction failed: %s", status)
        yield status or "", "", "", "", normalized
        return
    preview = _shorten_extracted(description, lang)
    global _last_extracted_full, _last_extracted_preview, _last_extracted_url
    _last_extracted_full = description
    _last_extracted_preview = preview
    _last_extracted_url = normalized
    if lang == Language.PL:
        preview = f"**Wyciągnięte dane:** {preview}"
        full = f"**Pełny opis:** {description}"
        done = "✅ Pobrano dane z URL."
    else:
        preview = f"**Extracted data:** {preview}"
        full = f"**Full description:** {description}"
        done = "✅ Extracted data from URL."
    yield done, preview, full, description, normalized
    return


def update_extract_button_label(lang_code: str):
    lang = get_lang(lang_code)
    return gr.update(value=get_label(lang, "extract_url"))


# === Main Simulation Tab ===


async def run_simulation_async(
    lang_code: str,
    product_description: str,
    extracted_raw: str,
    last_extracted_url: str,
    n_agents: int,
    age_min: int,
    age_max: int,
    gender: Optional[str],
    income_level: Optional[str],
    education_level: Optional[str],
    location_type: Optional[str],
    region: Optional[str],
    enable_web_search: bool = True,
    temperature: float = 1.0,
    project_id: str | None = None,
    progress=gr.Progress(),
):
    """Run SSR simulation and return results."""
    lang = get_lang(lang_code)
    
    if not product_description.strip():
        err = get_label(lang, "error_no_product")
        return (
            None,
            err,
            "",
            err,
            gr.update(),
            "",
            "",
            "",
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )

    progress(0, desc="Initializing..." if lang == Language.EN else "Inicjalizacja...")
    url_status_msg = ""
    extract_status = ""
    extracted_preview = ""
    extracted_full = ""
    extracted_url = ""
    input_product_raw = product_description.strip()

    try:
        # Process product input - extract from URL if needed
        if is_url(product_description.strip()):
            normalized_input = normalize_url(product_description.strip())
            if extracted_raw and normalized_input == (last_extracted_url or ""):
                product_description = extracted_raw
                extracted_full = extracted_raw
                extracted_preview = _shorten_extracted(extracted_raw, lang)
                extracted_url = normalized_input
                extract_status = (
                    "✅ Użyto wcześniej pobranych danych z URL."
                    if lang == Language.PL
                    else "✅ Reused previously extracted URL data."
                )
            else:
                progress(0.05, desc="🔗 Fetching product from URL..." if lang == Language.EN else "🔗 Pobieranie produktu z URL...")
                product_description, url_status = await process_product_input(product_description, lang)
                url_status_msg = url_status or ""
                extracted_full = product_description
                extracted_preview = _shorten_extracted(product_description, lang)
                extracted_url = normalized_input
                if not product_description:
                    return (
                        None,
                        "Could not extract product from URL",
                        "",
                        "❌",
                        gr.update(),
                        "",
                        "",
                        "",
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                    )
        income = normalize_income_value(income_level)
        location = normalize_location_value(location_type)
        region_value = normalize_region_value(region)
        education = normalize_education_value(education_level)
        
        # Build demographic profile
        profile = DemographicProfile(
            age_min=age_min,
            age_max=age_max,
            gender=normalize_gender_value(gender),
            income_level=income,
            location_type=location,
            region=region_value,
            education=education,
        )

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
        chart_fig = create_histogram_chart(dist, lang)

        summary = build_simulation_summary(result, lang)
        opinions = build_simulation_opinions(result, lang)

        progress(1.0, desc="Done!" if lang == Language.EN else "Gotowe!")

        # Store result for report generation and project saving
        global _last_simulation_result, _last_product_description, _last_product_input_raw
        global _last_simulation_inputs, _last_extracted_preview, _last_extracted_full, _last_extracted_url
        _last_simulation_result = result
        _last_product_description = product_description
        _last_product_input_raw = input_product_raw
        if extracted_full:
            _last_extracted_full = extracted_full
            _last_extracted_preview = extracted_preview
            _last_extracted_url = extracted_url or normalize_url(input_product_raw)
        sim_inputs = {
            "product_description": product_description,
            "product_input_raw": input_product_raw,
            "product_extracted_full": extracted_full,
            "product_extracted_preview": extracted_preview,
            "product_extracted_url": extracted_url,
            "n_agents": n_agents,
            "target_audience": profile.model_dump(),
            "enable_web_search": enable_web_search,
            "temperature": temperature,
        }
        _last_simulation_inputs = sim_inputs
        _last_extracted_preview = extracted_preview
        _last_extracted_full = extracted_full

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

        if extracted_full:
            if lang == Language.PL:
                extracted_full = f"**Pełny opis:** {extracted_full}"
            else:
                extracted_full = f"**Full description:** {extracted_full}"

        return (
            chart_fig,
            summary,
            opinions,
            status_msg,
            dirty_value,
            extract_status,
            extracted_preview,
            extracted_full,
            extracted_full or "",
            extracted_url or "",
            gr.update(value=input_product_raw),
            gr.update(value=input_product_raw),
            gr.update(value=input_product_raw),
        )

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
            "",
            "",
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )


def run_simulation(*args):
    """Wrapper to run async function."""
    return asyncio.run(run_simulation_async(*args))


def generate_report(lang_code: str, only_cited_sources: bool):
    """Generate HTML report preview from last simulation results."""
    global _last_simulation_result, _last_product_description
    global _last_report_analysis, _last_report_analysis_key
    lang = get_lang(lang_code)
    
    if _last_simulation_result is None:
        if lang == Language.EN:
            return "", "❌ No simulation results. Run a simulation first."
        else:
            return "", "❌ Brak wyników symulacji. Najpierw uruchom symulację."
    
    try:
        analysis_sections = None
        settings = get_settings()
        model_name = getattr(settings, "report_analysis_model", "gemini-3-pro-preview")
        analysis_key = f"{lang.value}:{_last_simulation_result.id}:{model_name}:v3"
        if _last_report_analysis_key == analysis_key:
            analysis_sections = _last_report_analysis
            if analysis_sections:
                narrative = (analysis_sections.get("narrative") or "").strip()
                if narrative.startswith("Analiza niedostępna") or narrative.startswith("Analysis unavailable"):
                    analysis_sections = None
        if not analysis_sections:
            try:
                analysis_sections = asyncio.run(
                    _generate_report_analysis_async(
                        _last_simulation_result,
                        _last_product_description,
                        lang,
                    )
                )
            except Exception as e:
                analysis_sections = None
                logging.getLogger(__name__).warning("Report analysis failed: %s", e)
            if analysis_sections:
                logging.getLogger(__name__).info("Report analysis status | narrative_len=%s | agent_len=%s | rec_len=%s",
                                               len((analysis_sections.get("narrative") or "")),
                                               len((analysis_sections.get("agent_summary") or "")),
                                               len((analysis_sections.get("recommendations") or "")))
                _last_report_analysis = analysis_sections
                _last_report_analysis_key = analysis_key

        # Generate HTML report content
        html_content = generate_html_report(
            result=_last_simulation_result,
            product_description=_last_product_description,
            lang=lang,
            include_only_cited_sources=bool(only_cited_sources),
            analysis_sections=analysis_sections,
        )
        
        # Store for export (raw HTML)
        global _last_report_html, _last_report_only_cited
        _last_report_html = html_content
        _last_report_only_cited = bool(only_cited_sources)
        
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
_last_report_only_cited = None


def _build_app_url(relative_path: str, request: gr.Request | None) -> str:
    logger = logging.getLogger(__name__)
    sign_token: str | None = None
    if request:
        try:
            params = getattr(request, "query_params", None)
            if params is not None:
                sign_token = params.get("__sign")
        except Exception:
            sign_token = None
    external_base_url = os.getenv("MARKET_WIZARD_EXTERNAL_BASE_URL")
    if external_base_url:
        logger.info(
            "Download URL override via MARKET_WIZARD_EXTERNAL_BASE_URL=%s path=%s",
            external_base_url,
            relative_path,
        )
        relative = relative_path.lstrip("/")
        base = urljoin(external_base_url.rstrip("/") + "/", relative)
        if sign_token:
            parsed = urlparse(base)
            query = dict(parse_qsl(parsed.query))
            query.setdefault("__sign", sign_token)
            base = urlunparse(parsed._replace(query=urlencode(query)))
            logger.info("Download URL appended __sign token (override)")
        return base
    if request:
        try:
            scope = getattr(request, "scope", {}) or {}
            root_path = scope.get("root_path")
            url = getattr(request, "url", None)
            logger.debug(
                "Download URL request_url=%s root_path=%s path=%s",
                url,
                root_path,
                relative_path,
            )
        except Exception:
            logger.debug("Download URL request_url unavailable, path=%s", relative_path)
        base_url = getattr(request, "base_url", None)
        if base_url:
            logger.debug("Download URL base_url=%s path=%s", base_url, relative_path)
            relative = relative_path.lstrip("/")
            base = urljoin(str(base_url), relative)
            if sign_token:
                parsed = urlparse(base)
                query = dict(parse_qsl(parsed.query))
                query.setdefault("__sign", sign_token)
                base = urlunparse(parsed._replace(query=urlencode(query)))
                logger.info("Download URL appended __sign token (base_url)")
            return base
        headers = dict(request.headers) if request.headers else {}
        proto = headers.get("x-forwarded-proto") or headers.get("x-scheme")
        host = headers.get("x-forwarded-host") or headers.get("host")
        if proto and host:
            logger.debug(
                "Download URL forwarded proto=%s host=%s path=%s",
                proto,
                host,
                relative_path,
            )
            relative = relative_path.lstrip("/")
            base = f"{proto}://{host}/{relative}"
            if sign_token:
                parsed = urlparse(base)
                query = dict(parse_qsl(parsed.query))
                query.setdefault("__sign", sign_token)
                base = urlunparse(parsed._replace(query=urlencode(query)))
                logger.info("Download URL appended __sign token (forwarded)")
            return base
        filtered_headers = {
            key: headers.get(key)
            for key in ("host", "x-forwarded-host", "x-forwarded-proto", "x-scheme")
            if headers.get(key) is not None
        }
        logger.warning(
            "Download URL fallback to relative path, headers=%s",
            filtered_headers,
        )
    base = relative_path if relative_path.startswith("/") else f"/{relative_path}"
    if sign_token:
        parsed = urlparse(base)
        query = dict(parse_qsl(parsed.query))
        query.setdefault("__sign", sign_token)
        base = urlunparse(parsed._replace(query=urlencode(query)))
        logger.info("Download URL appended __sign token (relative)")
    return base


def export_report(
    lang_code: str,
    export_format: str,
    only_cited_sources: bool,
    request: gr.Request | None = None,
):
    """Export report to HTML or PDF file."""
    global _last_report_html, _last_report_only_cited
    global _last_report_analysis, _last_report_analysis_key
    lang = get_lang(lang_code)
    
    if _last_simulation_result is None:
        if lang == Language.EN:
            return None, "❌ No simulation results. Run a simulation first."
        else:
            return None, "❌ Brak wyników symulacji. Najpierw uruchom symulację."

    if _last_report_html is None or _last_report_only_cited != bool(only_cited_sources):
        analysis_sections = None
        settings = get_settings()
        model_name = getattr(settings, "report_analysis_model", "gemini-3-pro-preview")
        analysis_key = f"{lang.value}:{_last_simulation_result.id}:{model_name}:v3"
        if _last_report_analysis_key == analysis_key:
            analysis_sections = _last_report_analysis
            if analysis_sections:
                narrative = (analysis_sections.get("narrative") or "").strip()
                if narrative.startswith("Analiza niedostępna") or narrative.startswith("Analysis unavailable"):
                    analysis_sections = None
        if not analysis_sections:
            try:
                analysis_sections = asyncio.run(
                    _generate_report_analysis_async(
                        _last_simulation_result,
                        _last_product_description,
                        lang,
                    )
                )
            except Exception as e:
                analysis_sections = None
                logging.getLogger(__name__).warning("Report analysis failed: %s", e)
            if analysis_sections:
                logging.getLogger(__name__).info("Report analysis status | narrative_len=%s | agent_len=%s | rec_len=%s",
                                               len((analysis_sections.get("narrative") or "")),
                                               len((analysis_sections.get("agent_summary") or "")),
                                               len((analysis_sections.get("recommendations") or "")))
                _last_report_analysis = analysis_sections
                _last_report_analysis_key = analysis_key
        html_content = generate_html_report(
            result=_last_simulation_result,
            product_description=_last_product_description,
            lang=lang,
            include_only_cited_sources=bool(only_cited_sources),
            analysis_sections=analysis_sections,
        )
        _last_report_html = html_content
        _last_report_only_cited = bool(only_cited_sources)
    
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # Use system temp directory directly - Gradio auto-caches files from tempfile.gettempdir()
        output_dir = Path(tempfile.gettempdir())
        
        if export_format == "PDF":
            # Export as PDF using Playwright if available, otherwise use weasyprint
            try:
                from playwright.sync_api import sync_playwright
                output_path = output_dir / f"ssr_report_{timestamp}.pdf"
                with sync_playwright() as p:
                    browser = p.chromium.launch()
                    page = browser.new_page(viewport={"width": 1200, "height": 1800})
                    page.set_content(_last_report_html, wait_until="networkidle")
                    page.emulate_media(media="screen")
                    page.pdf(
                        path=str(output_path),
                        format="A4",
                        print_background=True,
                        margin={"top": "10mm", "right": "10mm", "bottom": "12mm", "left": "10mm"},
                    )
                    browser.close()
            except Exception:
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
                        return str(output_path), "⚠️ PDF export requires Playwright or weasyprint. Saved as HTML - use browser Print to PDF."
                    else:
                        return str(output_path), "⚠️ Eksport PDF wymaga Playwright lub weasyprint. Zapisano HTML - użyj Drukuj do PDF w przeglądarce."
        else:
            # Export as HTML
            output_path = output_dir / f"ssr_report_{timestamp}.html"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(_last_report_html)
        
        # Log successful export for debugging
        logger = logging.getLogger(__name__)
        logger.info(f"Report exported to: {output_path} | exists={output_path.exists()}")
        
        # Return download URL as HTML link (bypasses Gradio file handling bug)
        cache_dir = str(gr_utils.get_cache_folder())
        cached_path = processing_utils.save_file_to_cache(output_path, cache_dir)
        logger.info("Export file cached | filename=%s cached_path=%s", output_path.name, cached_path)
        if lang == Language.EN:
            return cached_path, "✅ Ready to download"
        else:
            return cached_path, "✅ Gotowe do pobrania"

    
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
    age_min: int,
    age_max: int,
    gender: Optional[str],
    income_level: Optional[str],
    location_type: Optional[str],
    region: Optional[str],
    enable_web_search: bool = True,
    progress=gr.Progress(),
):
    """Run A/B test comparison."""
    lang = get_lang(lang_code)
    
    if not variant_a.strip() or not variant_b.strip():
        return get_label(lang, "error_no_variants"), "❌", gr.update()

    progress(0, desc="Running A/B test..." if lang == Language.EN else "Uruchamianie testu A/B...")

    try:
        target_audience = DemographicProfile(
            **normalize_target_audience(
                lang,
                age_min,
                age_max,
                gender or "",
                income_level or "",
                location_type or "",
                region or "",
            )
        )
        engine = ABTestEngine(language=lang)
        from uuid import uuid4
        result = await engine.run_ab_test(
            project_id=uuid4(),
            variant_a=variant_a,
            variant_b=variant_b,
            target_audience=target_audience,
            n_agents=n_agents,
            enable_web_search=enable_web_search,
        )

        progress(1.0)

        global _last_ab_test_result, _last_ab_test_inputs
        _last_ab_test_result = result
        _last_ab_test_inputs = {
            "variant_a": variant_a,
            "variant_b": variant_b,
            "n_agents": n_agents,
            "target_audience": target_audience.model_dump(),
            "enable_web_search": enable_web_search,
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
    age_min: int,
    age_max: int,
    gender: Optional[str],
    income_level: Optional[str],
    location_type: Optional[str],
    region: Optional[str],
    enable_web_search: bool = True,
    progress=gr.Progress(),
):
    """Run price sensitivity analysis."""
    lang = get_lang(lang_code)
    
    if not product_description.strip():
        return get_label(lang, "error_no_product"), None, "❌", gr.update()

    progress(0, desc="Analyzing price sensitivity..." if lang == Language.EN else "Analizowanie wrażliwości cenowej...")

    try:
        if is_url(product_description.strip()):
            normalized_input = normalize_url(product_description.strip())
            global _last_extracted_full, _last_extracted_url, _last_extracted_preview
            if _last_extracted_full and normalized_input == (_last_extracted_url or ""):
                product_description = _last_extracted_full
            else:
                product_description, _ = await process_product_input(product_description, lang)
                if not product_description:
                    return get_label(lang, "error_no_product"), None, "❌", gr.update()
                _last_extracted_full = product_description
                _last_extracted_preview = _shorten_extracted(product_description, lang)
                _last_extracted_url = normalized_input
        # Generate price points
        price_points = list(np.linspace(price_min, price_max, int(n_points)))
        target_audience = DemographicProfile(
            **normalize_target_audience(
                lang,
                age_min,
                age_max,
                gender or "",
                income_level or "",
                location_type or "",
                region or "",
            )
        )

        engine = PriceSensitivityEngine(language=lang)
        from uuid import uuid4
        result = await engine.analyze_price_sensitivity(
            project_id=uuid4(),
            base_product_description=product_description,
            price_points=price_points,
            target_audience=target_audience,
            n_agents=n_agents,
            enable_web_search=enable_web_search,
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
            "target_audience": target_audience.model_dump(),
            "enable_web_search": enable_web_search,
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
    age_min: int,
    age_max: int,
    gender: Optional[str],
    income_level: Optional[str],
    location_type: Optional[str],
    region: Optional[str],
    enable_web_search: bool = True,
    project_id: str | None = None,
    progress=gr.Progress(),
) -> tuple[str, str, str, object]:
    """Run virtual focus group discussion."""
    lang = get_lang(lang_code)
    
    if not product or not product.strip():
        error_msg = "❌ Enter product description" if lang == Language.EN else "❌ Wprowadź opis produktu"
        return "", "", error_msg, gr.update()

    progress(0, desc="Starting focus group..." if lang == Language.EN else "Uruchamianie grupy fokusowej...")
    
    try:
        # Process product input - extract from URL if needed
        if is_url(product.strip()):
            normalized_input = normalize_url(product.strip())
            global _last_extracted_full, _last_extracted_url, _last_extracted_preview
            if _last_extracted_full and normalized_input == (_last_extracted_url or ""):
                product = _last_extracted_full
            else:
                product, _ = await process_product_input(product, lang)
                if not product:
                    return "", "", "❌ Could not extract product from URL", gr.update()
                _last_extracted_full = product
                _last_extracted_preview = _shorten_extracted(product, lang)
                _last_extracted_url = normalized_input
        
        engine = FocusGroupEngine(language=lang)
        
        target_audience = DemographicProfile(
            **normalize_target_audience(
                lang,
                age_min,
                age_max,
                gender or "",
                income_level or "",
                location_type or "",
                region or "",
            )
        )

        # Run focus group
        result = await engine.run_focus_group(
            product_description=product.strip(),
            n_participants=int(n_participants),
            n_rounds=int(n_rounds),
            target_audience=target_audience,
            enable_web_search=enable_web_search,
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
                "target_audience": target_audience.model_dump(),
                "enable_web_search": enable_web_search,
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
        "target_audience": normalize_target_audience(
            get_lang(args[0]),
            int(args[4]) if len(args) > 4 else 25,
            int(args[5]) if len(args) > 5 else 45,
            args[6] if len(args) > 6 else "Wszystkie",
            args[7] if len(args) > 7 else "Wszystkie",
            args[8] if len(args) > 8 else "Wszystkie",
            args[9] if len(args) > 9 else "Wszystkie regiony",
        ),
        "enable_web_search": args[10] if len(args) > 10 else True,
    }
    return result


def export_focus_group(
    lang_code: str,
    export_format: str,
    request: gr.Request | None = None,
) -> tuple[str | None, str]:
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
    # Use system temp directory directly - Gradio auto-caches files from tempfile.gettempdir()
    output_dir = Path(tempfile.gettempdir())
    
    if export_format == "PDF":
        try:
            from weasyprint import HTML
            filename = f"focus_group_{timestamp}.pdf"
            filepath = output_dir / filename
            HTML(string=html_content).write_pdf(str(filepath))
        except ImportError:
            return None, "❌ PDF export requires weasyprint. Install with: pip install weasyprint"
    else:
        filename = f"focus_group_{timestamp}.html"
        filepath = output_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)
    
    # Return download URL as HTML link (bypasses Gradio file handling bug)
    cache_dir = str(gr_utils.get_cache_folder())
    cached_path = processing_utils.save_file_to_cache(filepath, cache_dir)
    logging.getLogger(__name__).info(
        "Export file cached | filename=%s cached_path=%s",
        filepath.name,
        cached_path,
    )
    if lang == Language.EN:
        return cached_path, "✅ Ready to download"
    else:
        return cached_path, "✅ Gotowe do pobrania"


# === Project Management ===


def _empty_histogram_chart(lang: Language) -> go.Figure:
    return create_histogram_chart(
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
    region: str,
    n_agents: int,
    enable_web_search: bool,
    temperature: float,
    variant_a_input: str,
    variant_b_input: str,
    ab_n_agents: int,
    ab_enable_web_search: bool,
    price_product: str,
    price_min: float,
    price_max: float,
    price_points: int,
    price_n_agents: int,
    price_enable_web_search: bool,
    fg_product: str,
    fg_participants: int,
    fg_rounds: int,
    fg_enable_web_search: bool,
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

    product_input_raw = input_product or _last_product_input_raw or existing.get("product_input_raw", "")
    product_extracted_full = _last_extracted_full or existing.get("product_extracted_full", "")
    product_extracted_preview = _last_extracted_preview or _shorten_extracted(product_extracted_full, lang)
    product_extracted_url = _last_extracted_url or existing.get("product_extracted_url", "")

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
            region,
        )

    research = existing.get("research", {})

    if _last_simulation_result is not None or _last_simulation_inputs is not None:
        sim_payload = research.get("simulation", {})
        if _last_simulation_inputs is not None:
            sim_payload["inputs"] = _last_simulation_inputs
        else:
            sim_payload["inputs"] = {
                "product_description": input_product or product_description,
                "product_input_raw": product_input_raw,
                "product_extracted_full": product_extracted_full,
                "product_extracted_preview": product_extracted_preview,
                "product_extracted_url": product_extracted_url,
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
                "product_input_raw": product_input_raw,
                "product_extracted_full": product_extracted_full,
                "product_extracted_preview": product_extracted_preview,
                "product_extracted_url": product_extracted_url,
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
                "target_audience": target_audience,
                "enable_web_search": ab_enable_web_search,
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
                "target_audience": target_audience,
                "enable_web_search": price_enable_web_search,
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
                "target_audience": target_audience,
                "enable_web_search": fg_enable_web_search,
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
    if product_extracted_full:
        project["product_extracted_full"] = product_extracted_full
        if product_extracted_preview:
            project["product_extracted_preview"] = product_extracted_preview
    if product_extracted_url:
        project["product_extracted_url"] = product_extracted_url

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
    import logging

    lang = get_lang(lang_code)
    logger = logging.getLogger(__name__)
    def _no_change_updates(count: int) -> list:
        return [gr.update() for _ in range(count)]
    total_outputs = 51
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
            _empty_histogram_chart(lang),
            "",
            "",
            "",
            _empty_price_df(lang),
            "",
            "",
            msg,
            gr.update(visible=False),
            gr.update(visible=False),
            *_no_change_updates(total_outputs - 12),
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
            _empty_histogram_chart(lang),
            "",
            "",
            "",
            _empty_price_df(lang),
            "",
            "",
            msg,
            gr.update(visible=False),
            gr.update(visible=False),
            *_no_change_updates(total_outputs - 12),
        )

    try:
        info = build_project_info(project, lang)

        research = project.get("research", {})

        sim_summary = ""
        sim_opinions = ""
        sim_chart = _empty_histogram_chart(lang)
        sim_payload = research.get("simulation", {})
        if sim_payload.get("result"):
            sim_result = SimulationResult.model_validate(sim_payload["result"])
            sim_summary = build_simulation_summary(sim_result, lang)
            sim_opinions = build_simulation_opinions(sim_result, lang)
            sim_chart = create_histogram_chart(sim_result.aggregate_distribution.model_dump(), lang)
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
            sim_inputs.get("product_input_raw")
            or project.get("product_input_raw", "")
            or sim_inputs.get("product_description")
            or project.get("product_description", "")
        )

        target_audience = sim_inputs.get("target_audience") or project.get("target_audience")
        age_min_value, age_max_value, gender_value, income_value, location_value, region_value = (
            target_audience_to_ui(lang, target_audience)
        )

        n_agents_value = int(sim_inputs.get("n_agents") or 20)
        n_agents_value = max(5, min(100, n_agents_value))
        enable_web_search_value = bool(sim_inputs.get("enable_web_search", True))
        temperature_value = float(sim_inputs.get("temperature") or 1.0)

        variant_a_value = ab_inputs.get("variant_a") or product_value
        variant_b_value = ab_inputs.get("variant_b") or ""
        ab_n_agents_value = int(ab_inputs.get("n_agents") or 30)
        ab_n_agents_value = max(10, min(100, ab_n_agents_value))
        ab_enable_web_search_value = bool(ab_inputs.get("enable_web_search", True))

        price_product_value = (
            price_inputs.get("base_product_description")
            or product_value
        )
        price_min_value = float(price_inputs.get("price_min") or 19.99)
        price_max_value = float(price_inputs.get("price_max") or 59.99)
        price_points_value = int(price_inputs.get("price_points") or 5)
        price_points_value = max(3, min(7, price_points_value))
        price_n_agents_value = int(price_inputs.get("n_agents") or 20)
        price_n_agents_value = max(10, min(100, price_n_agents_value))
        price_enable_web_search_value = bool(price_inputs.get("enable_web_search", True))

        fg_product_value = fg_inputs.get("product_description") or product_value
        fg_participants_value = int(fg_inputs.get("n_participants") or 6)
        fg_participants_value = max(4, min(8, fg_participants_value))
        fg_rounds_value = int(fg_inputs.get("n_rounds") or 3)
        fg_rounds_value = max(2, min(4, fg_rounds_value))
        fg_enable_web_search_value = bool(fg_inputs.get("enable_web_search", True))

        extracted_full = sim_inputs.get("product_extracted_full") or project.get("product_extracted_full", "")
        extracted_preview = sim_inputs.get("product_extracted_preview") or _shorten_extracted(extracted_full, lang)
        extracted_url_value = (
            sim_inputs.get("product_extracted_url")
            or project.get("product_extracted_url", "")
        )
        if not extracted_url_value and is_url(product_value):
            extracted_url_value = normalize_url(product_value)

        global _last_extracted_full, _last_extracted_preview, _last_extracted_url
        if extracted_full:
            _last_extracted_full = extracted_full
            _last_extracted_preview = _shorten_extracted(extracted_full, lang)
        if extracted_url_value:
            _last_extracted_url = extracted_url_value

        if extracted_preview:
            if lang == Language.PL:
                extracted_preview = f"**Wyciągnięte dane:** {extracted_preview}"
            else:
                extracted_preview = f"**Extracted data:** {extracted_preview}"

        if extracted_full:
            if lang == Language.PL:
                extracted_full = f"**Pełny opis:** {extracted_full}"
            else:
                extracted_full = f"**Full description:** {extracted_full}"

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
        gr.update(value=extracted_preview or ""),
        gr.update(value=extracted_full or ""),
        gr.update(value=extracted_full or ""),
        gr.update(value=extracted_url_value or ""),
        gr.update(value=age_min_value),
        gr.update(value=age_max_value),
        gr.update(value=gender_value),
        gr.update(value=income_value),
        gr.update(value=location_value),
        gr.update(value=region_value),
        gr.update(value=n_agents_value),
        gr.update(value=enable_web_search_value),
        gr.update(value=temperature_value),
        gr.update(value=variant_a_value),
        gr.update(value=variant_b_value),
        gr.update(value=ab_n_agents_value),
        gr.update(value=ab_enable_web_search_value),
        gr.update(value=price_product_value),
        gr.update(value=price_min_value),
        gr.update(value=price_max_value),
        gr.update(value=price_points_value),
        gr.update(value=price_n_agents_value),
        gr.update(value=price_enable_web_search_value),
        gr.update(value=fg_product_value),
        gr.update(value=fg_participants_value),
        gr.update(value=fg_rounds_value),
        gr.update(value=fg_enable_web_search_value),
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
    except Exception as e:
        logger.exception("Load project failed: %s", e)
        msg = "❌ Błąd wczytywania projektu." if lang == Language.PL else "❌ Failed to load project."
        updates = [gr.update() for _ in range(total_outputs)]
        updates[status_index] = msg
        updates[12] = gr.update(value="")
        updates[13] = gr.update(value="")
        updates[14] = gr.update(value="")
        updates[15] = gr.update(value="")
        updates[16] = gr.update(value="")
        updates[17] = gr.update(value="")
        return tuple(updates)


def maybe_autoload_project(
    lang_code: str,
    project_id: str | None,
    allow_discard: bool,
    auto_load: bool,
    project_dirty: bool,
):
    """Auto-load project on selection if enabled."""
    if not auto_load:
        return tuple(gr.update() for _ in range(51))
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
        gr.HTML(
            "<style>"
            ".modebar,.modebar-container,.plotly-logo,.modebar-btn--logo{display:none !important;}"
            ".pi-plot{background:#ffffff !important;}"
            ".pi-plot .plotly-graph-div{width:100% !important;}"
            ".pi-plot .plot-container{width:100% !important;}"
            ".pi-plot .svg-container{width:100% !important;}"
            "</style>"
        )
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
        last_url_state = gr.State(value="")
        extracted_raw_state = gr.State(value="")
        
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
                            elem_id="product_input_main",
                        )
                        extract_url_btn = gr.Button(
                            value=get_label(Language.PL, "extract_url"),
                        )
                        extract_status = gr.Markdown("")
                        extracted_product_preview = gr.Markdown("")
                        with gr.Accordion("📄 Extracted details / Wyciągnięte dane", open=False):
                            extracted_product_full = gr.Markdown("")

                        gr.Markdown("### Target Audience / Grupa docelowa")
                        with gr.Row():
                            age_min = gr.Slider(18, 80, value=25, step=1, label="Age min / Wiek min")
                            age_max = gr.Slider(18, 80, value=45, step=1, label="Age max / Wiek max")

                        with gr.Row():
                            gender = gr.Dropdown(
                                choices=["Wszystkie", "M", "K"],
                                value="Wszystkie",
                                label="Gender / Płeć",
                            )
                            income = gr.Dropdown(
                                choices=INCOME_CHOICES[Language.PL],
                                value="Wszystkie",
                                label="Income / Dochód (netto)",
                            )
                            education = gr.Dropdown(
                                choices=EDUCATION_CHOICES[Language.PL],
                                value="Wszystkie",
                                label="Education / Wykształcenie",
                            )
                        with gr.Row():
                            location = gr.Dropdown(
                                choices=LOCATION_CHOICES[Language.PL],
                                value="Wszystkie",
                                label="Location / Lokalizacja (GUS 2024)",
                            )
                            region = gr.Dropdown(
                                choices=REGION_CHOICES[Language.PL],
                                value="Wszystkie regiony",
                                label="Region / Województwo (GUS 2024)",
                            )

                        n_agents = gr.Slider(5, 100, value=20, step=5, label="Number of agents / Liczba agentów")

                        with gr.Accordion("🔍 Web Search (RAG)", open=False):
                            enable_web_search = gr.Checkbox(
                                label="Enable Google Search / Włącz wyszukiwanie",
                                value=True,
                                info="Agents will search the web for market info / Agenci wyszukają informacje o rynku"
                            )

                        with gr.Accordion("⚙️ Advanced Settings / Zaawansowane", open=False):
                            temperature = gr.Slider(
                                minimum=0.01, 
                                maximum=2.0, 
                                value=float(settings.ssr_temperature), 
                                step=0.01, 
                                label="SSR Temperature / Temperatura",
                                info="Lower = more decisive, Higher = smoother (paper 1.0) / Niższa = bardziej zdecydowane"
                            )

                        run_btn = gr.Button("🚀 Run Simulation / Uruchom", variant="primary")
                        status = gr.Markdown("")

                    with gr.Column(scale=1):
                        summary_output = gr.Markdown(label="Summary / Podsumowanie")
                        chart_output = gr.Plot(
                            label="",
                            show_label=False,
                            elem_classes="pi-plot",
                        )

                with gr.Accordion("📝 Sample Agent Opinions / Przykładowe opinie", open=False):
                    opinions_output = gr.Markdown()

                # Report generation section
                gr.Markdown("### 📄 Report / Raport")
                with gr.Row():
                    report_btn = gr.Button("👁️ Generate Preview / Generuj podgląd", variant="secondary")
                    report_status = gr.Markdown("")
                only_cited_sources = gr.Checkbox(
                    value=False,
                    label="Only cited sources / Tylko cytowane źródła",
                    info="Show only sources referenced with [n] citations in agent responses",
                )
                
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
                export_file = gr.File(label="📥 Download / Pobierz")

                report_btn.click(
                    fn=generate_report,
                    inputs=[language_select, only_cited_sources],
                    outputs=[report_preview, report_status],
                )

                export_btn.click(
                    fn=export_report,
                    inputs=[language_select, export_format, only_cited_sources],
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
                with gr.Accordion("🔍 Web Search (RAG)", open=False):
                    ab_enable_web_search = gr.Checkbox(
                        label="Enable Google Search / Włącz wyszukiwanie",
                        value=True,
                        info="Agents will search the web for market info / Agenci wyszukają informacje o rynku",
                    )
                ab_run_btn = gr.Button("🔬 Run A/B Test / Uruchom test", variant="primary")
                ab_status = gr.Markdown("")
                ab_result = gr.Markdown()

                ab_run_btn.click(
                    fn=run_ab_test,
                    inputs=[
                        language_select,
                        variant_a_input,
                        variant_b_input,
                        ab_n_agents,
                        age_min,
                        age_max,
                        gender,
                        income,
                        location,
                        region,
                        ab_enable_web_search,
                    ],
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

                price_n_agents = gr.Slider(10, 100, value=50, step=5, label="Agents per price / Agentów na cenę")
                with gr.Accordion("🔍 Web Search (RAG)", open=False):
                    price_enable_web_search = gr.Checkbox(
                        label="Enable Google Search / Włącz wyszukiwanie",
                        value=True,
                        info="Agents will search the web for market info / Agenci wyszukają informacje o rynku",
                    )
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
                    inputs=[
                        language_select,
                        price_product,
                        price_min,
                        price_max,
                        price_points,
                        price_n_agents,
                        age_min,
                        age_max,
                        gender,
                        income,
                        location,
                        region,
                        price_enable_web_search,
                    ],
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
                        fg_enable_web_search = gr.Checkbox(
                            label="Enable Google Search / Włącz wyszukiwanie",
                            value=True,
                            info="Agents will search the web for market info / Agenci wyszukają informacje o rynku",
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
                    inputs=[
                        language_select,
                        fg_product,
                        fg_participants,
                        fg_rounds,
                        age_min,
                        age_max,
                        gender,
                        income,
                        location,
                        region,
                        fg_enable_web_search,
                        project_state,
                    ],
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
                    fg_export_file = gr.File(label="Download")
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
                    saved_sim_chart = gr.Plot(
                        label="",
                        show_label=False,
                        elem_classes="pi-plot",
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
                        extracted_product_preview,
                        extracted_product_full,
                        extracted_raw_state,
                        last_url_state,
                        age_min,
                        age_max,
                        gender,
                        income,
                        location,
                        region,
                        n_agents,
                        enable_web_search,
                        temperature,
                        variant_a_input,
                        variant_b_input,
                        ab_n_agents,
                        ab_enable_web_search,
                        price_product,
                        price_min,
                        price_max,
                        price_points,
                        price_n_agents,
                        price_enable_web_search,
                        fg_product,
                        fg_participants,
                        fg_rounds,
                        fg_enable_web_search,
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
                        region,
                        n_agents,
                        enable_web_search,
                        temperature,
                        variant_a_input,
                        variant_b_input,
                        ab_n_agents,
                        ab_enable_web_search,
                        price_product,
                        price_min,
                        price_max,
                        price_points,
                        price_n_agents,
                        price_enable_web_search,
                        fg_product,
                        fg_participants,
                        fg_rounds,
                        fg_enable_web_search,
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
                        extracted_product_preview,
                        extracted_product_full,
                        extracted_raw_state,
                        last_url_state,
                        age_min,
                        age_max,
                        gender,
                        income,
                        location,
                        region,
                        n_agents,
                        enable_web_search,
                        temperature,
                        variant_a_input,
                        variant_b_input,
                        ab_n_agents,
                        ab_enable_web_search,
                        price_product,
                        price_min,
                        price_max,
                        price_points,
                        price_n_agents,
                        price_enable_web_search,
                        fg_product,
                        fg_participants,
                        fg_rounds,
                        fg_enable_web_search,
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
                        region,
                        n_agents,
                        enable_web_search,
                        temperature,
                        variant_a_input,
                        variant_b_input,
                        ab_n_agents,
                        ab_enable_web_search,
                        price_product,
                        price_min,
                        price_max,
                        price_points,
                        price_n_agents,
                        price_enable_web_search,
                        fg_product,
                        fg_participants,
                        fg_rounds,
                        fg_enable_web_search,
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
                        extracted_product_preview,
                        extracted_product_full,
                        extracted_raw_state,
                        last_url_state,
                        age_min,
                        age_max,
                        gender,
                        income,
                        location,
                        region,
                        n_agents,
                        enable_web_search,
                        temperature,
                        variant_a_input,
                        variant_b_input,
                        ab_n_agents,
                        ab_enable_web_search,
                        price_product,
                        price_min,
                        price_max,
                        price_points,
                        price_n_agents,
                        price_enable_web_search,
                        fg_product,
                        fg_participants,
                        fg_rounds,
                        fg_enable_web_search,
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
                        extracted_product_preview,
                        extracted_product_full,
                        extracted_raw_state,
                        last_url_state,
                        age_min,
                        age_max,
                        gender,
                        income,
                        location,
                        region,
                        n_agents,
                        enable_web_search,
                        temperature,
                        variant_a_input,
                        variant_b_input,
                        ab_n_agents,
                        ab_enable_web_search,
                        price_product,
                        price_min,
                        price_max,
                        price_points,
                        price_n_agents,
                        price_enable_web_search,
                        fg_product,
                        fg_participants,
                        fg_rounds,
                        fg_enable_web_search,
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
            education,
            location,
            region,
            n_agents,
            enable_web_search,
            temperature,
            variant_a_input,
            variant_b_input,
            ab_n_agents,
            ab_enable_web_search,
            price_product,
            price_min,
            price_max,
            price_points,
            price_n_agents,
            price_enable_web_search,
            fg_product,
            fg_participants,
            fg_rounds,
            fg_enable_web_search,
        ]
        mark_dirty_inputs = dirty_inputs + [suppress_dirty]
        for comp in dirty_inputs:
            comp.change(
                fn=mark_dirty,
                inputs=mark_dirty_inputs,
                outputs=[project_dirty],
            )

        extract_url_btn.click(
            fn=_preview_extract_from_input_async,
            inputs=[product_input, language_select, last_url_state],
            outputs=[
                extract_status,
                extracted_product_preview,
                extracted_product_full,
                extracted_raw_state,
                last_url_state,
            ],
        )
        language_select.change(
            fn=update_extract_button_label,
            inputs=[language_select],
            outputs=[extract_url_btn],
        )
        language_select.change(
            fn=update_demographic_dropdowns,
            inputs=[language_select, gender, income, education, location, region],
            outputs=[gender, income, education, location, region],
        )

        gr.Markdown(
            """
            ---
            *Market Wizard v0.5.0 | Based on arXiv:2510.08338 | PL/EN*
            """
        )

        # Wire actions after all components are defined.
        run_btn.click(
            fn=run_simulation,
            inputs=[
                language_select,
                product_input,
                extracted_raw_state,
                last_url_state,
                n_agents,
                age_min,
                age_max,
                gender,
                income,
                education,
                location,
                region,
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
                extract_status,
                extracted_product_preview,
                extracted_product_full,
                extracted_raw_state,
                last_url_state,
                variant_a_input,
                price_product,
                fg_product,
            ],
        )

    return demo


if __name__ == "__main__":
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("System temp directory: %s", tempfile.gettempdir())

    demo = create_interface()

    app = FastAPI()

    # Add download endpoint that bypasses Gradio's file handling
    @app.get("/download-report/{filename}")
    async def download_report(filename: str):
        """Direct file download endpoint - bypasses Gradio 6.x file handling bug."""
        filepath = Path(tempfile.gettempdir()) / filename
        if not filepath.exists():
            logger.error("Download failed - file not found: %s", filepath)
            raise HTTPException(status_code=404, detail=f"File not found: {filename}")

        # Determine media type based on extension
        media_type = "application/pdf" if filename.endswith(".pdf") else "text/html"
        logger.info("Serving download: %s", filepath)
        return FileResponse(
            path=str(filepath),
            filename=filename,
            media_type=media_type,
        )

    logger.info("Download endpoint registered at /download-report/{filename}")

    # Mount Gradio app on the FastAPI app
    app = gr.mount_gradio_app(app, demo, path="/")
    logger.info("Gradio app mounted on FastAPI at path '/'")

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")
