"""
Report Generator - Creates comprehensive HTML reports for SSR simulations.

Generates reports with:
- Executive summary
- All agent responses with SSR scores
- Distribution charts
- Demographic breakdowns
- Methodology description
"""

import base64
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from app.models import SimulationResult, Persona, AgentResponse
from app.i18n import Language, get_label


def _localized_gender_label(gender: str, lang: Language) -> str:
    """Map internal gender code to language-specific report label."""
    code = (gender or "").strip().upper()
    if code == "F":
        return "K" if lang == Language.PL else "F"
    if code == "M":
        return "M"
    return gender or "-"


def generate_distribution_chart(result: SimulationResult, lang: Language) -> str:
    """Generate base64-encoded distribution chart."""
    dist = result.aggregate_distribution
    
    if lang == Language.PL:
        labels = ['1-Nie', '2-Raczej nie', '3-Neutralny', '4-Raczej tak', '5-Tak']
        title = 'Rozkład intencji zakupu'
        ylabel = 'Procent respondentów'
    else:
        labels = ['1-No', '2-Probably not', '3-Neutral', '4-Probably yes', '5-Yes']
        title = 'Purchase Intent Distribution'
        ylabel = 'Percent of respondents'
    
    values = [
        dist.scale_1 * 100,
        dist.scale_2 * 100,
        dist.scale_3 * 100,
        dist.scale_4 * 100,
        dist.scale_5 * 100,
    ]
    
    # Professional financial color palette (blue gradient)
    colors = ['#1e3a5f', '#2d5a87', '#4a7c9b', '#6b9db8', '#8ebfd4']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values, color=colors, edgecolor='white', linewidth=2)
    
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 100)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}%"))
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f'{val:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    # Convert to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close(fig)
    
    return img_base64


def generate_age_distribution_chart(responses: List[AgentResponse], lang: Language) -> str:
    """Generate age distribution chart of agents."""
    ages = [r.persona.age for r in responses]
    
    if lang == Language.PL:
        title = 'Rozkład wiekowy respondentów'
        xlabel = 'Wiek'
        ylabel = 'Liczba respondentów'
    else:
        title = 'Age Distribution of Respondents'
        xlabel = 'Age'
        ylabel = 'Number of respondents'
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(ages, bins=range(18, 85, 5), color='#2d5a87', edgecolor='white', linewidth=1)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close(fig)
    
    return img_base64


def generate_income_vs_intent_chart(responses: List[AgentResponse], lang: Language) -> str:
    """Generate scatter plot of income vs purchase intent."""
    incomes = [r.persona.income for r in responses]
    intents = [r.likert_score for r in responses]
    
    if lang == Language.PL:
        title = 'Dochód vs Intencja zakupu'
        xlabel = 'Dochód (PLN/mies.)'
        ylabel = 'Intencja zakupu (1-5)'
    else:
        title = 'Income vs Purchase Intent'
        xlabel = 'Income ($/month)'
        ylabel = 'Purchase Intent (1-5)'
    
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(incomes, intents, c=intents, cmap='Blues', 
                         s=80, alpha=0.7, edgecolors='white', linewidth=1)
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0.5, 5.5)
    
    # Add trend line
    z = np.polyfit(incomes, intents, 1)
    p = np.poly1d(z)
    ax.plot(sorted(incomes), p(sorted(incomes)), "--", color='#1e3a5f', alpha=0.8, linewidth=2)
    
    plt.colorbar(scatter, label='Intent Score')
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close(fig)
    
    return img_base64


def generate_html_report(
    result: SimulationResult,
    product_description: str,
    lang: Language = Language.PL,
    include_only_cited_sources: bool = False,
    analysis_sections: dict[str, str] | None = None,
) -> str:
    """Generate comprehensive HTML report."""
    
    # Generate charts
    dist_chart = generate_distribution_chart(result, lang)
    age_chart = generate_age_distribution_chart(result.agent_responses, lang)
    income_chart = generate_income_vs_intent_chart(result.agent_responses, lang)
    
    # Calculate statistics
    scores = [r.likert_score for r in result.agent_responses]
    incomes = [r.persona.income for r in result.agent_responses]
    ages = [r.persona.age for r in result.agent_responses]
    
    stats = {
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'min_score': min(scores),
        'max_score': max(scores),
        'mean_income': np.mean(incomes),
        'mean_age': np.mean(ages),
        'gender_m': sum(1 for r in result.agent_responses if r.persona.gender == 'M'),
        'gender_f': sum(1 for r in result.agent_responses if r.persona.gender == 'F'),
    }
    
    # Sort responses by score
    sorted_responses = sorted(result.agent_responses, key=lambda r: r.likert_score, reverse=True)
    
    # Generate HTML
    sources = _select_report_sources(result, include_only_cited_sources)
    total_sources_count = len(result.web_sources or [])

    if lang == Language.PL:
        html = _generate_html_pl(
            result,
            product_description,
            stats,
            sorted_responses,
            dist_chart,
            age_chart,
            income_chart,
            sources,
            total_sources_count,
            include_only_cited_sources,
            analysis_sections,
        )
    else:
        html = _generate_html_en(
            result,
            product_description,
            stats,
            sorted_responses,
            dist_chart,
            age_chart,
            income_chart,
            sources,
            total_sources_count,
            include_only_cited_sources,
            analysis_sections,
        )
    
    return html


def _generate_html_pl(
    result,
    product,
    stats,
    responses,
    dist_chart,
    age_chart,
    income_chart,
    sources,
    total_sources_count,
    only_cited,
    analysis_sections,
):
    """Generate Polish HTML report."""
    
    # Build responses HTML
    responses_html = ""
    for i, r in enumerate(responses, 1):
        p = r.persona
        score_color = _get_score_color(r.likert_score)
        gender_label = _localized_gender_label(p.gender, Language.PL)
        
        # Build optional demographic fields
        education_html = f'<span style="color: #000000; font-weight: 500;">🎓 {p.education}</span>' if p.education else ''
        marital_html = f'<span style="color: #000000; font-weight: 500;">💍 {p.marital_status}</span>' if p.marital_status else ''
        children_html = ''
        if p.has_children is not None:
            children_label = 'ma dzieci' if p.has_children else 'bez dzieci'
            children_html = f'<span style="color: #000000; font-weight: 500;">👶 {children_label}</span>'
        
        responses_html += f"""
        <div style="background: #ffffff; border-radius: 0.75rem; padding: 1rem; margin-bottom: 1rem; border-left: 4px solid #1e3a5f; border: 1px solid #e5e7eb;">
            <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 0.5rem;">
                <span style="color: #000000; font-weight: bold;">#{i}</span>
                <span style="font-weight: bold; flex-grow: 1; color: #000000;">{p.name}</span>
                <span style="padding: 0.25rem 0.75rem; border-radius: 1rem; color: white; font-weight: bold; background-color: {score_color}">{r.likert_score:.2f}</span>
            </div>
            <div style="display: flex; flex-wrap: wrap; gap: 1rem; margin-bottom: 0.75rem; font-size: 0.875rem;">
                <span style="color: #000000; font-weight: 500;">🎂 {p.age} lat</span>
                <span style="color: #000000; font-weight: 500;">👤 {gender_label}</span>
                <span style="color: #000000; font-weight: 500;">📍 {p.location}</span>
                <span style="color: #000000; font-weight: 500;">💰 {p.income:,} PLN</span>
                {f'<span style="color: #000000; font-weight: 500;">💼 {p.occupation}</span>' if p.occupation else ''}
                {education_html}
                {marital_html}
                {children_html}
            </div>
            <div style="background: #f8fafc; padding: 1rem; border-radius: 0.5rem; border-left: 3px solid #1e3a5f; color: #000000; margin: 0;">{_format_response_html(r.text_response)}</div>
        </div>
        """
    
    dist = result.aggregate_distribution

    analysis_html = _generate_analysis_sections_pl(analysis_sections)

    return f"""<!DOCTYPE html>
<html lang="pl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Raport SSR - Market Wizard</title>
    <style>
        :root {{
            --primary: #1e3a5f;
            --primary-light: #2d5a87;
            --accent: #4a7c9b;
            --bg: #f8fafc;
            --card: #ffffff;
            --text: #1e293b;
            --muted: #64748b;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        html, body {{
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: #ffffff !important;
            color: #1f2937 !important;
            line-height: 1.6;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 2rem; background: #ffffff; }}
        .header {{
            background: linear-gradient(135deg, #1e3a5f, #2d5a87);
            color: white;
            padding: 3rem 2rem;
            text-align: center;
            border-radius: 0.5rem;
            margin-bottom: 2rem;
            border-bottom: 4px solid #4a7c9b;
        }}
        .header h1 {{ font-size: 2.5rem; margin-bottom: 0.5rem; }}
        .header p {{ opacity: 0.9; }}
        .card {{
            background: var(--card);
            border-radius: 1rem;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .card h2 {{
            font-size: 1.25rem;
            margin-bottom: 1rem;
            color: var(--primary);
            border-bottom: 2px solid var(--primary);
            padding-bottom: 0.5rem;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
        }}
        .stat-item {{
            text-align: center;
            padding: 1.5rem;
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 0.75rem;
        }}
        .stat-value {{ font-size: 2rem; font-weight: bold; color: #000000 !important; }}
        .stat-label {{ color: #000000 !important; font-size: 0.875rem; font-weight: 600; }}
        .chart-container {{ text-align: center; margin: 1rem 0; }}
        .chart-container img {{ max-width: 100%; height: auto; border-radius: 0.5rem; }}
        .avoid-break {{ break-inside: avoid; page-break-inside: avoid; }}
        .avoid-break {{ break-inside: avoid; page-break-inside: avoid; }}
        .distribution-bar {{
            display: flex;
            height: 2rem;
            border-radius: 0.5rem;
            overflow: hidden;
            margin: 1rem 0;
        }}
        .distribution-bar > div {{ display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 0.75rem; }}
        .response-card {{
            background: #f8fafc;
            border-radius: 0.75rem;
            padding: 1rem;
            margin-bottom: 1rem;
            border-left: 4px solid var(--primary);
        }}
        .response-header {{
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 0.5rem;
        }}
        .response-number {{ color: var(--muted); font-weight: bold; }}
        .persona-name {{ font-weight: bold; flex-grow: 1; }}
        .score-badge {{
            padding: 0.25rem 0.75rem;
            border-radius: 1rem;
            color: white;
            font-weight: bold;
        }}
        .persona-details {{
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
            margin-bottom: 0.75rem;
            font-size: 0.875rem;
            color: #374151;
            font-weight: 500;
        }}
        .opinion {{
            background: white;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 3px solid var(--primary);
            font-style: italic;
        }}
        .product-box {{
            background: #f1f5f9;
            border: 1px solid #e2e8f0;
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }}
        .footer {{
            text-align: center;
            padding: 2rem;
            color: var(--muted);
            font-size: 0.875rem;
        }}
        @media print {{
            .container {{ max-width: 100%; padding: 1rem; }}
            .response-card {{ page-break-inside: avoid; }}
            body {{
                -webkit-print-color-adjust: exact;
                print-color-adjust: exact;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔮 Raport SSR</h1>
            <p>Market Wizard - Symulacja intencji zakupowych</p>
            <p style="margin-top: 1rem; font-size: 0.875rem;">
                Wygenerowano: {datetime.now().strftime('%Y-%m-%d %H:%M')}
            </p>
        </div>
        
        <div style="background: #ffffff; border-radius: 1rem; padding: 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
            <h2 style="color: #000000; font-size: 1.25rem; margin-bottom: 1rem; border-bottom: 2px solid #1e3a5f; padding-bottom: 0.5rem;">📦 Analizowany produkt</h2>
            <div style="background: #ffffff; border: 1px solid #e5e7eb; padding: 1.5rem; border-radius: 0.5rem; color: #000000;">
                {product}
            </div>
        </div>
        
        <div class="card" style="background: #ffffff; border-radius: 1rem; padding: 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
            <h2 style="color: #000000; font-size: 1.25rem; margin-bottom: 1rem; border-bottom: 2px solid #1e3a5f; padding-bottom: 0.5rem;">📊 Podsumowanie wyników</h2>
            <div class="stats-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
                <div style="text-align: center; padding: 1.5rem; background: #ffffff; border: 2px solid #1e3a5f; border-radius: 0.75rem;">
                    <div style="font-size: 2rem; font-weight: bold; color: #000000;">{stats['mean_score']:.2f}</div>
                    <div style="color: #000000; font-size: 0.875rem; font-weight: 600;">Średnia intencja zakupu (1-5)</div>
                </div>
                <div style="text-align: center; padding: 1.5rem; background: #ffffff; border: 2px solid #1e3a5f; border-radius: 0.75rem;">
                    <div style="font-size: 2rem; font-weight: bold; color: #000000;">{result.n_agents}</div>
                    <div style="color: #000000; font-size: 0.875rem; font-weight: 600;">Liczba respondentów</div>
                </div>
                <div style="text-align: center; padding: 1.5rem; background: #ffffff; border: 2px solid #1e3a5f; border-radius: 0.75rem;">
                    <div style="font-size: 2rem; font-weight: bold; color: #000000;">{stats['std_score']:.2f}</div>
                    <div style="color: #000000; font-size: 0.875rem; font-weight: 600;">Odchylenie standardowe</div>
                </div>
                <div style="text-align: center; padding: 1.5rem; background: #ffffff; border: 2px solid #1e3a5f; border-radius: 0.75rem;">
                    <div style="font-size: 2rem; font-weight: bold; color: #000000;">{stats['min_score']:.2f} - {stats['max_score']:.2f}</div>
                    <div style="color: #000000; font-size: 0.875rem; font-weight: 600;">Zakres ocen</div>
                </div>
            </div>
        </div>
        
        <div class="avoid-break" style="background: #ffffff; border-radius: 1rem; padding: 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
            <h2 style="color: #000000; font-size: 1.25rem; margin-bottom: 1rem; border-bottom: 2px solid #1e3a5f; padding-bottom: 0.5rem;">📈 Rozkład intencji zakupu</h2>
            <div style="display: flex; height: 2rem; border-radius: 0.5rem; overflow: hidden; margin: 1rem 0;">
                <div style="width: {dist.scale_1*100}%; background: #1e3a5f; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 0.75rem;">1</div>
                <div style="width: {dist.scale_2*100}%; background: #2d5a87; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 0.75rem;">2</div>
                <div style="width: {dist.scale_3*100}%; background: #4a7c9b; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 0.75rem;">3</div>
                <div style="width: {dist.scale_4*100}%; background: #6b9db8; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 0.75rem;">4</div>
                <div style="width: {dist.scale_5*100}%; background: #8ebfd4; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 0.75rem;">5</div>
            </div>
            <div class="chart-container avoid-break">
                <img src="data:image/png;base64,{dist_chart}" alt="Distribution Chart" style="max-width: 100%; height: auto; border-radius: 0.5rem;">
            </div>
        </div>
        
        <div class="avoid-break" style="background: #ffffff; border-radius: 1rem; padding: 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
            <h2 style="color: #000000; font-size: 1.25rem; margin-bottom: 1rem; border-bottom: 2px solid #1e3a5f; padding-bottom: 0.5rem;">👥 Profil demograficzny respondentów</h2>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
                <div style="text-align: center; padding: 1.5rem; background: #ffffff; border: 2px solid #1e3a5f; border-radius: 0.75rem;">
                    <div style="font-size: 2rem; font-weight: bold; color: #000000;">{stats['mean_age']:.1f}</div>
                    <div style="color: #000000; font-size: 0.875rem; font-weight: 600;">Średni wiek</div>
                </div>
                <div style="text-align: center; padding: 1.5rem; background: #ffffff; border: 2px solid #1e3a5f; border-radius: 0.75rem;">
                    <div style="font-size: 2rem; font-weight: bold; color: #000000;">{stats['mean_income']:,.0f} PLN</div>
                    <div style="color: #000000; font-size: 0.875rem; font-weight: 600;">Średni dochód</div>
                </div>
                <div style="text-align: center; padding: 1.5rem; background: #ffffff; border: 2px solid #1e3a5f; border-radius: 0.75rem;">
                    <div style="font-size: 2rem; font-weight: bold; color: #000000;">{stats['gender_m']} M / {stats['gender_f']} K</div>
                    <div style="color: #000000; font-size: 0.875rem; font-weight: 600;">Rozkład płci</div>
                </div>
            </div>
            <div class="chart-container avoid-break">
                <img src="data:image/png;base64,{age_chart}" alt="Age Distribution" style="max-width: 100%; height: auto; border-radius: 0.5rem;">
            </div>
        </div>
        
        <div class="avoid-break" style="background: #ffffff; border-radius: 1rem; padding: 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
            <h2 style="color: #000000; font-size: 1.25rem; margin-bottom: 1rem; border-bottom: 2px solid #1e3a5f; padding-bottom: 0.5rem;">💰 Dochód vs Intencja zakupu</h2>
            <div class="chart-container avoid-break">
                <img src="data:image/png;base64,{income_chart}" alt="Income vs Intent" style="max-width: 100%; height: auto; border-radius: 0.5rem;">
            </div>
        </div>

        {analysis_html}
        
        <div style="background: #ffffff; border-radius: 1rem; padding: 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
            <h2 style="color: #000000; font-size: 1.25rem; margin-bottom: 1rem; border-bottom: 2px solid #1e3a5f; padding-bottom: 0.5rem;">📝 Wszystkie odpowiedzi ({len(responses)} respondentów)</h2>
            {responses_html}
        </div>
        
        {_generate_sources_section_pl(sources, total_sources_count, only_cited) if sources else ''}
        
        <div style="text-align: center; padding: 2rem; color: #000000; font-size: 0.875rem;">
            <p>Raport wygenerowany przez Market Wizard</p>
            <p>Metodologia: SSR (Semantic Similarity Rating) - arXiv:2510.08338</p>
        </div>
    </div>
</body>
</html>"""


def _generate_sources_section_pl(
    sources: list[str],
    total_sources_count: int,
    only_cited: bool,
) -> str:
    """Generate sources section for Polish report."""
    if not sources:
        return ""

    from urllib.parse import urlparse

    domains = {
        urlparse(url).netloc.lower()
        for url in sources
        if urlparse(url).netloc
    }
    domain_count = len(domains)
    
    sources_html = ""
    for i, url in enumerate(sources[:20], 1):  # Limit to 20 sources
        # Clean URL for display
        display_url = url[:80] + "..." if len(url) > 80 else url
        sources_html += f'''
        <div style="padding: 0.5rem; border-bottom: 1px solid #e5e7eb;">
            <a href="{url}" target="_blank" style="color: #1e3a5f; text-decoration: none;">
                {i}. {display_url}
            </a>
        </div>'''
    
    cited_info = (
        f", {len(sources)} cytowane z {total_sources_count}"
        if only_cited and total_sources_count
        else ""
    )
    return f'''
    <div style="background: #ffffff; border-radius: 1rem; padding: 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
        <h2 style="color: #000000; font-size: 1.25rem; margin-bottom: 1rem; border-bottom: 2px solid #1e3a5f; padding-bottom: 0.5rem;">
            🔍 Źródła wykorzystane do analizy rynku ({len(sources)} źródeł, {domain_count} domen{cited_info})
        </h2>
        <p style="color: #6b7280; font-size: 0.875rem; margin-bottom: 1rem;">
            Agenci wyszukali poniższe źródła aby porównać produkt z konkurencją:
        </p>
        {sources_html}
    </div>'''


def _generate_analysis_sections_pl(analysis_sections: dict[str, str] | None) -> str:
    if not analysis_sections:
        return ""

    narrative = _format_response_html(analysis_sections.get("narrative", ""))
    agent_summary = _format_response_html(analysis_sections.get("agent_summary", ""))
    recommendations = _format_response_html(analysis_sections.get("recommendations", ""))

    if not any([narrative, agent_summary, recommendations]):
        return ""

    return f'''
        <div style="background: #ffffff; border-radius: 1rem; padding: 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
            <h2 style="color: #000000; font-size: 1.25rem; margin-bottom: 1rem; border-bottom: 2px solid #1e3a5f; padding-bottom: 0.5rem;">🧠 Analiza narracyjna</h2>
            <div style="background: #f8fafc; padding: 1rem; border-radius: 0.5rem; border-left: 3px solid #1e3a5f; color: #000000;">{narrative}</div>
        </div>
        <div style="background: #ffffff; border-radius: 1rem; padding: 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
            <h2 style="color: #000000; font-size: 1.25rem; margin-bottom: 1rem; border-bottom: 2px solid #1e3a5f; padding-bottom: 0.5rem;">🗣️ Podsumowanie odpowiedzi agentów</h2>
            <div style="background: #f8fafc; padding: 1rem; border-radius: 0.5rem; border-left: 3px solid #1e3a5f; color: #000000;">{agent_summary}</div>
        </div>
        <div style="background: #ffffff; border-radius: 1rem; padding: 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
            <h2 style="color: #000000; font-size: 1.25rem; margin-bottom: 1rem; border-bottom: 2px solid #1e3a5f; padding-bottom: 0.5rem;">📣 Rekomendacje sprzedażowo-marketingowe</h2>
            <div style="background: #f8fafc; padding: 1rem; border-radius: 0.5rem; border-left: 3px solid #1e3a5f; color: #000000;">{recommendations}</div>
        </div>
    '''


def _select_report_sources(result: SimulationResult, only_cited: bool) -> list[str]:
    if not only_cited:
        return list(result.web_sources or [])

    import re

    cited_urls: list[str] = []
    seen: set[str] = set()
    citation_re = re.compile(r"\[(\d+)\]")

    for resp in result.agent_responses:
        indices = [int(m.group(1)) for m in citation_re.finditer(resp.text_response or "")]
        if not indices:
            continue
        for idx in indices:
            pos = idx - 1
            if pos < 0 or pos >= len(resp.sources):
                continue
            url = resp.sources[pos]
            if url and url not in seen:
                seen.add(url)
                cited_urls.append(url)

    return cited_urls


def _format_response_html(text: str) -> str:
    import html
    import re

    raw = (text or "").strip()
    if not raw:
        return ""

    escaped = html.escape(raw)
    lines = [line.strip() for line in escaped.split("\n")]
    out: list[str] = []
    in_list = False

    def close_list():
        nonlocal in_list
        if in_list:
            out.append("</ul>")
            in_list = False

    bold_re = re.compile(r"\*\*(.+?)\*\*")

    for line in lines:
        if not line:
            close_list()
            continue
        # Headings like ### or ##
        if line.startswith("### "):
            close_list()
            out.append(f"<div style=\"font-weight: 700; margin: 0.5rem 0;\">{line[4:]}</div>")
            continue
        if line.startswith("## "):
            close_list()
            out.append(f"<div style=\"font-weight: 700; margin: 0.5rem 0;\">{line[3:]}</div>")
            continue
        # Bullets
        if line.startswith("* ") or line.startswith("- "):
            if not in_list:
                out.append("<ul style=\"margin: 0.5rem 0 0.5rem 1.25rem;\">")
                in_list = True
            item = line[2:].strip()
            item = bold_re.sub(r"<strong>\1</strong>", item)
            out.append(f"<li>{item}</li>")
            continue
        close_list()
        paragraph = bold_re.sub(r"<strong>\1</strong>", line)
        out.append(f"<p style=\"margin: 0.5rem 0;\">{paragraph}</p>")

    close_list()
    return "".join(out)


def _generate_analysis_sections_en(analysis_sections: dict[str, str] | None) -> str:
    if not analysis_sections:
        return ""

    narrative = _format_response_html(analysis_sections.get("narrative", ""))
    agent_summary = _format_response_html(analysis_sections.get("agent_summary", ""))
    recommendations = _format_response_html(analysis_sections.get("recommendations", ""))

    if not any([narrative, agent_summary, recommendations]):
        return ""

    return f'''
        <div class="card">
            <h2>🧠 Narrative Analysis</h2>
            <div style="background: #f8fafc; padding: 1rem; border-radius: 0.5rem; border-left: 3px solid #1e3a5f; color: #000000;">{narrative}</div>
        </div>
        <div class="card">
            <h2>🗣️ Agent Response Summary</h2>
            <div style="background: #f8fafc; padding: 1rem; border-radius: 0.5rem; border-left: 3px solid #1e3a5f; color: #000000;">{agent_summary}</div>
        </div>
        <div class="card">
            <h2>📣 Sales & Marketing Recommendations</h2>
            <div style="background: #f8fafc; padding: 1rem; border-radius: 0.5rem; border-left: 3px solid #1e3a5f; color: #000000;">{recommendations}</div>
        </div>
    '''


def _generate_html_en(
    result,
    product,
    stats,
    responses,
    dist_chart,
    age_chart,
    income_chart,
    sources,
    total_sources_count,
    only_cited,
    analysis_sections,
):
    """Generate English HTML report."""
    
    # Build responses HTML
    responses_html = ""
    for i, r in enumerate(responses, 1):
        p = r.persona
        score_color = _get_score_color(r.likert_score)
        gender_label = _localized_gender_label(p.gender, Language.EN)
        
        # Build optional demographic fields with English labels
        education_html = f'<span style="color: #000000; font-weight: 500;">🎓 {p.education}</span>' if p.education else ''
        
        # Translate marital status
        marital_map = {
            "kawaler/panna": "single",
            "małżeństwo": "married", 
            "rozwiedziony": "divorced",
            "wdowiec/wdowa": "widowed",
        }
        marital_label = marital_map.get(p.marital_status, p.marital_status) if p.marital_status else None
        marital_html = f'<span style="color: #000000; font-weight: 500;">💍 {marital_label}</span>' if marital_label else ''
        
        children_html = ''
        if p.has_children is not None:
            children_label = 'has children' if p.has_children else 'no children'
            children_html = f'<span style="color: #000000; font-weight: 500;">👶 {children_label}</span>'
        
        responses_html += f"""
        <div style="background: #ffffff; border-radius: 0.75rem; padding: 1rem; margin-bottom: 1rem; border-left: 4px solid #1e3a5f; border: 1px solid #e5e7eb;">
            <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 0.5rem;">
                <span style="color: #000000; font-weight: bold;">#{i}</span>
                <span style="font-weight: bold; flex-grow: 1; color: #000000;">{p.name}</span>
                <span style="padding: 0.25rem 0.75rem; border-radius: 1rem; color: white; font-weight: bold; background-color: {score_color}">{r.likert_score:.2f}</span>
            </div>
            <div style="display: flex; flex-wrap: wrap; gap: 1rem; margin-bottom: 0.75rem; font-size: 0.875rem;">
                <span style="color: #000000; font-weight: 500;">🎂 {p.age} y.o.</span>
                <span style="color: #000000; font-weight: 500;">👤 {gender_label}</span>
                <span style="color: #000000; font-weight: 500;">📍 {p.location}</span>
                <span style="color: #000000; font-weight: 500;">💰 ${p.income:,}</span>
                {f'<span style="color: #000000; font-weight: 500;">💼 {p.occupation}</span>' if p.occupation else ''}
                {education_html}
                {marital_html}
                {children_html}
            </div>
            <div style="background: #f8fafc; padding: 1rem; border-radius: 0.5rem; border-left: 3px solid #1e3a5f; color: #000000; margin: 0;">{_format_response_html(r.text_response)}</div>
        </div>
        """
    
    dist = result.aggregate_distribution

    analysis_html = _generate_analysis_sections_en(analysis_sections)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SSR Report - Market Wizard</title>
    <style>
        :root {{
            --primary: #1e3a5f;
            --primary-light: #2d5a87;
            --accent: #4a7c9b;
            --bg: #f8fafc;
            --card: #ffffff;
            --text: #1e293b;
            --muted: #64748b;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        html, body {{
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: #ffffff !important;
            color: #1f2937 !important;
            line-height: 1.6;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 2rem; background: #ffffff; }}
        .header {{
            background: linear-gradient(135deg, #1e3a5f, #2d5a87);
            color: white;
            padding: 3rem 2rem;
            text-align: center;
            border-radius: 0.5rem;
            margin-bottom: 2rem;
            border-bottom: 4px solid #4a7c9b;
        }}
        .header h1 {{ font-size: 2.5rem; margin-bottom: 0.5rem; }}
        .header p {{ opacity: 0.9; }}
        .card {{
            background: var(--card);
            border-radius: 1rem;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .card h2 {{
            font-size: 1.25rem;
            margin-bottom: 1rem;
            color: var(--primary);
            border-bottom: 2px solid var(--primary);
            padding-bottom: 0.5rem;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
        }}
        .stat-item {{
            text-align: center;
            padding: 1.5rem;
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 0.75rem;
        }}
        .stat-value {{ font-size: 2rem; font-weight: bold; color: #000000 !important; }}
        .stat-label {{ color: #000000 !important; font-size: 0.875rem; font-weight: 600; }}
        .chart-container {{ text-align: center; margin: 1rem 0; }}
        .chart-container img {{ max-width: 100%; height: auto; border-radius: 0.5rem; }}
        .response-card {{
            background: #f8fafc;
            border-radius: 0.75rem;
            padding: 1rem;
            margin-bottom: 1rem;
            border-left: 4px solid var(--primary);
        }}
        .response-header {{
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 0.5rem;
        }}
        .response-number {{ color: var(--muted); font-weight: bold; }}
        .persona-name {{ font-weight: bold; flex-grow: 1; }}
        .score-badge {{
            padding: 0.25rem 0.75rem;
            border-radius: 1rem;
            color: white;
            font-weight: bold;
        }}
        .persona-details {{
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
            margin-bottom: 0.75rem;
            font-size: 0.875rem;
            color: #374151;
            font-weight: 500;
        }}
        .opinion {{
            background: white;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 3px solid var(--primary);
            font-style: italic;
        }}
        .product-box {{
            background: #f1f5f9;
            border: 1px solid #e2e8f0;
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }}
        .footer {{
            text-align: center;
            padding: 2rem;
            color: var(--muted);
            font-size: 0.875rem;
        }}
        @media print {{
            .container {{ max-width: 100%; padding: 1rem; }}
            .response-card {{ page-break-inside: avoid; }}
            body {{
                -webkit-print-color-adjust: exact;
                print-color-adjust: exact;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔮 SSR Report</h1>
            <p>Market Wizard - Purchase Intent Simulation</p>
            <p style="margin-top: 1rem; font-size: 0.875rem;">
                Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
            </p>
        </div>
        
        <div class="card">
            <h2>📦 Analyzed Product</h2>
            <div class="product-box">
                {product}
            </div>
        </div>
        
        <div class="card">
            <h2>📊 Results Summary</h2>
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-value">{stats['mean_score']:.2f}</div>
                    <div class="stat-label">Mean Purchase Intent (1-5)</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{result.n_agents}</div>
                    <div class="stat-label">Number of Respondents</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{stats['std_score']:.2f}</div>
                    <div class="stat-label">Standard Deviation</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{stats['min_score']:.2f} - {stats['max_score']:.2f}</div>
                    <div class="stat-label">Score Range</div>
                </div>
            </div>
        </div>
        
        <div class="card avoid-break">
            <h2>📈 Purchase Intent Distribution</h2>
            <div class="chart-container">
                <img src="data:image/png;base64,{dist_chart}" alt="Distribution Chart">
            </div>
        </div>
        
        <div class="card avoid-break">
            <h2>👥 Respondent Demographics</h2>
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-value">{stats['mean_age']:.1f}</div>
                    <div class="stat-label">Average Age</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">${stats['mean_income']:,.0f}</div>
                    <div class="stat-label">Average Income</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{stats['gender_m']} M / {stats['gender_f']} F</div>
                    <div class="stat-label">Gender Split</div>
                </div>
            </div>
            <div class="chart-container avoid-break">
                <img src="data:image/png;base64,{age_chart}" alt="Age Distribution">
            </div>
        </div>
        
        <div class="card avoid-break">
            <h2>💰 Income vs Purchase Intent</h2>
            <div class="chart-container avoid-break">
                <img src="data:image/png;base64,{income_chart}" alt="Income vs Intent">
            </div>
        </div>

        {analysis_html}
        
        <div class="card">
            <h2>📝 All Responses ({len(responses)} respondents)</h2>
            {responses_html}
        </div>
        
        {_generate_sources_section_en(sources, total_sources_count, only_cited) if sources else ''}
        
        <div class="footer">
            <p>Report generated by Market Wizard</p>
            <p>Methodology: SSR (Semantic Similarity Rating) - arXiv:2510.08338</p>
        </div>
    </div>
</body>
</html>"""


def _generate_sources_section_en(
    sources: list[str],
    total_sources_count: int,
    only_cited: bool,
) -> str:
    """Generate sources section for English report."""
    if not sources:
        return ""

    from urllib.parse import urlparse

    domains = {
        urlparse(url).netloc.lower()
        for url in sources
        if urlparse(url).netloc
    }
    domain_count = len(domains)
    
    sources_html = ""
    for i, url in enumerate(sources[:20], 1):
        display_url = url[:80] + "..." if len(url) > 80 else url
        sources_html += f'''
        <div style="padding: 0.5rem; border-bottom: 1px solid #e5e7eb;">
            <a href="{url}" target="_blank" style="color: #1e3a5f; text-decoration: none;">
                {i}. {display_url}
            </a>
        </div>'''
    
    cited_info = (
        f", {len(sources)} cited of {total_sources_count}"
        if only_cited and total_sources_count
        else ""
    )
    return f'''
    <div class="card">
        <h2>🔍 Web Sources Used for Market Analysis ({len(sources)} sources, {domain_count} domains{cited_info})</h2>
        <p style="color: #6b7280; font-size: 0.875rem; margin-bottom: 1rem;">
            Agents searched these sources to compare product with competition:
        </p>
        {sources_html}
    </div>'''


def _get_score_color(score: float) -> str:
    """Get color for score badge - professional blue gradient."""
    if score >= 4.0:
        return "#8ebfd4"  # light blue
    elif score >= 3.5:
        return "#6b9db8"  # medium-light blue
    elif score >= 3.0:
        return "#4a7c9b"  # medium blue
    elif score >= 2.5:
        return "#2d5a87"  # medium-dark blue
    else:
        return "#1e3a5f"  # dark blue


def save_report(
    result: SimulationResult,
    product_description: str,
    output_path: str | Path,
    lang: Language = Language.PL,
) -> Path:
    """Generate and save HTML report to file."""
    html = generate_html_report(result, product_description, lang)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    return output_path
