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

# Now import from backend app package
from app.models import DemographicProfile
from app.services import SimulationEngine, ABTestEngine, PriceSensitivityEngine
from app.services.report_generator import generate_html_report, save_report
from app.i18n import Language, get_label


# Store last simulation result for report generation
_last_simulation_result = None
_last_product_description = None


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
    progress=gr.Progress(),
):
    """Run SSR simulation and return results."""
    lang = get_lang(lang_code)
    
    if not product_description.strip():
        err = get_label(lang, "error_no_product")
        return None, err, "", err

    progress(0, desc="Initializing SSR engine..." if lang == Language.EN else "Inicjalizacja silnika SSR...")

    try:
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

        progress(0.1, desc="Generating personas..." if lang == Language.EN else "Generowanie person...")

        # Run simulation with language parameter
        engine = SimulationEngine(language=lang)
        from uuid import uuid4
        result = await engine.run_simulation(
            project_id=uuid4(),
            product_description=product_description,
            target_audience=profile,
            n_agents=n_agents,
        )

        progress(0.9, desc="Processing results..." if lang == Language.EN else "Przetwarzanie wyników...")

        # Prepare histogram
        dist = result.aggregate_distribution.model_dump()
        chart_df = create_histogram_data(dist, lang)

        # Format summary based on language
        if lang == Language.PL:
            summary = (
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
        else:
            summary = (
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

        # Format opinions (top 5)
        opinions = "\n\n".join(
            format_opinion(r, lang) for r in result.agent_responses[:5]
        )

        progress(1.0, desc="Done!" if lang == Language.EN else "Gotowe!")

        # Store result for report generation
        global _last_simulation_result, _last_product_description
        _last_simulation_result = result
        _last_product_description = product_description

        return chart_df, summary, opinions, get_label(lang, "success")

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"❌ Error: {str(e)}" if lang == Language.EN else f"❌ Błąd: {str(e)}", "", f"❌ {str(e)}"


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

        # Format results based on language
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
            ), "✅"
        else:
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
            ), "✅"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ Error: {str(e)}" if lang == Language.EN else f"❌ Błąd: {str(e)}", "❌"


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

        # Format demand curve based on language
        demand = result["demand_curve"]
        currency = "$" if lang == Language.EN else "PLN"
        
        if lang == Language.PL:
            curve_text = "### Krzywa popytu\n\n| Cena (PLN) | Intencja zakupu |\n|------------|----------------|\n"
        else:
            curve_text = "### Demand Curve\n\n| Price ($) | Purchase Intent |\n|-----------|----------------|\n"
            
        for price in sorted(demand.keys()):
            pi = demand[price]["mean_purchase_intent"]
            curve_text += f"| {price:.2f} | {pi:.2f}/5 |\n"

        # Add optimal price
        if lang == Language.PL:
            curve_text += f"\n\n**Optymalna cena:** {result['optimal_price']:.2f} PLN"
        else:
            curve_text += f"\n\n**Optimal price:** ${result['optimal_price']:.2f}"

        # Add elasticities
        if result["elasticities"]:
            if lang == Language.PL:
                curve_text += "\n\n### Elastyczność cenowa\n"
            else:
                curve_text += "\n\n### Price Elasticity\n"
            for e in result["elasticities"]:
                curve_text += f"- {e['price_range']}: {e['elasticity']:.2f}\n"

        # Create chart data as DataFrame
        if lang == Language.PL:
            chart_df = pd.DataFrame({
                "Cena": sorted(demand.keys()),
                "Intencja": [demand[p]["mean_purchase_intent"] for p in sorted(demand.keys())],
            })
        else:
            chart_df = pd.DataFrame({
                "Price": sorted(demand.keys()),
                "Intent": [demand[p]["mean_purchase_intent"] for p in sorted(demand.keys())],
            })

        return curve_text, chart_df, "✅"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ Error: {str(e)}" if lang == Language.EN else f"❌ Błąd: {str(e)}", None, "❌"


def run_price_analysis(*args):
    """Wrapper for async price analysis."""
    return asyncio.run(run_price_analysis_async(*args))


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
                    inputs=[language_select, product_input, n_agents, age_min, age_max, gender, income, location],
                    outputs=[chart_output, summary_output, opinions_output, status],
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
                    outputs=[ab_result, ab_status],
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
                    outputs=[price_result, price_chart, price_status],
                )

            # === Tab 4: About ===
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
