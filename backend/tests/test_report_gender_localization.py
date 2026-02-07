"""Tests for localized gender labels in generated reports."""

from uuid import uuid4

from app.i18n import Language
from app.models import AgentResponse, LikertDistribution, Persona, SimulationResult
from app.services.report_generator import generate_html_report


def _sample_result(gender: str = "F") -> SimulationResult:
    dist = LikertDistribution(
        scale_1=0.0,
        scale_2=0.0,
        scale_3=0.2,
        scale_4=0.5,
        scale_5=0.3,
    )
    persona = Persona(
        name="TestUser_1",
        age=34,
        gender=gender,
        income=6500,
        location="Warszawa",
        location_type="large_city",
        occupation="analityk",
    )
    response = AgentResponse(
        persona=persona,
        text_response="Test opinion",
        likert_pmf=dist,
        likert_score=4.1,
        sources=[],
    )
    return SimulationResult(
        project_id=uuid4(),
        n_agents=1,
        aggregate_distribution=dist,
        mean_purchase_intent=4.1,
        agent_responses=[response],
        web_sources=[],
    )


def test_report_pl_localizes_female_label_to_k():
    result = _sample_result(gender="F")
    html = generate_html_report(
        result=result,
        product_description="Produkt testowy",
        lang=Language.PL,
    )
    assert "👤 K" in html
    assert "👤 F" not in html


def test_report_en_keeps_female_label_as_f():
    result = _sample_result(gender="F")
    html = generate_html_report(
        result=result,
        product_description="Test product",
        lang=Language.EN,
    )
    assert "👤 F" in html
