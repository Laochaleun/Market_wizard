"""Simulations router for running SSR-based market research."""

from typing import Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.models import DemographicProfile, SimulationResult
from app.services import ABTestEngine, PriceSensitivityEngine, SimulationEngine

router = APIRouter()


# === Request Models ===


class RunSimulationRequest(BaseModel):
    """Request for running a basic simulation."""

    product_description: str = Field(..., min_length=10)
    target_audience: Optional[DemographicProfile] = None
    n_agents: int = Field(50, ge=1, le=1000)
    llm_model: Optional[str] = None


class ABTestRequest(BaseModel):
    """Request for running an A/B test."""

    variant_a_description: str = Field(..., min_length=10)
    variant_b_description: str = Field(..., min_length=10)
    target_audience: Optional[DemographicProfile] = None
    n_agents: int = Field(50, ge=1, le=500)


class PriceSensitivityRequest(BaseModel):
    """Request for price sensitivity analysis."""

    product_description: str = Field(..., min_length=10)
    price_points: list[float] = Field(..., min_length=2, max_length=10)
    target_audience: Optional[DemographicProfile] = None
    n_agents: int = Field(30, ge=1, le=200)


# === Endpoints ===


@router.post("/simulations/run", response_model=SimulationResult)
async def run_simulation(request: RunSimulationRequest):
    """
    Run an SSR simulation to estimate purchase intent.
    
    This endpoint:
    1. Generates synthetic consumer personas matching the target audience
    2. Has each persona evaluate the product using an LLM
    3. Maps text responses to Likert ratings via SSR methodology
    4. Returns aggregate statistics and individual responses
    """
    try:
        engine = SimulationEngine(model_override=request.llm_model)
        result = await engine.run_simulation(
            project_id=uuid4(),
            product_description=request.product_description,
            target_audience=request.target_audience,
            n_agents=request.n_agents,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/simulations/ab-test")
async def run_ab_test(request: ABTestRequest):
    """
    Run an A/B test comparing two product variants.
    
    Returns comparative statistics including purchase intent
    for each variant and the estimated lift.
    """
    try:
        engine = ABTestEngine()
        result = await engine.run_ab_test(
            project_id=uuid4(),
            variant_a=request.variant_a_description,
            variant_b=request.variant_b_description,
            target_audience=request.target_audience,
            n_agents=request.n_agents,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/simulations/price-sensitivity")
async def analyze_price_sensitivity(request: PriceSensitivityRequest):
    """
    Analyze purchase intent at different price points.
    
    Returns a demand curve showing how purchase intent
    changes with price, along with price elasticity estimates.
    """
    try:
        engine = PriceSensitivityEngine()
        result = await engine.analyze_price_sensitivity(
            project_id=uuid4(),
            base_product_description=request.product_description,
            price_points=request.price_points,
            target_audience=request.target_audience,
            n_agents=request.n_agents,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/simulations/demo")
async def demo_simulation():
    """
    Run a quick demo simulation with a sample product.
    
    Uses a small number of agents for fast results.
    """
    try:
        engine = SimulationEngine()
        result = await engine.run_simulation(
            project_id=uuid4(),
            product_description=(
                "Pasta do zębów z węglem aktywnym i miętą. "
                "Naturalnie wybiela zęby i odświeża oddech. "
                "Opakowanie 75ml, cena 24.99 PLN."
            ),
            target_audience=DemographicProfile(
                age_min=25,
                age_max=45,
                location_type="urban",
            ),
            n_agents=5,  # Small for demo
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
