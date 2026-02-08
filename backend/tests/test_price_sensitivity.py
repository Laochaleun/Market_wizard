import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
from app.services import PriceSensitivityEngine, SimulationEngine
from app.models import DemographicProfile, SimulationResult, AgentResponse, LikertDistribution

@pytest.fixture
def mock_simulation_engine():
    engine = MagicMock(spec=SimulationEngine)
    engine.language = "PL"
    engine.llm_client = MagicMock()
    # Mock generate_market_sources behavior
    engine.llm_client.generate_market_sources = AsyncMock(return_value=[])
    
    # Needs a persona_manager with generate_population
    engine.persona_manager = MagicMock()
    engine.persona_manager.generate_population = MagicMock(return_value=[MagicMock() for _ in range(10)])
    
    return engine

@pytest.mark.asyncio
async def test_analyze_price_sensitivity(mock_simulation_engine):
    # Setup
    price_engine = PriceSensitivityEngine(simulation_engine=mock_simulation_engine)
    
    # Mock run_simulation to return a dummy result
    async def mock_run_simulation(*args, **kwargs):
        product_desc = kwargs.get("product_description", "")
        # Extract price from description to simulate lower intent for higher price
        import re
        price_match = re.search(r"Cena: (\d+\.\d+) PLN", product_desc)
        price = float(price_match.group(1)) if price_match else 0.0
        
        # Simple logic: higher price -> lower intent
        intent = max(1.0, 5.0 - (price / 20.0)) 
        
        return SimulationResult(
            project_id=uuid4(),
            n_agents=kwargs.get("n_agents", 10),
            aggregate_distribution=LikertDistribution(
                scale_1=0.1, scale_2=0.1, scale_3=0.2, scale_4=0.3, scale_5=0.3
            ),
            mean_purchase_intent=intent,
            agent_responses=[],
            web_sources=[],
            created_at=datetime.now()
        )
    
    mock_simulation_engine.run_simulation = AsyncMock(side_effect=mock_run_simulation)

    # Execute
    project_id = uuid4()
    base_description = "Test Product"
    price_points = [20.0, 30.0, 40.0]
    target_audience = DemographicProfile(age_min=20, age_max=30)
    
    result = await price_engine.analyze_price_sensitivity(
        project_id=project_id,
        base_product_description=base_description,
        price_points=price_points,
        target_audience=target_audience,
        n_agents=10
    )

    # Verify
    assert "demand_curve" in result
    assert "elasticities" in result
    assert "optimal_price" in result
    assert "seed" in result
    
    demand_curve = result["demand_curve"]
    assert len(demand_curve) == 3
    assert 20.0 in demand_curve
    assert 30.0 in demand_curve
    assert 40.0 in demand_curve
    
    # Verify values indicate downward trend (as mocked)
    intent_20 = demand_curve[20.0]["mean_purchase_intent"]
    intent_40 = demand_curve[40.0]["mean_purchase_intent"]
    assert intent_20 > intent_40

    # Verify run_simulation was called 3 times
    assert mock_simulation_engine.run_simulation.call_count == 3
