#!/usr/bin/env python3
"""Debug script to test price sensitivity analysis directly."""

import asyncio
import sys
import logging
from pathlib import Path

# Add backend to path
backend_path = str(Path(__file__).parent.parent / "backend")
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from uuid import uuid4
from app.services import PriceSensitivityEngine
from app.models import DemographicProfile
from app.i18n import Language

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_price_sensitivity():
    print("Starting price sensitivity test...")
    
    engine = PriceSensitivityEngine(language=Language.PL)
    
    print("Engine created. Running analysis...")
    print("Price points: [19.99, 29.99, 39.99]")
    print("N agents: 5 (small for quick test)")
    
    try:
        result = await engine.analyze_price_sensitivity(
            project_id=uuid4(),
            base_product_description="Testowy produkt - pasta do zębów z węglem aktywnym",
            price_points=[19.99, 29.99, 39.99],
            target_audience=DemographicProfile(age_min=25, age_max=45),
            n_agents=5,
            enable_web_search=False,
        )
        print("\n=== SUCCESS ===")
        print(f"Demand curve: {result['demand_curve']}")
        print(f"Optimal price: {result['optimal_price']}")
    except Exception as e:
        print(f"\n=== ERROR ===")
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_price_sensitivity())
