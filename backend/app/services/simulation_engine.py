"""
Simulation Engine - Orchestrator for running SSR-based market research.

Coordinates persona generation, LLM opinion generation, and SSR rating
to produce complete simulation results.
"""

import asyncio
from datetime import datetime
from typing import List, Optional
from uuid import UUID

from app.models import (
    AgentResponse,
    DemographicProfile,
    LikertDistribution,
    Persona,
    SimulationResult,
)
from app.services.llm_client import LLMClient, get_llm_client
from app.services.persona_manager import PersonaManager
from app.services.ssr_engine import SSREngine, SSRResult
from app.i18n import Language


class SimulationEngine:
    """
    Orchestrates the complete SSR simulation process.
    
    Flow:
    1. Generate synthetic consumer personas
    2. For each persona, generate LLM opinion about product
    3. Rate each opinion using SSR methodology
    4. Aggregate results to population-level statistics
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        ssr_engine: SSREngine | None = None,
        persona_manager: PersonaManager | None = None,
        model_override: str | None = None,


        language: Language = Language.PL,
        temperature: float = 0.01,
        llm_temperature: float | None = None,
    ):
        self.language = language
        self.llm_client = llm_client or get_llm_client(model_override)
        self.ssr_engine = ssr_engine or SSREngine(language=language, temperature=temperature)
        self.persona_manager = persona_manager or PersonaManager(language=language)
        self.llm_temperature = llm_temperature

    async def _generate_opinion_for_persona(
        self,
        persona: Persona,
        product_description: str,
        enable_web_search: bool = False,
    ) -> tuple[Persona, str, list[str]]:
        """Generate opinion for a single persona. Returns (persona, opinion, sources)."""
        if enable_web_search and hasattr(self.llm_client, 'generate_opinion_with_search'):
            opinion, sources = await self.llm_client.generate_opinion_with_search(
                persona,
                product_description,
                language=self.language,
                temperature=self.llm_temperature,
            )
        else:
            opinion = await self.llm_client.generate_opinion(
                persona,
                product_description,
                language=self.language,
                temperature=self.llm_temperature,
            )
            sources = []
        return persona, opinion, sources

    async def run_simulation(
        self,
        project_id: UUID,
        product_description: str,
        target_audience: DemographicProfile | None = None,
        n_agents: int = 100,
        concurrency_limit: int = 10,
        enable_web_search: bool = False,
        personas: List[Persona] | None = None,
    ) -> SimulationResult:
        """
        Run a complete simulation.
        
        Args:
            project_id: ID of the research project
            product_description: Description of the product to evaluate
            target_audience: Demographic profile constraints
            n_agents: Number of synthetic consumers
            concurrency_limit: Max concurrent LLM requests
            enable_web_search: Enable Google Search grounding for market research
            
        Returns:
            SimulationResult with aggregate and individual results
        """
        # Step 1: Generate personas (unless provided)
        if personas is None:
            personas = self.persona_manager.generate_population(
                profile=target_audience,
                n_agents=n_agents,
            )
        else:
            n_agents = len(personas)

        # Step 2: Generate opinions with concurrency control
        opinions: List[tuple[Persona, str, list[str]]] = []
        semaphore = asyncio.Semaphore(concurrency_limit)

        async def generate_with_limit(persona: Persona) -> tuple[Persona, str, list[str]]:
            async with semaphore:
                return await self._generate_opinion_for_persona(
                    persona, product_description, enable_web_search
                )

        tasks = [generate_with_limit(p) for p in personas]
        opinions = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out failed generations (exceptions)
        valid_opinions: List[tuple[Persona, str, list[str]]] = []
        all_sources: List[str] = []
        for result in opinions:
            if isinstance(result, Exception):
                # Log error but continue
                print(f"⚠️ LLM error: {result}")
                continue
            if isinstance(result, tuple) and len(result) == 3:
                persona, opinion, sources = result
                if isinstance(opinion, str) and opinion:
                    valid_opinions.append((persona, opinion, sources))
                    all_sources.extend(sources)

        if not valid_opinions:
            raise RuntimeError("Wszystkie zapytania do LLM zakończyły się błędem")

        # Step 3: Rate opinions using SSR
        text_responses = [o for _, o, _ in valid_opinions]
        ssr_results: List[SSRResult] = self.ssr_engine.rate_responses(text_responses)

        # Step 4: Build agent responses
        agent_responses: List[AgentResponse] = []
        for (persona, _, _), ssr_result in zip(valid_opinions, ssr_results):
            agent_responses.append(
                AgentResponse(
                    persona=persona,
                    text_response=ssr_result.text_response,
                    likert_pmf=ssr_result.likert_pmf,
                    likert_score=ssr_result.expected_score,
                )
            )

        # Step 5: Aggregate to survey-level PMF
        aggregate_pmf = self.ssr_engine.aggregate_to_survey_pmf(ssr_results)
        mean_purchase_intent = sum(
            r * getattr(aggregate_pmf, f"scale_{r}") for r in range(1, 6)
        )

        return SimulationResult(
            project_id=project_id,
            n_agents=len(agent_responses),
            aggregate_distribution=aggregate_pmf,
            mean_purchase_intent=mean_purchase_intent,
            agent_responses=agent_responses,
            web_sources=list(set(all_sources)),  # Deduplicate sources
            created_at=datetime.now(),
        )


class ABTestEngine:
    """
    A/B Testing engine for comparing product variants.
    
    Runs parallel simulations for two product variants and 
    reports comparative statistics.
    """

    def __init__(
        self, 
        simulation_engine: SimulationEngine | None = None,
        language: Language = Language.PL,
    ):
        self.simulation_engine = simulation_engine or SimulationEngine(language=language)

    async def run_ab_test(
        self,
        project_id: UUID,
        variant_a: str,  # Product description A
        variant_b: str,  # Product description B
        target_audience: DemographicProfile | None = None,
        n_agents: int = 100,
    ) -> dict:
        """
        Run A/B test comparing two product variants.
        
        Returns comparison statistics including:
        - Mean purchase intent for each variant
        - Statistical significance
        - Conversion lift estimate
        """
        # Run both simulations with same personas for fair comparison
        personas = self.simulation_engine.persona_manager.generate_population(
            profile=target_audience,
            n_agents=n_agents,
        )

        # Run simulations in parallel
        result_a, result_b = await asyncio.gather(
            self.simulation_engine.run_simulation(
                project_id=project_id,
                product_description=variant_a,
                n_agents=n_agents,
                personas=personas,
            ),
            self.simulation_engine.run_simulation(
                project_id=project_id,
                product_description=variant_b,
                n_agents=n_agents,
                personas=personas,
            ),
        )

        # Calculate lift
        lift = (
            (result_b.mean_purchase_intent - result_a.mean_purchase_intent)
            / result_a.mean_purchase_intent
            * 100
            if result_a.mean_purchase_intent > 0
            else 0
        )

        return {
            "variant_a": {
                "mean_purchase_intent": result_a.mean_purchase_intent,
                "distribution": result_a.aggregate_distribution.model_dump(),
            },
            "variant_b": {
                "mean_purchase_intent": result_b.mean_purchase_intent,
                "distribution": result_b.aggregate_distribution.model_dump(),
            },
            "comparison": {
                "lift_percent": lift,
                "winner": "B" if lift > 0 else "A" if lift < 0 else "TIE",
                "n_agents_per_variant": n_agents,
            },
        }


class PriceSensitivityEngine:
    """
    Engine for analyzing price sensitivity.
    
    Runs simulations at multiple price points to generate
    a demand curve.
    """

    def __init__(
        self, 
        simulation_engine: SimulationEngine | None = None,
        language: Language = Language.PL,
    ):
        self.simulation_engine = simulation_engine or SimulationEngine(
            language=language,
            llm_temperature=0.2,
        )

    async def analyze_price_sensitivity(
        self,
        project_id: UUID,
        base_product_description: str,
        price_points: List[float],  # e.g., [19.99, 29.99, 39.99, 49.99, 59.99]
        target_audience: DemographicProfile | None = None,
        n_agents: int = 50,
    ) -> dict:
        """
        Analyze purchase intent at different price points.
        
        Returns a demand curve mapping price to purchase intent.
        """
        import hashlib
        import json
        import random
        import numpy as np

        results = {}

        # Stable persona seed for repeatability (based on inputs).
        seed_payload = {
            "product": base_product_description,
            "prices": [round(p, 2) for p in price_points],
            "n_agents": n_agents,
            "target": target_audience.model_dump() if target_audience else None,
        }
        seed_raw = json.dumps(seed_payload, sort_keys=True)
        seed = int(hashlib.md5(seed_raw.encode("utf-8")).hexdigest()[:8], 16)

        # Preserve RNG state to avoid affecting other runs.
        rand_state = random.getstate()
        np_state = np.random.get_state()
        random.seed(seed)
        np.random.seed(seed)
        try:
            personas = self.simulation_engine.persona_manager.generate_population(
                profile=target_audience,
                n_agents=n_agents,
            )
        finally:
            random.setstate(rand_state)
            np.random.set_state(np_state)

        for price in price_points:
            # Inject price into product description
            product_with_price = f"{base_product_description}\n\nCena: {price:.2f} PLN"

            result = await self.simulation_engine.run_simulation(
                project_id=project_id,
                product_description=product_with_price,
                target_audience=target_audience,
                n_agents=n_agents,
                personas=personas,
            )

            results[price] = {
                "mean_purchase_intent": result.mean_purchase_intent,
                "distribution": result.aggregate_distribution.model_dump(),
            }

        # Calculate price elasticity between adjacent points
        prices = sorted(price_points)
        elasticities = []
        for i in range(len(prices) - 1):
            p1, p2 = prices[i], prices[i + 1]
            q1, q2 = results[p1]["mean_purchase_intent"], results[p2]["mean_purchase_intent"]

            # Price elasticity of demand
            if q1 > 0 and p1 > 0:
                elasticity = ((q2 - q1) / q1) / ((p2 - p1) / p1)
                elasticities.append({
                    "price_range": f"{p1:.2f}-{p2:.2f}",
                    "elasticity": elasticity,
                })

        return {
            "demand_curve": results,
            "elasticities": elasticities,
            "optimal_price": max(results.keys(), key=lambda p: results[p]["mean_purchase_intent"]),
            "seed": seed,
        }
