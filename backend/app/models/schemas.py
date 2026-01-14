"""Pydantic models for API requests and responses."""

from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# === Persona Models ===


class DemographicProfile(BaseModel):
    """Demographic attributes for a synthetic consumer persona."""

    age_min: int = Field(18, ge=0, le=120, description="Minimum age")
    age_max: int = Field(65, ge=0, le=120, description="Maximum age")
    gender: Optional[str] = Field(None, description="Gender filter: 'M', 'F', or None for all")
    income_level: Optional[str] = Field(
        None, description="Income level: 'low', 'medium', 'high', or None"
    )
    location_type: Optional[str] = Field(
        None, description="Location type: 'urban', 'suburban', 'rural', or None"
    )
    region: Optional[str] = Field(None, description="Region/voivodeship name")


class Persona(BaseModel):
    """A single synthetic consumer persona."""

    id: UUID = Field(default_factory=uuid4)
    name: str
    age: int
    gender: str
    income: int  # Monthly income in PLN
    location: str
    location_type: str
    education: Optional[str] = None
    occupation: Optional[str] = None

    def to_prompt_description(self) -> str:
        """Generate a natural language description for LLM prompts."""
        income_desc = (
            "low" if self.income < 4000 else "medium" if self.income < 8000 else "high"
        )
        return (
            f"{self.name}, {self.age} years old, {self.gender}, "
            f"living in {self.location} ({self.location_type} area), "
            f"with {income_desc} income (~{self.income} PLN/month)"
        )


# === Project Models ===


class ProjectCreate(BaseModel):
    """Request model for creating a new research project."""

    name: str = Field(..., min_length=1, max_length=200)
    product_description: str = Field(..., min_length=10)
    product_price: Optional[float] = Field(None, ge=0)
    target_audience: DemographicProfile = Field(default_factory=DemographicProfile)


class ProjectResponse(BaseModel):
    """Response model for a research project."""

    id: UUID
    name: str
    product_description: str
    product_price: Optional[float]
    target_audience: DemographicProfile
    created_at: datetime


# === Simulation Models ===


class SimulationRequest(BaseModel):
    """Request model for running a simulation."""

    project_id: UUID
    n_agents: int = Field(100, ge=1, le=10000, description="Number of synthetic consumers")
    llm_model: Optional[str] = Field(None, description="Override default LLM model")


class LikertDistribution(BaseModel):
    """Probability distribution over Likert scale."""

    scale_1: float = Field(..., ge=0, le=1, description="Definitely won't buy")
    scale_2: float = Field(..., ge=0, le=1, description="Probably won't buy")
    scale_3: float = Field(..., ge=0, le=1, description="Neutral")
    scale_4: float = Field(..., ge=0, le=1, description="Probably will buy")
    scale_5: float = Field(..., ge=0, le=1, description="Definitely will buy")

    @property
    def mean_score(self) -> float:
        """Calculate weighted mean purchase intent score."""
        return (
            1 * self.scale_1
            + 2 * self.scale_2
            + 3 * self.scale_3
            + 4 * self.scale_4
            + 5 * self.scale_5
        )


class AgentResponse(BaseModel):
    """Response from a single synthetic agent."""

    persona: Persona
    text_response: str
    likert_pmf: LikertDistribution
    likert_score: float  # Expected value


class SimulationResult(BaseModel):
    """Complete results of a simulation run."""

    id: UUID = Field(default_factory=uuid4)
    project_id: UUID
    n_agents: int
    aggregate_distribution: LikertDistribution
    mean_purchase_intent: float
    agent_responses: list[AgentResponse]
    created_at: datetime = Field(default_factory=datetime.now)


# === Anchor Models ===


class AnchorSet(BaseModel):
    """A set of anchor statements for SSR mapping."""

    id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., description="Name of the anchor set, e.g., 'purchase_intent'")
    statements: dict[int, str] = Field(
        ...,
        description="Mapping from Likert score (1-5) to anchor statement",
        examples=[
            {
                1: "I definitely won't buy this product",
                2: "I probably won't buy this product",
                3: "I'm not sure if I would buy this product",
                4: "I would probably buy this product",
                5: "I would definitely buy this product",
            }
        ],
    )
