"""LLM client for generating synthetic consumer responses."""

from typing import Protocol

from app.config import get_settings
from app.models import Persona
from app.i18n import Language, get_persona_prompt


class LLMClient(Protocol):
    """Protocol for LLM clients."""

    async def generate_opinion(
        self, 
        persona: Persona, 
        product_description: str,
        language: Language = Language.PL,
    ) -> str:
        """Generate a purchase intent opinion from a synthetic consumer."""
        ...


class GeminiClient:
    """Google Gemini LLM client using new google-genai SDK."""

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash-001"):
        from google import genai

        self.client = genai.Client(api_key=api_key)
        self.model_name = model

    async def generate_opinion(
        self, 
        persona: Persona, 
        product_description: str,
        language: Language = Language.PL,
    ) -> str:
        """Generate opinion using Gemini."""
        from google.genai import types
        
        # Get language-specific prompt from i18n module
        prompt = get_persona_prompt(
            language=language,
            name=persona.name,
            age=persona.age,
            gender=persona.gender,
            location=persona.location,
            income=persona.income,
            occupation=persona.occupation,
            product_description=product_description,
        )

        # Use synchronous call wrapped for async (genai doesn't have native async yet)
        import asyncio
        loop = asyncio.get_event_loop()
        
        config = types.GenerateContentConfig(
            temperature=0.9,  # Higher temperature for more detailed responses
            max_output_tokens=2048,  # Enough for full responses
        )
        
        response = await loop.run_in_executor(
            None,
            lambda: self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config,
            )
        )

        return response.text or ""


class OpenAIClient:
    """OpenAI GPT client (alternative)."""

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model

    async def generate_opinion(
        self, 
        persona: Persona, 
        product_description: str,
        language: Language = Language.PL,
    ) -> str:
        """Generate opinion using OpenAI."""
        # Get language-specific prompt from i18n module
        prompt = get_persona_prompt(
            language=language,
            name=persona.name,
            age=persona.age,
            gender=persona.gender,
            location=persona.location,
            income=persona.income,
            occupation=persona.occupation,
            product_description=product_description,
        )

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9,
            max_tokens=2048,
        )

        return response.choices[0].message.content or ""


def get_llm_client(model_override: str | None = None) -> LLMClient:
    """Factory function to get the configured LLM client."""
    settings = get_settings()
    model = model_override or settings.llm_model

    if model.startswith("gemini"):
        if not settings.google_api_key:
            raise ValueError("GOOGLE_API_KEY required for Gemini models")
        return GeminiClient(api_key=settings.google_api_key, model=model)
    elif model.startswith("gpt"):
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY required for OpenAI models")
        return OpenAIClient(api_key=settings.openai_api_key, model=model)
    else:
        raise ValueError(f"Unknown model: {model}")
