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

    async def generate_opinion_with_search(
        self,
        persona: Persona,
        product_description: str,
        language: Language = Language.PL,
    ) -> tuple[str, list[str]]:
        """Generate opinion with Google Search grounding.
        
        The agent will search the web for market information before forming opinion.
        
        Returns:
            tuple of (opinion_text, list_of_source_urls)
        """
        from google.genai import types
        import asyncio
        
        # Enhanced prompt with proper comparison instructions
        if language == Language.PL:
            prompt = f"""Jesteś {persona.name}, masz {persona.age} lat, płeć: {persona.gender}.
Mieszkasz w {persona.location}, zarabiasz {persona.income:,} PLN miesięcznie.
{f'Pracujesz jako {persona.occupation}.' if persona.occupation else ''}

PRODUKT DO OCENY: {product_description}

INSTRUKCJE WYSZUKIWANIA:
1. Wyszukaj PODOBNE produkty (ta sama kategoria, zbliżone parametry)
2. Zwróć uwagę na AKTUALNE ceny (2024-2025)
3. Jeśli nie ma identycznego produktu, znajdź najbliższy odpowiednik

PORÓWNANIE (uwzględnij w odpowiedzi):
- Podaj 1-2 konkretne produkty konkurencji z cenami (np. "Model X kosztuje 120 zł")
- Jeśli produkty różnią się cechami (pojemność, marka, funkcje), zaznacz to
- Oceń czy cena analizowanego produktu jest uczciwa biorąc pod uwagę różnice

JEŚLI NIE MA ANALOGICZNYCH PRODUKTÓW:
- Zaznacz że produkt jest UNIKALNY na rynku
- Oceń DLACZEGO nie ma konkurencji:
  * Czy to INNOWACJA (nowa wartość, rozwiązuje problem)?
  * Czy to NISZA (mały rynek, ale realny)?
  * Czy to produkt BEZSENSOWNY (nikt tego nie potrzebuje)?
- Rozważ czy cena jest uzasadniona

TWOJA ODPOWIEDŹ (jako konsument):
1. Jak ten produkt wypada vs konkurencja (lub zaznacz że jest unikalny)?
2. Czy cena jest atrakcyjna biorąc pod uwagę Twoje zarobki ({persona.income:,} PLN)?
3. Czy kupiłbyś/kupiłabyś? Dlaczego?

Odpowiedz naturalnie, jak w rozmowie. Wspomnij konkretne znalezione produkty i ceny."""
        else:
            prompt = f"""You are {persona.name}, {persona.age} years old, {persona.gender}.
Living in {persona.location}, earning ${persona.income:,} monthly.
{f'Working as {persona.occupation}.' if persona.occupation else ''}

PRODUCT TO EVALUATE: {product_description}

SEARCH INSTRUCTIONS:
1. Search for SIMILAR products (same category, similar specs)
2. Focus on CURRENT prices (2024-2025)
3. If no identical product exists, find closest equivalent

COMPARISON (include in your response):
- Name 1-2 specific competing products with prices (e.g., "Model X costs $120")
- If products differ in features (capacity, brand, functions), note it
- Assess if product's price is fair considering the differences

IF NO ANALOGOUS PRODUCTS EXIST:
- Note that product is UNIQUE on the market
- Assess WHY there's no competition:
  * Is it INNOVATIVE (new value, solves a problem)?
  * Is it a NICHE (small market, but real)?
  * Is it POINTLESS (nobody needs this)?
- Consider if the price is justified

YOUR RESPONSE (as a consumer):
1. How does this product compare to competition (or note if it's unique)?
2. Is the price attractive given your income (${persona.income:,})?
3. Would you buy it? Why?

Answer naturally, as in conversation. Mention specific products and prices you found."""

        # Enable Google Search grounding
        grounding_tool = types.Tool(google_search=types.GoogleSearch())
        
        config = types.GenerateContentConfig(
            tools=[grounding_tool],
            temperature=0.9,
            max_output_tokens=2048,
        )
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config,
            )
        )
        
        # Extract source URLs from grounding metadata
        sources = []
        if response.candidates and response.candidates[0].grounding_metadata:
            metadata = response.candidates[0].grounding_metadata
            if hasattr(metadata, 'grounding_chunks') and metadata.grounding_chunks:
                for chunk in metadata.grounding_chunks:
                    if hasattr(chunk, 'web') and chunk.web:
                        sources.append(chunk.web.uri)
        
        return response.text or "", sources

    async def extract_product_from_url(
        self,
        url: str,
        language: Language = Language.PL,
    ) -> str:
        """Extract product description from a shop URL using hybrid fetch + Gemini fallback.
        
        Args:
            url: URL of the product page
            language: Output language
            
        Returns:
            Extracted product description
        """
        import asyncio
        from google.genai import types
        from app.services.product_extractor import extract_product_description
        
        # Try direct extraction first (HTTP + optional Playwright)
        extracted = await extract_product_description(url, language.value)
        if extracted:
            return extracted

        if language == Language.PL:
            prompt = f"""Przeanalizuj stronę produktu: {url}

Zwróć TYLKO opis produktu w formacie:
"[Nazwa produktu] marki [marka], cena [cena]. [Kluczowe cechy w 1-2 zdaniach]."

WAŻNE:
- NIE dodawaj żadnych komentarzy, pytań ani wyjaśnień
- NIE pisz "Name included?" ani podobnych uwag
- Odpowiedź ma zawierać WYŁĄCZNIE opis produktu
- Maksymalnie 2-3 zdania"""
        else:
            prompt = f"""Analyze product page: {url}

Return ONLY the product description in format:
"[Product name] by [brand], price [price]. [Key features in 1-2 sentences]."

IMPORTANT:
- Do NOT add any comments, questions or explanations
- Do NOT write "Name included?" or similar notes
- Response must contain ONLY the product description
- Maximum 2-3 sentences"""

        # Enable URL Context tool
        url_context_tool = types.Tool(url_context=types.UrlContext())
        
        config = types.GenerateContentConfig(
            tools=[url_context_tool],
            temperature=0.1,  # Very low temp for factual extraction
            max_output_tokens=300,
        )
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config,
            )
        )
        
        text = response.text or ""
        
        # Post-process to remove thinking artifacts
        text = self._clean_extraction_artifacts(text)
        
        return text

    def _clean_extraction_artifacts(self, text: str) -> str:
        """Remove internal thinking artifacts from Gemini response."""
        import re
        
        # Remove lines with common thinking patterns
        patterns_to_remove = [
            r'\*.*included\?.*\*',  # *Name included? Yes*
            r'\*.*Format.*\*',
            r'\*.*Self-Correction.*\*',
            r'\*.*Let me.*\*',
            r'\*.*Wait.*\*',
            r'\*.*double-check.*\*',
            r'\*.*browsing.*\*',
            r'\*Only description\?.*\*',
            r'\*No comments\?.*\*',
            r'\*Max.*sentences\?.*\*',
            r'\*.*Polish Version.*\*',
            r'\*.*Final.*Version.*\*',
            r'^\s*\*\s*\*.*$',  # Lines with just asterisks
        ]
        
        for pattern in patterns_to_remove:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
        
        # Remove multiple spaces and newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        # Get first coherent paragraph (the actual description)
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        clean_lines = []
        for line in lines:
            # Skip lines that look like internal notes
            if any(x in line.lower() for x in ['included?', 'format?', 'let me', 'self-correction', 'double-check']):
                continue
            clean_lines.append(line)
        
        return ' '.join(clean_lines).strip()


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
