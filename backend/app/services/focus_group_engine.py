"""
Focus Group Engine - Multi-agent discussion simulation.

Orchestrates virtual focus group sessions where multiple synthetic
consumers discuss a product, reacting to each other's opinions.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
from uuid import UUID, uuid4

from app.i18n import Language
from app.models import Persona, DemographicProfile
from app.services.persona_manager import PersonaManager
from app.services.llm_client import LLMClient, get_llm_client
from app.config import get_settings


@dataclass
class FocusGroupMessage:
    """A single message in the focus group discussion."""
    persona_name: str
    persona_demographics: str  # e.g., "35F, Warszawa, 8000 PLN"
    round: int
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    

@dataclass
class FocusGroupResult:
    """Complete result of a focus group session."""
    session_id: UUID
    product: str
    participants: List[Persona]
    discussion: List[FocusGroupMessage]
    summary: str  # AI moderator summary
    key_insights: List[str]
    consensus_topics: List[str]  # Points of agreement
    disagreement_topics: List[str]  # Points of disagreement
    

class FocusGroupEngine:
    """
    Engine for running virtual focus group sessions.
    
    Flow:
    1. Generate diverse personas (4-8 participants)
    2. Round 1: Each participant gives initial impressions
    3. Rounds 2-N: Participants react to previous statements
    4. AI Moderator synthesizes insights
    """
    
    def __init__(
        self,
        llm_client: LLMClient | None = None,
        persona_manager: PersonaManager | None = None,
        language: Language = Language.PL,
    ):
        self.llm_client = llm_client or get_llm_client()
        self.persona_manager = persona_manager or PersonaManager(language=language)
        self.language = language
        
    async def run_focus_group(
        self,
        product_description: str,
        n_participants: int = 6,
        n_rounds: int = 3,
        target_audience: DemographicProfile | None = None,
    ) -> FocusGroupResult:
        """
        Run a complete focus group session.
        
        Args:
            product_description: Product to discuss
            n_participants: Number of participants (4-8)
            n_rounds: Number of discussion rounds (2-4)
            target_audience: Optional demographic constraints
            
        Returns:
            FocusGroupResult with full discussion transcript and insights
        """
        # Clamp participants to reasonable range
        n_participants = max(4, min(8, n_participants))
        n_rounds = max(2, min(4, n_rounds))
        
        # Generate diverse personas
        personas = self.persona_manager.generate_population(
            n_agents=n_participants,
            profile=target_audience,
        )
        
        discussion: List[FocusGroupMessage] = []
        
        # Round 1: Initial impressions
        round1_tasks = [
            self._generate_initial_opinion(persona, product_description)
            for persona in personas
        ]
        round1_messages = await asyncio.gather(*round1_tasks)
        discussion.extend(round1_messages)
        
        # Rounds 2-N: Discussion (react to previous messages)
        for round_num in range(2, n_rounds + 1):
            previous_messages = [m for m in discussion if m.round == round_num - 1]
            
            round_tasks = [
                self._generate_response(
                    persona, 
                    previous_messages, 
                    product_description,
                    round_num,
                )
                for persona in personas
            ]
            round_messages = await asyncio.gather(*round_tasks)
            discussion.extend(round_messages)
        
        # Generate moderator summary
        summary, insights, consensus, disagreement = await self._generate_moderator_summary(
            product_description, 
            personas, 
            discussion,
        )
        
        return FocusGroupResult(
            session_id=uuid4(),
            product=product_description,
            participants=personas,
            discussion=discussion,
            summary=summary,
            key_insights=insights,
            consensus_topics=consensus,
            disagreement_topics=disagreement,
        )
    
    async def _generate_initial_opinion(
        self, 
        persona: Persona, 
        product: str,
    ) -> FocusGroupMessage:
        """Generate initial impression for round 1."""
        if self.language == Language.PL:
            prompt = f"""Jesteś {persona.name}, uczestnikiem grupy fokusowej.
Masz {persona.age} lat, mieszkasz w {persona.location}, zarabiasz {persona.income:,} PLN miesięcznie.
{f'Pracujesz jako {persona.occupation}.' if persona.occupation else ''}

Moderator przedstawił produkt do oceny:
"{product}"

Podziel się swoimi pierwszymi wrażeniami (2-3 zdania). Mów naturalnie, jak w prawdziwej dyskusji.
Odpowiedz TYLKO swoją wypowiedzią, bez żadnych wstępów."""
        else:
            prompt = f"""You are {persona.name}, a focus group participant.
You are {persona.age} years old, living in {persona.location}, earning ${persona.income:,} monthly.
{f'You work as {persona.occupation}.' if persona.occupation else ''}

The moderator presented this product for evaluation:
"{product}"

Share your first impressions (2-3 sentences). Speak naturally, as in a real discussion.
Reply ONLY with your statement, no introductions."""

        from google import genai
        from google.genai import types
        
        settings = get_settings()
        client = genai.Client(api_key=settings.google_api_key)
        
        config = types.GenerateContentConfig(
            temperature=0.9,
            max_output_tokens=300,
        )
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.models.generate_content(
                model="gemini-2.0-flash-001",
                contents=prompt,
                config=config,
            )
        )
        
        demographics = f"{persona.age}{persona.gender[0]}, {persona.location}, {persona.income:,} PLN"
        
        return FocusGroupMessage(
            persona_name=persona.name,
            persona_demographics=demographics,
            round=1,
            content=response.text.strip() if response.text else "",
        )
    
    async def _generate_response(
        self,
        persona: Persona,
        previous_messages: List[FocusGroupMessage],
        product: str,
        round_num: int,
    ) -> FocusGroupMessage:
        """Generate response to previous round's statements."""
        # Format previous opinions
        prev_opinions = "\n".join([
            f"- {m.persona_name}: \"{m.content}\""
            for m in previous_messages
            if m.persona_name != persona.name
        ])
        
        if self.language == Language.PL:
            prompt = f"""Jesteś {persona.name}, uczestnikiem grupy fokusowej.
Masz {persona.age} lat, mieszkasz w {persona.location}, zarabiasz {persona.income:,} PLN miesięcznie.

Dyskutujecie o produkcie: "{product}"

Inni uczestnicy właśnie powiedzieli:
{prev_opinions}

Odpowiedz krótko (2-3 zdania). Możesz:
- Zgodzić się z kimś i rozwinąć myśl
- Nie zgodzić się i podać swój punkt widzenia
- Dodać nową perspektywę

Bądź naturalny/a i autentyczny/a. Odpowiedz TYLKO swoją wypowiedzią."""
        else:
            prompt = f"""You are {persona.name}, a focus group participant.
You are {persona.age} years old, living in {persona.location}, earning ${persona.income:,} monthly.

You are discussing: "{product}"

Other participants just said:
{prev_opinions}

Reply briefly (2-3 sentences). You may:
- Agree with someone and expand on their point
- Disagree and share your perspective
- Add a new viewpoint

Be natural and authentic. Reply ONLY with your statement."""

        from google import genai
        from google.genai import types
        
        settings = get_settings()
        client = genai.Client(api_key=settings.google_api_key)
        
        config = types.GenerateContentConfig(
            temperature=0.9,
            max_output_tokens=300,
        )
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.models.generate_content(
                model="gemini-2.0-flash-001",
                contents=prompt,
                config=config,
            )
        )
        
        demographics = f"{persona.age}{persona.gender[0]}, {persona.location}, {persona.income:,} PLN"
        
        return FocusGroupMessage(
            persona_name=persona.name,
            persona_demographics=demographics,
            round=round_num,
            content=response.text.strip() if response.text else "",
        )
    
    async def _generate_moderator_summary(
        self,
        product: str,
        participants: List[Persona],
        discussion: List[FocusGroupMessage],
    ) -> tuple[str, List[str], List[str], List[str]]:
        """Generate AI moderator summary of the discussion."""
        # Format full discussion
        discussion_text = ""
        current_round = 0
        for msg in discussion:
            if msg.round != current_round:
                current_round = msg.round
                if self.language == Language.PL:
                    discussion_text += f"\n--- Runda {current_round} ---\n"
                else:
                    discussion_text += f"\n--- Round {current_round} ---\n"
            discussion_text += f"{msg.persona_name}: {msg.content}\n"
        
        if self.language == Language.PL:
            prompt = f"""Jesteś moderatorem grupy fokusowej. Przeanalizuj poniższą dyskusję o produkcie.

PRODUKT: {product}

UCZESTNICY: {', '.join([p.name for p in participants])}

DYSKUSJA:
{discussion_text}

Napisz raport w formacie:

PODSUMOWANIE:
[2-3 zdania ogólnego podsumowania]

KLUCZOWE WNIOSKI:
- [wniosek 1]
- [wniosek 2]
- [wniosek 3]

PUNKTY ZGODNOŚCI:
- [temat w którym uczestnicy się zgadzają]

PUNKTY NIEZGODNOŚCI:
- [temat kontrowersyjny]

REKOMENDACJE DLA PRODUCENTA:
[1-2 zdania z rekomendacją]"""
        else:
            prompt = f"""You are a focus group moderator. Analyze the following product discussion.

PRODUCT: {product}

PARTICIPANTS: {', '.join([p.name for p in participants])}

DISCUSSION:
{discussion_text}

Write a report in this format:

SUMMARY:
[2-3 sentences overall summary]

KEY INSIGHTS:
- [insight 1]
- [insight 2]
- [insight 3]

POINTS OF AGREEMENT:
- [topic participants agree on]

POINTS OF DISAGREEMENT:
- [controversial topic]

RECOMMENDATIONS FOR PRODUCER:
[1-2 sentences with recommendation]"""

        from google import genai
        from google.genai import types
        
        settings = get_settings()
        client = genai.Client(api_key=settings.google_api_key)
        
        config = types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=1000,
        )
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.models.generate_content(
                model="gemini-2.0-flash-001",
                contents=prompt,
                config=config,
            )
        )
        
        summary_text = response.text.strip() if response.text else ""
        
        # Parse insights (simplified - just return the full summary)
        # In production, would parse structured sections
        insights = []
        consensus = []
        disagreement = []
        
        # Basic parsing
        lines = summary_text.split('\n')
        current_section = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if 'WNIOSKI' in line.upper() or 'INSIGHTS' in line.upper():
                current_section = 'insights'
            elif 'ZGODNOŚCI' in line.upper() or 'AGREEMENT' in line.upper():
                current_section = 'consensus'
            elif 'NIEZGODNOŚCI' in line.upper() or 'DISAGREEMENT' in line.upper():
                current_section = 'disagreement'
            elif 'REKOMEND' in line.upper() or 'SUMMAR' in line.upper():
                current_section = None
            elif line.startswith('-') or line.startswith('•'):
                item = line.lstrip('-•').strip()
                if current_section == 'insights':
                    insights.append(item)
                elif current_section == 'consensus':
                    consensus.append(item)
                elif current_section == 'disagreement':
                    disagreement.append(item)
        
        return summary_text, insights, consensus, disagreement
