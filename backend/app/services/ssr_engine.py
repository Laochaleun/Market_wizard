"""
SSR Engine - Semantic Similarity Rating implementation.

Based on the methodology from:
Maier, B. F., et al. (2025). "LLMs Reproduce Human Purchase Intent via 
Semantic Similarity Elicitation of Likert Ratings". arXiv:2510.08338.

This module implements the core SSR algorithm which maps textual responses
to Likert scale ratings through semantic similarity with anchor statements.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List

from app.models import AnchorSet, LikertDistribution
from app.services.embedding_client import EmbeddingClient, get_embedding_client
from app.i18n import Language, get_anchor_sets


@dataclass
class SSRResult:
    """Result of SSR rating for a single response."""

    text_response: str
    likert_pmf: LikertDistribution
    expected_score: float
    raw_similarities: Dict[int, float]


class SSREngine:
    """
    Semantic Similarity Rating Engine.
    
    Maps textual LLM responses to Likert scale probability distributions
    by comparing embeddings with anchor statements.
    """

    def __init__(
        self,
        embedding_client: EmbeddingClient | None = None,
        anchor_sets: List[Dict[int, str]] | None = None,
        language: Language = Language.PL,
        temperature: float = 0.01,
        epsilon: float = 0.0,
    ):
        """
        Initialize the SSR Engine.
        
        Args:
            embedding_client: Client for generating embeddings.
            anchor_sets: List of anchor statement sets. If None, uses 
                        language-specific anchors from i18n module.
            language: Language for anchor statements (PL or EN).
            temperature: Controls how "spread out" the resulting PMF is.
                        T=1.0 is default from the paper.
            epsilon: Small regularization parameter to prevent division 
                    by zero and add smoothing.
        """
        self.embedding_client = embedding_client or get_embedding_client()
        self.anchor_sets = anchor_sets or get_anchor_sets(language)
        self.language = language
        self.temperature = temperature
        self.epsilon = epsilon

        # Pre-compute anchor embeddings
        self._anchor_embeddings = self._compute_anchor_embeddings()

    def _compute_anchor_embeddings(self) -> List[Dict[int, np.ndarray]]:
        """Pre-compute embeddings for all anchor statements."""
        anchor_embeddings = []

        for anchor_set in self.anchor_sets:
            statements = [anchor_set[i] for i in range(1, 6)]
            embeddings = self.embedding_client.embed(statements)

            anchor_embeddings.append(
                {i + 1: embeddings[i] for i in range(5)}
            )

        return anchor_embeddings

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def _compute_pmf_for_set(
        self, response_embedding: np.ndarray, anchor_set_idx: int
    ) -> Dict[int, float]:
        """
        Compute PMF for a single anchor set.
        
        Following the paper's methodology:
        1. Compute cosine similarity with each anchor
        2. Subtract minimum similarity (to handle low variance)
        3. Apply temperature scaling
        4. Normalize to get PMF
        """
        anchor_emb = self._anchor_embeddings[anchor_set_idx]

        # Compute similarities
        similarities = {
            r: self._cosine_similarity(response_embedding, anchor_emb[r])
            for r in range(1, 6)
        }

        # Subtract minimum (as per paper equation)
        min_sim = min(similarities.values())
        adjusted = {r: similarities[r] - min_sim + self.epsilon for r in range(1, 6)}

        # Apply temperature scaling
        if self.temperature != 1.0:
            adjusted = {r: adjusted[r] ** (1 / self.temperature) for r in range(1, 6)}

        # Normalize to PMF
        total = sum(adjusted.values())
        if total > 0:
            pmf = {r: adjusted[r] / total for r in range(1, 6)}
        else:
            pmf = {r: 0.2 for r in range(1, 6)}  # Uniform fallback

        return pmf

    def rate_response(self, text_response: str) -> SSRResult:
        """
        Rate a single text response using SSR methodology.
        
        Args:
            text_response: The LLM-generated opinion text.
            
        Returns:
            SSRResult containing the Likert PMF and expected score.
        """
        # Embed the response
        response_embedding = self.embedding_client.embed([text_response])[0]

        # Compute PMF for each anchor set and average
        pmfs = [
            self._compute_pmf_for_set(response_embedding, i)
            for i in range(len(self.anchor_sets))
        ]

        # Average across all anchor sets (as in the paper with m=6)
        avg_pmf = {
            r: np.mean([pmf[r] for pmf in pmfs]) for r in range(1, 6)
        }

        # Normalize averaged PMF
        total = sum(avg_pmf.values())
        avg_pmf = {r: avg_pmf[r] / total for r in range(1, 6)}

        # Compute expected Likert score
        expected_score = sum(r * avg_pmf[r] for r in range(1, 6))

        # Convert to LikertDistribution
        likert_dist = LikertDistribution(
            scale_1=avg_pmf[1],
            scale_2=avg_pmf[2],
            scale_3=avg_pmf[3],
            scale_4=avg_pmf[4],
            scale_5=avg_pmf[5],
        )

        return SSRResult(
            text_response=text_response,
            likert_pmf=likert_dist,
            expected_score=expected_score,
            raw_similarities=avg_pmf,
        )

    def rate_responses(self, text_responses: List[str]) -> List[SSRResult]:
        """Rate multiple responses efficiently."""
        return [self.rate_response(resp) for resp in text_responses]

    def aggregate_to_survey_pmf(self, results: List[SSRResult]) -> LikertDistribution:
        """
        Aggregate individual PMFs to a survey-level distribution.
        
        This represents the expected distribution of responses if we
        sampled from each individual's PMF.
        """
        if not results:
            return LikertDistribution(
                scale_1=0.2, scale_2=0.2, scale_3=0.2, scale_4=0.2, scale_5=0.2
            )

        # Average PMFs across all respondents
        avg = {r: np.mean([res.likert_pmf.model_dump()[f"scale_{r}"] for res in results]) 
               for r in range(1, 6)}

        # Normalize
        total = sum(avg.values())
        avg = {r: avg[r] / total for r in range(1, 6)}

        return LikertDistribution(
            scale_1=avg[1],
            scale_2=avg[2],
            scale_3=avg[3],
            scale_4=avg[4],
            scale_5=avg[5],
        )
