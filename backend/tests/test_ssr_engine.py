"""Tests for SSR Engine."""

import numpy as np
import pytest

from app.models import LikertDistribution
from app.services.ssr_engine import SSREngine, SSRResult
from app.i18n import Language, get_anchor_sets


class MockEmbeddingClient:
    """Mock embedding client for testing without actual API calls."""

    def __init__(self):
        # Pre-defined embeddings for testing
        self.embeddings = {
            # Negative responses should be close to anchor 1
            "I definitely won't buy this product": np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
            "This product doesn't interest me at all": np.array([0.9, 0.1, 0.0, 0.0, 0.0]),
            "No way I would ever buy this": np.array([0.95, 0.05, 0.0, 0.0, 0.0]),
            
            # Positive responses should be close to anchor 5
            "I would definitely buy this product": np.array([0.0, 0.0, 0.0, 0.1, 0.9]),
            "I will certainly buy this product": np.array([0.0, 0.0, 0.0, 0.0, 1.0]),
            "I can't wait to buy this product": np.array([0.0, 0.0, 0.0, 0.05, 0.95]),
            
            # Neutral responses
            "I'm not sure if I would buy this product": np.array([0.0, 0.2, 0.6, 0.2, 0.0]),
            "I might consider buying this": np.array([0.0, 0.1, 0.5, 0.3, 0.1]),
            
            # Test inputs
            "This is terrible, I would never buy it": np.array([0.85, 0.1, 0.05, 0.0, 0.0]),
            "I love this, definitely buying!": np.array([0.0, 0.0, 0.05, 0.1, 0.85]),
            "It's okay, might consider it": np.array([0.1, 0.15, 0.5, 0.2, 0.05]),
        }
        
        # Also add anchor embeddings
        for anchor_set in get_anchor_sets(Language.EN):
            for score, statement in anchor_set.items():
                if statement not in self.embeddings:
                    # Create simple embeddings based on score
                    emb = np.zeros(5)
                    emb[score - 1] = 1.0
                    self.embeddings[statement] = emb

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return mock embeddings."""
        result = []
        for text in texts:
            if text in self.embeddings:
                result.append(self.embeddings[text])
            else:
                # Default to neutral
                result.append(np.array([0.1, 0.2, 0.4, 0.2, 0.1]))
        return np.array(result)


class TestSSREngine:
    """Test suite for SSR Engine."""

    @pytest.fixture
    def engine(self):
        """Create SSR Engine with mock embeddings."""
        return SSREngine(embedding_client=MockEmbeddingClient(), language=Language.EN)

    def test_rate_negative_response(self, engine):
        """Test that negative responses get low scores."""
        result = engine.rate_response("This is terrible, I would never buy it")
        
        assert isinstance(result, SSRResult)
        assert result.expected_score < 2.5  # Should be in lower half
        assert result.likert_pmf.scale_1 > result.likert_pmf.scale_5

    def test_rate_positive_response(self, engine):
        """Test that positive responses get high scores."""
        result = engine.rate_response("I love this, definitely buying!")
        
        assert result.expected_score > 3.5  # Should be in upper half
        assert result.likert_pmf.scale_5 > result.likert_pmf.scale_1

    def test_rate_neutral_response(self, engine):
        """Test that neutral responses get middle scores."""
        result = engine.rate_response("It's okay, might consider it")
        
        assert 2.0 < result.expected_score < 4.0  # Should be in middle

    def test_pmf_sums_to_one(self, engine):
        """Test that PMF is a valid probability distribution."""
        result = engine.rate_response("Any random text here")
        
        pmf = result.likert_pmf
        total = pmf.scale_1 + pmf.scale_2 + pmf.scale_3 + pmf.scale_4 + pmf.scale_5
        assert abs(total - 1.0) < 0.01  # Should sum to 1

    def test_aggregate_multiple_responses(self, engine):
        """Test aggregation of multiple responses."""
        responses = [
            "This is terrible, I would never buy it",
            "I love this, definitely buying!",
            "It's okay, might consider it",
        ]
        results = engine.rate_responses(responses)
        
        assert len(results) == 3
        
        aggregate = engine.aggregate_to_survey_pmf(results)
        assert isinstance(aggregate, LikertDistribution)
        
        # Check PMF is valid
        total = aggregate.scale_1 + aggregate.scale_2 + aggregate.scale_3 + aggregate.scale_4 + aggregate.scale_5
        assert abs(total - 1.0) < 0.01

    def test_expected_score_range(self, engine):
        """Test that expected scores are in valid range."""
        responses = [
            "Very negative opinion",
            "Somewhat negative",
            "Neutral thoughts",
            "Somewhat positive",
            "Very positive opinion!"
        ]
        
        for resp in responses:
            result = engine.rate_response(resp)
            assert 1.0 <= result.expected_score <= 5.0


class TestLikertDistribution:
    """Test Likert distribution model."""

    def test_mean_score_calculation(self):
        """Test mean score calculation."""
        dist = LikertDistribution(
            scale_1=0.0,
            scale_2=0.0,
            scale_3=0.0,
            scale_4=0.0,
            scale_5=1.0,
        )
        assert dist.mean_score == 5.0

        dist2 = LikertDistribution(
            scale_1=1.0,
            scale_2=0.0,
            scale_3=0.0,
            scale_4=0.0,
            scale_5=0.0,
        )
        assert dist2.mean_score == 1.0

        dist3 = LikertDistribution(
            scale_1=0.2,
            scale_2=0.2,
            scale_3=0.2,
            scale_4=0.2,
            scale_5=0.2,
        )
        assert dist3.mean_score == 3.0  # Uniform distribution
