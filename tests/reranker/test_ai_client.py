"""Tests for the AI reranker client."""

import pytest
import os

from recommender.reranker.schemas import (
    MovieCandidate,
    UserContext,
    RerankedMovie,
    RerankerResponse,
)
from recommender.reranker.ai_client import TogetherAIClient


class TestSchemas:
    """Test Pydantic schemas."""
    
    def test_movie_candidate_creation(self):
        """Test creating a MovieCandidate."""
        candidate = MovieCandidate(
            movie_id=1,
            title="The Matrix (1999)",
            genres=["Action", "Sci-Fi"],
            year=1999,
            gnn_score=0.95,
            gnn_rank=1,
        )
        assert candidate.movie_id == 1
        assert candidate.title == "The Matrix (1999)"
        assert "Sci-Fi" in candidate.genres
        assert candidate.year == 1999
    
    def test_movie_candidate_optional_year(self):
        """Test MovieCandidate with no year."""
        candidate = MovieCandidate(
            movie_id=2,
            title="Unknown Movie",
            genres=["Drama"],
            gnn_score=0.5,
            gnn_rank=2,
        )
        assert candidate.year is None
    
    def test_user_context_creation(self):
        """Test creating UserContext."""
        context = UserContext(
            user_id=42,
            liked_movies=["Inception (2010)", "The Dark Knight (2008)"],
            disliked_movies=["Bad Movie (2020)"],
        )
        assert context.user_id == 42
        assert len(context.liked_movies) == 2
        assert len(context.disliked_movies) == 1
    
    def test_user_context_empty_lists(self):
        """Test UserContext with empty lists."""
        context = UserContext(user_id=1)
        assert context.liked_movies == []
        assert context.disliked_movies == []
    
    def test_reranker_response_validation(self):
        """Test RerankerResponse from dict."""
        data = {
            "recommendations": [
                {"movie_id": 1, "rank": 1, "reasoning": "Matches sci-fi preference"},
                {"movie_id": 2, "rank": 2, "reasoning": "Good action movie"},
            ]
        }
        response = RerankerResponse.model_validate(data)
        assert len(response.recommendations) == 2
        assert response.recommendations[0].rank == 1


class TestTogetherAIClient:
    """Test Together AI client (requires API key for integration tests)."""
    
    def test_client_requires_api_key(self):
        """Test that client raises error without API key."""
        # Temporarily remove API key
        original_key = os.environ.pop("TOGETHER_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="TOGETHER_API_KEY"):
                TogetherAIClient()
        finally:
            if original_key:
                os.environ["TOGETHER_API_KEY"] = original_key
    
    def test_client_initialization_with_key(self):
        """Test client initializes with API key."""
        if not os.environ.get("TOGETHER_API_KEY"):
            pytest.skip("TOGETHER_API_KEY not set")
        
        client = TogetherAIClient()
        assert client.model == TogetherAIClient.DEFAULT_MODEL
        assert client.temperature == 0.3
    
    def test_build_prompt(self):
        """Test prompt building."""
        if not os.environ.get("TOGETHER_API_KEY"):
            pytest.skip("TOGETHER_API_KEY not set")
        
        client = TogetherAIClient()
        
        candidates = [
            MovieCandidate(
                movie_id=1,
                title="The Matrix (1999)",
                genres=["Action", "Sci-Fi"],
                year=1999,
                gnn_score=0.95,
                gnn_rank=1,
            ),
        ]
        user_context = UserContext(
            user_id=42,
            liked_movies=["Inception (2010)"],
            disliked_movies=[],
        )
        
        prompt = client._build_prompt(candidates, user_context, top_k=5)
        
        assert "The Matrix (1999)" in prompt
        assert "ID 1" in prompt
        assert "Inception (2010)" in prompt
        assert "User #42" in prompt


@pytest.mark.integration
class TestTogetherAIIntegration:
    """Integration tests that call the actual Together AI API."""
    
    def test_rerank_movies(self):
        """Test actual reranking call to Together AI."""
        if not os.environ.get("TOGETHER_API_KEY"):
            pytest.skip("TOGETHER_API_KEY not set")
        
        client = TogetherAIClient(
            temperature=0.1,  # Use default model (Llama-3.1-8B)
        )
        
        candidates = [
            MovieCandidate(
                movie_id=1,
                title="The Matrix (1999)",
                genres=["Action", "Sci-Fi"],
                year=1999,
                gnn_score=0.95,
                gnn_rank=1,
            ),
            MovieCandidate(
                movie_id=2,
                title="Titanic (1997)",
                genres=["Drama", "Romance"],
                year=1997,
                gnn_score=0.90,
                gnn_rank=2,
            ),
            MovieCandidate(
                movie_id=3,
                title="Blade Runner (1982)",
                genres=["Sci-Fi", "Thriller"],
                year=1982,
                gnn_score=0.85,
                gnn_rank=3,
            ),
        ]
        
        user_context = UserContext(
            user_id=42,
            liked_movies=[
                "Inception (2010)",
                "Interstellar (2014)",
                "The Dark Knight (2008)",
            ],
            disliked_movies=["The Notebook (2004)"],
        )
        
        response = client.rerank(candidates, user_context, top_k=3)
        
        # Validate response structure
        assert isinstance(response, RerankerResponse)
        assert len(response.recommendations) > 0
        assert all(isinstance(r, RerankedMovie) for r in response.recommendations)
        
        # Check that movie IDs are from candidates
        candidate_ids = {c.movie_id for c in candidates}
        for rec in response.recommendations:
            assert rec.movie_id in candidate_ids
            assert rec.rank >= 1
            assert len(rec.reasoning) > 0
        
        print("\n=== Reranking Results ===")
        for rec in response.recommendations:
            print(f"Rank {rec.rank}: Movie {rec.movie_id} - {rec.reasoning}")

