"""Pydantic schemas for AI reranker input/output."""

from typing import List, Optional

from pydantic import BaseModel, Field


class MovieCandidate(BaseModel):
    """A movie candidate from the GNN recommender with metadata."""
    
    movie_id: int = Field(description="Unique movie identifier")
    title: str = Field(description="Movie title including year")
    genres: List[str] = Field(description="List of genre tags")
    year: Optional[int] = Field(default=None, description="Release year if available")
    gnn_score: float = Field(description="Score from the GNN model")
    gnn_rank: int = Field(description="Original rank from GNN (1 = highest)")


class UserContext(BaseModel):
    """User context derived from their rating history."""
    
    user_id: int = Field(description="Unique user identifier")
    liked_movies: List[str] = Field(
        default_factory=list,
        description="Titles of movies the user rated highly (>= 4 stars)"
    )
    disliked_movies: List[str] = Field(
        default_factory=list,
        description="Titles of movies the user rated poorly (<= 2 stars)"
    )


class RerankedMovie(BaseModel):
    """A single reranked movie in the AI response."""
    
    movie_id: int = Field(description="Movie ID from the candidates")
    rank: int = Field(description="New rank after AI reranking (1 = best)")
    reasoning: str = Field(description="Brief explanation for this ranking")


class RerankerResponse(BaseModel):
    """Full response from the AI reranker."""
    
    recommendations: List[RerankedMovie] = Field(
        description="List of reranked movies ordered by new rank"
    )

