"""Together AI client for LLM-based reranking."""

import json
import logging
import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
import together

# Load .env file from project root
load_dotenv(Path(__file__).parents[3] / ".env")

from recommender.reranker.schemas import (
    MovieCandidate,
    UserContext,
    RerankedMovie,
    RerankerResponse,
)

logger = logging.getLogger(__name__)

# Simple JSON schema for Together AI's json_schema mode
RERANKER_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "recommendations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "movie_id": {"type": "integer"},
                    "rank": {"type": "integer"},
                    "reasoning": {"type": "string"}
                },
                "required": ["movie_id", "rank", "reasoning"]
            }
        }
    },
    "required": ["recommendations"]
}


class TogetherAIClient:
    """Client for Together AI's official SDK."""
    
    # Note: Llama-3.2-3B doesn't support json_schema mode, use 3.1-8B
    DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ):
        """
        Initialize Together AI client.
        
        Args:
            model: Model name from Together AI's model library
            api_key: API key (defaults to TOGETHER_API_KEY env var)
            temperature: Sampling temperature (lower = more deterministic)
            max_tokens: Maximum tokens in response
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        api_key = api_key or os.environ.get("TOGETHER_API_KEY")
        if not api_key:
            raise ValueError(
                "TOGETHER_API_KEY environment variable not set. "
                "Get your API key at https://api.together.ai/"
            )
        
        self.client = together.Together(api_key=api_key)
    
    def rerank(
        self,
        candidates: List[MovieCandidate],
        user_context: UserContext,
        top_k: int = 10,
    ) -> RerankerResponse:
        """
        Rerank movie candidates based on user preferences.
        
        Args:
            candidates: List of movie candidates from GNN
            user_context: User's rating history context
            top_k: Number of movies to return in reranked list
            
        Returns:
            RerankerResponse with reranked movies and reasoning
        """
        prompt = self._build_prompt(candidates, user_context, top_k)
        
        logger.info(f"Calling {self.model} for reranking {len(candidates)} candidates")


        # GET AI RESPONSE
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={
                "type": "json_schema",
                "schema": RERANKER_JSON_SCHEMA,
            },
        )

        # Parse response
        content = response.choices[0].message.content
        logger.debug(f"Raw response: {content}")
        
        # Parse and validate response
        try:
            data = json.loads(content)
            return RerankerResponse.model_validate(data)
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse AI response: {e}")
            logger.error(f"Response content: {content}")
            raise ValueError(f"AI returned invalid response: {e}") from e


    
    
    def _get_system_prompt(self) -> str:
        """System prompt for the reranker."""
        return """You are a movie recommendation expert. Your task is to rerank movie 
recommendations based on a user's viewing history and preferences.

You will receive:
1. A list of candidate movies with their metadata (title, genres, year, GNN score)
2. The user's liked movies (rated >= 4 stars)
3. The user's disliked movies (rated <= 2 stars)

Analyze the user's preferences and rerank the candidates to best match their taste.
Consider genre preferences, time periods, and patterns in their ratings.
Provide brief reasoning for each recommendation, ordered by rank (1 = best match)."""



    def _build_prompt(
        self,
        candidates: List[MovieCandidate],
        user_context: UserContext,
        top_k: int,
    ) -> str:
        """Build the user prompt with candidates and context."""

        # Prepare movie candidates.
        candidates_text = "## Candidate Movies to Rerank:\n\n"
        for c in candidates:
            genres_str = ", ".join(c.genres) if c.genres else "Unknown"
            year_str = str(c.year) if c.year else "Unknown"
            candidates_text += (
                f"- **ID {c.movie_id}**: {c.title}\n"
                f"  Genres: {genres_str} | Year: {year_str} | GNN Score: {c.gnn_score:.3f}\n\n"
            )
        
        # Format user context
        user_text = f"## User #{user_context.user_id} Preferences:\n\n"
        
        if user_context.liked_movies:
            user_text += "**Liked Movies (>= 4 stars):**\n"
            for movie in user_context.liked_movies:
                user_text += f"- {movie}\n"
            user_text += "\n"
        else:
            user_text += "**Liked Movies:** No high ratings available\n\n"
        print(user_text)
        
        if user_context.disliked_movies:
            user_text += "**Disliked Movies (<= 2 stars):**\n"
            for movie in user_context.disliked_movies[:10]:
                user_text += f"- {movie}\n"
            user_text += "\n"
        
        instruction = f"\n## Task:\nRerank these candidates and return the top {top_k} movies that best match this user's preferences. Respond with JSON only."
        
        return candidates_text + user_text + instruction

