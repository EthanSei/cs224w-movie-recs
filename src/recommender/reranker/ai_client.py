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
    # DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    # DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
    # DEFAULT_MODEL = "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
    # DEFAULT_MODEL = "Qwen/Qwen2.5-72B-Instruct-Turbo"
    # DEFAULT_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    # DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct-Turbo"


    # DEFAULT_MODEL = "meta-llama/Meta-Llama-3.3-70B-Instruct-Turbo"  # unavailable
    # DEFAULT_MODEL = "Salesforce/Llama-Rank-v1" # does not support json_schema
    # DEFAULT_MODEL = "mixedbread-ai/Mxbai-Rerank-Large-V2" # does not support json_schema

    # THREE TO ACTUALLY TEST FOR FULL RUNS
    # DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
    # DEFAULT_MODEL = "Qwen/Qwen2.5-72B-Instruct-Turbo"
    # DEFAULT_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    DEFAULT_MODEL = "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
    
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
        return """You are a movie recommendation expert helping to refine recommendations from a Graph Neural Network (GNN) model.

IMPORTANT CONTEXT:
- The GNN model has learned from millions of user-movie interactions using collaborative filtering
- GNN scores and ranks capture patterns like "users who liked X also liked Y"
- Higher GNN scores indicate stronger collaborative filtering signals
- The GNN's top-ranked movies are already strong candidates

YOUR ROLE:
You should make TARGETED ADJUSTMENTS to the GNN ranking, not completely reorder it.
- PROMOTE movies that strongly match the user's genre/era preferences AND have decent GNN scores
- DEMOTE movies that clash with user preferences (wrong genres, disliked themes)
- TRUST the GNN ranking for movies where you have no strong signal either way
- Keep most top GNN-ranked movies in your top selections unless there's a clear mismatch

Think of yourself as a refinement layer, not a replacement for the GNN."""



    def _build_prompt(
        self,
        candidates: List[MovieCandidate],
        user_context: UserContext,
        top_k: int,
    ) -> str:
        """Build the user prompt with candidates and context."""

        # Prepare movie candidates with GNN rank prominently displayed
        candidates_text = "## Candidate Movies (ordered by GNN collaborative filtering rank):\n\n"
        for c in candidates:
            genres_str = ", ".join(c.genres) if c.genres else "Unknown"
            year_str = str(c.year) if c.year else "Unknown"
            candidates_text += (
                f"**GNN Rank #{c.gnn_rank}** - ID {c.movie_id}: {c.title}\n"
                f"  Genres: {genres_str} | Year: {year_str} | GNN Score: {c.gnn_score:.3f}\n\n"
            )
        
        # Format user context
        user_text = f"## User #{user_context.user_id} Viewing History:\n\n"
        
        if user_context.liked_movies:
            user_text += "**Movies they loved (4-5 stars):**\n"
            for movie in user_context.liked_movies:
                user_text += f"- {movie}\n"
            user_text += "\n"
        else:
            user_text += "**Liked Movies:** Limited rating history available\n\n"
        
        if user_context.disliked_movies:
            user_text += "**Movies they disliked (1-2 stars):**\n"
            for movie in user_context.disliked_movies[:10]:
                user_text += f"- {movie}\n"
            user_text += "\n"
        
        instruction = f"""## Task:
Select the top {top_k} movies for this user from the candidates above.

GUIDELINES:
- The GNN's top-10 movies are VERY STRONG candidates backed by collaborative filtering
- Make SMALL adjustments (±3-5 positions) based on clear genre/era mismatches
- ONLY promote lower-ranked movies if they STRONGLY match AND have decent GNN scores (>15th percentile)
- ONLY demote top-ranked movies if they CLEARLY clash (wrong genre + user dislikes similar movies)
- When uncertain, TRUST THE GNN RANKING - it knows patterns you don't see
- Aim for subtle refinement, not complete reordering

Return your top {top_k} selections as JSON with movie_id, rank (1-{top_k}), and brief reasoning."""
        
        return candidates_text + user_text + instruction



# GUIDELINES:
# - The GNN's top-10 movies are VERY STRONG candidates backed by collaborative filtering
# - Make SMALL adjustments (±3-5 positions) based on clear genre/era mismatches
# - ONLY promote lower-ranked movies if they STRONGLY match AND have decent GNN scores (>15th percentile)
# - ONLY demote top-ranked movies if they CLEARLY clash (wrong genre + user dislikes similar movies)
# - When uncertain, TRUST THE GNN RANKING - it knows patterns you don't see
# - Aim for subtle refinement, not complete reordering