"""
Simplified Semantic Cache Wrapper

This module provides a clean, simplified interface for semantic caching.

Example usage:
    from helpers.wrapper import SemanticCacheWrapper
    
    # Create cache wrapper with custom parameters
    cache = SemanticCacheWrapper(
        name="my-cache",
        distance_threshold=0.3,
        ttl=3600
    )
    
    # Hydrate cache from DataFrame
    cache.hydrate_from_df(df, q_col="question", a_col="answer")
    
    # Check cache
    results = cache.check("What is your refund policy?")
    
    # Check multiple queries
    results = cache.check_many(queries, show_progress=True)
"""

import pandas as pd
import redis
from pydantic import BaseModel
from redisvl.extensions.cache.embeddings import EmbeddingsCache
from redisvl.extensions.cache.llm import SemanticCache
from redisvl.utils.vectorize import HFTextVectorizer
from tqdm.auto import tqdm
from typing import List, Optional

REDIS_URL = "redis://localhost:6379"

class CacheResult(BaseModel):
    """
    Standardised result for cache wrapper outputs.
    
    Fields:
    - prompt: cache key text
    - response: cached response text
    - vector_distance: semantic distance from vector index (lower = more similar)
    - cosine_similarity: cosine similarity score (higher = more similar)
    """
    
    prompt: str
    response: str
    vector_distance: float
    cosine_similarity: float


class CacheResults(BaseModel):
    """Container for cache check results."""
    query: str
    matches: List[CacheResult]
    
    def __repr__(self):
        return f"(Query: '{self.query}', Matches: {[m.prompt for m in self.matches]})"


def try_connect_to_redis(redis_url: str):
    """Test Redis connection and return client."""
    try:
        r = redis.Redis.from_url(redis_url)
        r.ping()
        print("Redis is running and accessible!")
    except redis.ConnectionError:
        print(
            """
            Cannot connect to Redis. Please make sure Redis is running on localhost:6379
                Try: docker run -d --name redis -p 6379:6379 redis/redis-stack:latest
            """
        )
        raise
        
    return r


class SemanticCacheWrapper:
    """
    A wrapper around RedisVL SemanticCache that provides:
    - DataFrame-based cache hydration
    - Batch checking with progress bars
    """
    
    def __init__(
        self,
        name: str = "semantic-cache",
        distance_threshold: float = 0.3,
        ttl: int = 3600,
        redis_url: Optional[str] = None,
    ):
        redis_connection_url = redis_url or REDIS_URL
        self.redis = try_connect_to_redis(redis_connection_url)
        
        self.embeddings_cache = EmbeddingsCache(redis_client=self.redis, ttl=ttl * 24)
        self.langcache_embed = HFTextVectorizer(
            model="redis/langcache-embed-v1", cache=self.embeddings_cache
        )
        self.cache = SemanticCache(
            name=name,
            vectorizer=self.langcache_embed,
            redis_client=self.redis,
            distance_threshold=distance_threshold,
            ttl=ttl,
        )
    
    def hydrate_from_df(
        self,
        df: pd.DataFrame,
        *,
        q_col: str = "question",
        a_col: str = "answer",
        clear: bool = True,
        ttl_override: Optional[int] = None,
    ) -> None:
        """
        Populate cache from a DataFrame.
        
        Args:
            df: DataFrame with question and answer columns
            q_col: Name of question column
            a_col: Name of answer column
            clear: Whether to clear existing cache first
            ttl_override: Optional TTL override for these entries
        """
        if clear:
            self.cache.clear()
        for row in df[[q_col, a_col]].itertuples(index=False, name=None):
            q, a = row
            self.cache.store(prompt=q, response=a, ttl=ttl_override)
    
    def check(
        self,
        query: str,
        distance_threshold: Optional[float] = None,
        num_results: int = 1,
    ) -> "CacheResults":
        """
        Check semantic cache for a single query.
        
        Args:
            query: The query string to search for
            distance_threshold: Maximum semantic distance (lower = more similar)
            num_results: Maximum number of results to return
            
        Returns:
            CacheResults object with matches
        """
        candidates = self.cache.check(
            query, distance_threshold=distance_threshold, num_results=num_results
        )
        
        if not candidates:
            return CacheResults(query=query, matches=[])
            
        results: List[CacheResult] = []
        for item in candidates:
            result = dict(item)
            result["vector_distance"] = float(result.get("vector_distance", 0.0))
            result["cosine_similarity"] = float((2 - result["vector_distance"]) / 2)
            results.append(CacheResult(**result))
            
        return CacheResults(query=query, matches=results)
    
    def check_many(
        self,
        queries: List[str],
        distance_threshold: Optional[float] = None,
        show_progress: bool = False,
        num_results: int = 1,
    ) -> List["CacheResults"]:
        """
        Check semantic cache for multiple queries.
        
        Args:
            queries: List of query strings
            distance_threshold: Maximum semantic distance
            show_progress: Whether to show progress bar
            num_results: Maximum results per query
        
        Returns:
            List of CacheResults (maintains query order)
        """
        results: List[CacheResults] = []
        for q in tqdm(queries, disable=not show_progress):
            cache_results = self.check(
                q, distance_threshold, num_results
            )
            results.append(cache_results)
        return results
    
    def store(self, prompt: str, response: str, **kwargs):
        """Store a prompt-response pair in the cache."""
        self.cache.store(prompt=prompt, response=response, **kwargs)
    
    def clear(self):
        """Clear all entries from the cache."""
        self.cache.clear()
