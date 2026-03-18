"""
Scouting-Optimized CRAG Configuration
======================================

Configuration optimized for scouting queries that prioritize internal database
(player stats + tactical articles) over web search.
"""

from typing import Dict, Any
import os


class ScoutingCRAGConfig:
    """Configuration for scouting-focused CRAG system"""
    
    # =============================================================================
    # GRADING OPTIMIZATION (Trust Database More)
    # =============================================================================
    
    # Reduce web search triggers
    TRUST_DATABASE_SEASONS = ["2021-2022", "2022-2023", "2023-2024", "2024-2025", "2025-2026"]
    
    # Keywords that indicate database-answerable queries
    DATABASE_KEYWORDS = [
        # Stats keywords
        "goals", "assists", "tackles", "passes", "shots", "saves",
        "minutes", "appearances", "statistics", "stats",
        # Position keywords
        "striker", "winger", "midfielder", "defender", "goalkeeper",
        "forward", "attacking", "defensive",
        # Tactical keywords  
        "pressing", "possession", "counter", "tiki-taka", "gegenpress",
        "false 9", "inverted", "wingback", "pivot",
        # Comparison keywords
        "compare", "vs", "versus", "better", "top", "best",
        # Age/scouting keywords
        "young", "under", "aged", "potential", "prospect",
    ]
    
    # Keywords that require web search
    WEB_SEARCH_KEYWORDS = [
        "today", "yesterday", "this week", "latest", "breaking",
        "transfer news", "injury", "injured", "suspended",
        "press conference", "interview", "quotes",
        "2026", "2027", "future", "prediction",
        "net worth", "salary", "wage", "contract",
    ]
    
    # =============================================================================
    # MODEL CONFIGURATION (Reduce API Calls)
    # =============================================================================
    
    # Use smaller/cheaper model for non-critical tasks
    GRADING_MODEL = "mixtral-8x7b-32768"  # Faster, cheaper than llama-3.3-70b
    GENERATION_MODEL = "llama-3.3-70b-versatile"  # Keep quality for generation
    
    # Reduce token usage
    MAX_CONTEXT_LENGTH = 400  # From 600 (reduce by 33%)
    MAX_GENERATION_TOKENS = 500  # From 2000 (reduce by 75%)
    
    # =============================================================================
    # RETRIEVAL OPTIMIZATION
    # =============================================================================
    
    # Retrieve more docs initially (better chance of finding answer)
    INITIAL_RETRIEVAL_COUNT = 10  # From 5
    MAX_DOCS_FOR_GRADING = 5  # But only grade top 5
    MAX_DOCS_FOR_GENERATION = 8  # Send more context to generation
    
    # Season prioritization (prefer recent data)
    SEASON_PRIORITY = {
        "2025-2026": 1.0,
        "2024-2025": 0.95,
        "2023-2024": 0.85,
        "2022-2023": 0.75,
        "2021-2022": 0.65,
    }
    
    # =============================================================================
    # WEB SEARCH CONFIGURATION
    # =============================================================================
    
    # Disable web search for database-answerable queries
    ENABLE_WEB_FALLBACK = True  # Set to False to force database-only
    WEB_SEARCH_THRESHOLD = 0.3  # Only search if confidence < 30%
    
    # Limit web searches
    MAX_WEB_RESULTS = 2  # From 5 (reduce API calls)
    
    # =============================================================================
    # REFRAG OPTIMIZATION
    # =============================================================================
    
    # Reduce sub-questions for faster processing
    MAX_SUB_QUESTIONS = 2  # From 4 (reduce LLM calls by 50%)
    REFRAG_MODEL = "mixtral-8x7b-32768"  # Cheaper model for decomposition
    
    # =============================================================================
    # SELF-CHECK OPTIMIZATION
    # =============================================================================
    
    # Relax verification to reduce regeneration
    CONFIDENCE_THRESHOLD = 0.6  # From 0.7 (accept more answers)
    MAX_RETRIES = 1  # From 2 (reduce regeneration attempts)
    SELFCHECK_MODEL = "mixtral-8x7b-32768"  # Cheaper verification
    
    # Skip verification for simple queries
    SKIP_VERIFICATION_FOR = [
        "simple stats queries",  # "How many goals did X score?"
        "single player lookups",  # "Show me X's stats"
    ]
    
    # =============================================================================
    # GRACEFUL DEGRADATION (Handle Rate Limits)
    # =============================================================================
    
    # Fallback strategies when rate limited
    RATE_LIMIT_FALLBACK = "database_only"  # Options: "database_only", "cache", "error"
    
    # Cache expensive operations
    ENABLE_CACHING = True
    CACHE_GRADING_RESULTS = True
    CACHE_WEB_RESULTS = True
    CACHE_TTL_SECONDS = 3600  # 1 hour
    
    # =============================================================================
    # QUERY OPTIMIZATION
    # =============================================================================
    
    @staticmethod
    def optimize_for_database(query: str) -> Dict[str, Any]:
        """
        Analyze query and return optimization hints
        
        Args:
            query: User query
            
        Returns:
            Dict with optimization hints
        """
        query_lower = query.lower()
        
        # Check if query is database-answerable
        has_db_keywords = any(kw in query_lower for kw in ScoutingCRAGConfig.DATABASE_KEYWORDS)
        has_web_keywords = any(kw in query_lower for kw in ScoutingCRAGConfig.WEB_SEARCH_KEYWORDS)
        
        # Determine if we can skip web search
        can_skip_web = has_db_keywords and not has_web_keywords
        
        # Check if query mentions season in our range
        has_valid_season = any(season.lower() in query_lower for season in ScoutingCRAGConfig.TRUST_DATABASE_SEASONS)
        
        # Determine if we can skip verification
        is_simple_query = (
            len(query.split()) < 10 and
            ("goals" in query_lower or "assists" in query_lower or "stats" in query_lower)
        )
        
        return {
            "can_skip_web_search": can_skip_web,
            "has_valid_season": has_valid_season,
            "is_simple_query": is_simple_query,
            "confidence_boost": 0.2 if can_skip_web else 0.0,
            "recommended_model": "mixtral" if is_simple_query else "llama",
            "enable_refrag": not is_simple_query,
            "enable_selfcheck": not is_simple_query,
        }
    
    @staticmethod
    def get_model_config() -> Dict[str, str]:
        """Get model configuration for different tasks"""
        return {
            "grading": ScoutingCRAGConfig.GRADING_MODEL,
            "generation": ScoutingCRAGConfig.GENERATION_MODEL,
            "refrag": ScoutingCRAGConfig.REFRAG_MODEL,
            "selfcheck": ScoutingCRAGConfig.SELFCHECK_MODEL,
        }
    
    @staticmethod
    def should_use_web_search(query: str, docs_count: int, confidence: float) -> bool:
        """
        Determine if web search is needed
        
        Args:
            query: User query
            docs_count: Number of retrieved docs
            confidence: Retrieval confidence
            
        Returns:
            True if web search needed
        """
        if not ScoutingCRAGConfig.ENABLE_WEB_FALLBACK:
            return False
        
        # Check for web keywords
        query_lower = query.lower()
        has_web_keywords = any(kw in query_lower for kw in ScoutingCRAGConfig.WEB_SEARCH_KEYWORDS)
        
        if has_web_keywords:
            return True  # Explicitly needs web
        
        # Check if database has enough data
        if docs_count >= 3 and confidence >= ScoutingCRAGConfig.WEB_SEARCH_THRESHOLD:
            return False  # Database has good data
        
        # Check for database keywords
        has_db_keywords = any(kw in query_lower for kw in ScoutingCRAGConfig.DATABASE_KEYWORDS)
        
        if has_db_keywords:
            return False  # Should be answerable from DB
        
        # Default: no web search for scouting queries
        return False


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("="*80)
    print("SCOUTING-OPTIMIZED CRAG CONFIGURATION")
    print("="*80)
    
    # Test query optimization
    test_queries = [
        "How many goals did Mohamed Salah score in 2024-2025?",
        "Compare Salah and Haaland's performance",
        "What are the latest transfer news for Salah?",
        "Find young strikers under 23 with 10+ goals",
        "What are high pressing tactics?",
    ]
    
    print("\nQuery Optimization Analysis:\n")
    
    for query in test_queries:
        print(f"Query: {query}")
        hints = ScoutingCRAGConfig.optimize_for_database(query)
        print(f"  Can skip web search: {hints['can_skip_web_search']}")
        print(f"  Is simple query: {hints['is_simple_query']}")
        print(f"  Enable REFRAG: {hints['enable_refrag']}")
        print(f"  Enable Self-Check: {hints['enable_selfcheck']}")
        print(f"  Recommended model: {hints['recommended_model']}")
        print()
    
    print("\n" + "-"*80)
    print("MODEL CONFIGURATION (Optimized for Cost & Speed)")

    
    models = ScoutingCRAGConfig.get_model_config()
    for task, model in models.items():
        print(f"{task.capitalize()}: {model}")
    
    print("\n" + "-"*80)
    print("KEY OPTIMIZATIONS")
    print("-"*80)
    print(f" Database keywords: {len(ScoutingCRAGConfig.DATABASE_KEYWORDS)} tracked")
    print(f" Web fallback: {'Enabled' if ScoutingCRAGConfig.ENABLE_WEB_FALLBACK else 'Disabled'}")
    print(f" Confidence threshold: {ScoutingCRAGConfig.CONFIDENCE_THRESHOLD} (relaxed from 0.7)")
    print(f" Max retries: {ScoutingCRAGConfig.MAX_RETRIES} (reduced from 2)")
    print(f"Sub-questions: {ScoutingCRAGConfig.MAX_SUB_QUESTIONS} (reduced from 4)")
    print(f" Token savings: ~60% (context + generation limits)")
