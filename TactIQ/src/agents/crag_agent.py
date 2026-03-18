"""
CRAG Agent (Corrective RAG)
===========================

Implements Corrective Retrieval Augmented Generation with:
- LangGraph state machine workflow
- CRAG grader for retrieval quality assessment
- Tavily web search fallback for insufficient retrievals
- LLM-based answer generation
"""

from typing import Dict, List, Any, Optional, TypedDict, Annotated
import os
from loguru import logger

try:
    from langgraph.graph import StateGraph, END
    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from tavily import TavilyClient
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    logger.warning("CRAG dependencies not installed. Run: pip install langgraph langchain-groq tavily-python")
    DEPENDENCIES_AVAILABLE = False

# Import position-aware prompts for goalkeeper fix
try:
    from src.agents.position_prompts import (
        get_prompt_for_position,
        detect_position_from_query,
        detect_position_from_metadata,
        GOALKEEPER_SCOUT_REPORT_PROMPT,
        DEFENDER_SCOUT_REPORT_PROMPT,
        MIDFIELDER_SCOUT_REPORT_PROMPT,
        FORWARD_SCOUT_REPORT_PROMPT
    )
    POSITION_PROMPTS_AVAILABLE = True
    logger.info("Position-aware prompts loaded (Goalkeeper fix enabled)")
except ImportError:
    POSITION_PROMPTS_AVAILABLE = False
    logger.warning("Position prompts not available - using generic template")


class CRAGState(TypedDict):
    """State for CRAG workflow"""
    query: str
    retrieved_docs: List[Dict[str, Any]]
    grade: str  # "sufficient", "insufficient", "partial"
    web_results: Optional[List[Dict[str, Any]]]
    final_answer: str
    sources: List[str]
    confidence: float
    intent: Optional[Any]  # QueryIntent enum
    intent_metadata: Dict[str, Any]
    reasoning_trace: Optional[str]  # Explanation of routing decision


class CRAGAgent:
    """
    Corrective RAG Agent with LangGraph workflow
    
    Workflow:
    1. Retrieve documents from vector DB
    2. Grade retrieval quality (sufficient/insufficient/partial)
    3. If insufficient -> Fallback to Tavily web search
    4. Generate answer from retrieved/web content
    """
    
    def __init__(self, vector_db, groq_api_key: str = None, tavily_api_key: str = None):
        """
        Initialize CRAG agent
        
        Args:
            vector_db: ChromaDB collection
            groq_api_key: Groq API key for LLaMA-3-8B
            tavily_api_key: Tavily API key for web search
        """
        if not DEPENDENCIES_AVAILABLE:
            raise ImportError("CRAG dependencies not installed")
        
        self.vector_db = vector_db
        
        # Initialize LLM (LLaMA-3.1-8B via Groq)
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            logger.warning("GROQ_API_KEY not set. Grading will be unavailable.")
            self.llm = None
        else:
            self.llm = ChatGroq(
                api_key=self.groq_api_key,
                model="llama-3.1-8b-instant",  # LLaMA-3.1-8B (replaces decommissioned llama3-8b-8192)
                temperature=0,
                max_tokens=1200  # Limit output tokens to stay under Groq's 6000 TPM limit
            )
        
        # Initialize Tavily client
        self.tavily_api_key = tavily_api_key or os.getenv("TAVILY_API_KEY")
        if not self.tavily_api_key:
            logger.warning("TAVILY_API_KEY not set. Web search fallback unavailable.")
            self.tavily = None
        else:
            self.tavily = TavilyClient(api_key=self.tavily_api_key)
        
        # Build LangGraph workflow
        self.workflow = self._build_workflow()
        logger.info("CRAG Agent initialized with LangGraph workflow")
    
    def _build_workflow(self) -> StateGraph:
        """Build LangGraph state machine"""
        workflow = StateGraph(CRAGState)
        
        # Add nodes
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("grade", self._grade_node)
        workflow.add_node("web_search", self._web_search_node)
        workflow.add_node("generate", self._generate_node)
        
        # Add edges
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "grade")
        
        # Conditional routing after grading
        workflow.add_conditional_edges(
            "grade",
            self._route_after_grade,
            {
                "generate": "generate",
                "web_search": "web_search"
            }
        )
        
        workflow.add_edge("web_search", "generate")
        workflow.add_edge("generate", END)
        
        return workflow.compile()
    
    def _extract_multiple_players(self, query: str) -> list:
        """Extract multiple player names from comparison query"""
        import re
        
        # Common patterns: "X vs Y", "X or Y", "compare X and Y"
        players = []
        
        # Split on common separators
        for separator in [' vs ', ' versus ', ' or ', ' and ', ', ']:
            if separator in query.lower():
                parts = re.split(separator, query, flags=re.IGNORECASE)
                for part in parts:
                    # Extract capitalized words (player names)
                    words = part.split()
                    name_parts = []
                    for word in words:
                        if word and word[0].isupper() and len(word) > 1:
                            if word.lower() not in ['compare', 'who', 'which', 'is', 'the', 'a', 'an']:
                                name_parts.append(word)
                    if name_parts:
                        players.append(' '.join(name_parts))
        
        return list(set(players))[:5]  # Max 5 players
    
    def _retrieve_comparison(self, players: list, original_query: str) -> list:
        """Retrieve documents for multiple players for comparison"""
        all_docs = []
        seen_ids = set()
        
        # Extract season from query if specified
        requested_season = None
        import re
        season_match = re.search(r'(202[0-9]-202[0-9])', original_query)
        if season_match:
            requested_season = season_match.group(1)
        else:
            # Default to current season if not specified
            requested_season = "2025-2026"
        
        logger.info(f" Comparison query - retrieving for players: {players} in season: {requested_season}")
        
        for player in players:
            # Search for this specific player with season filter
            try:
                results = self.vector_db.query(
                    query_texts=[f"{player} stats {requested_season}"],
                    n_results=10,
                    where={'season': requested_season} if requested_season else None
                )
                
                if results and results.get('documents'):
                    player_docs = []
                    for i, doc in enumerate(results['documents'][0]):
                        metadata = results['metadatas'][0][i] if results.get('metadatas') else {}
                        
                        # Verify this is the correct player (name matching)
                        doc_player = metadata.get('player', '').lower()
                        if player.lower() not in doc_player and doc_player not in player.lower():
                            continue  # Skip wrong player
                        
                        metadata['comparison_player'] = player
                        
                        player_docs.append({
                            'content': doc,
                            'metadata': metadata,
                            'similarity': results['distances'][0][i] if results.get('distances') else 0,
                            'season': metadata.get('season', '0000-0000')
                        })
                    
                    # Sort by season (descending) and module to get diverse stats
                    player_docs.sort(key=lambda x: (x['season'], x['metadata'].get('stat_module', '')), reverse=True)
                    
                    # Take top 3 docs for this player (identity + shooting + passing ideally)
                    for doc in player_docs[:3]:
                        doc_id = f"{player}_{doc['season']}_{doc['metadata'].get('stat_module', '')}"
                        if doc_id not in seen_ids:
                            all_docs.append(doc)
                            seen_ids.add(doc_id)
                            logger.info(f"  ✓ Added {player} - {doc['metadata'].get('stat_module', 'unknown')} module")
            except Exception as e:
                logger.error(f"Error retrieving docs for {player}: {e}")
        
        logger.info(f"Total docs retrieved for comparison: {len(all_docs)}")
        
        # Validate we have docs for BOTH players (minimum requirement for comparison)
        if len(all_docs) < 2:
            logger.warning(f"Comparison failed: Only {len(all_docs)} player(s) found in database")
            # Add a special marker to indicate comparison failure
            if len(all_docs) == 0:
                all_docs.append({
                    'content': f"No data found for {' or '.join(players)}. Players may not exist in database for season {requested_season}.",
                    'metadata': {'comparison_failed': True, 'reason': 'no_data'},
                    'similarity': 1.0,
                    'season': requested_season
                })
        
        return all_docs
    
    def _build_workflow(self) -> StateGraph:
        """Build LangGraph state machine"""
        workflow = StateGraph(CRAGState)
        
        # Add nodes
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("grade", self._grade_node)
        workflow.add_node("web_search", self._web_search_node)
        workflow.add_node("generate", self._generate_node)
        
        # Add edges
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "grade")
        
        # Conditional routing after grading
        workflow.add_conditional_edges(
            "grade",
            self._route_after_grade,
            {
                "generate": "generate",
                "web_search": "web_search"
            }
        )
        
        workflow.add_edge("web_search", "generate")
        workflow.add_edge("generate", END)
        
        return workflow.compile()
    
    def _retrieve_node(self, state: CRAGState) -> CRAGState:
        """Node: Retrieve documents from vector DB with MODULE-AWARE retrieval"""
        query = state["query"]
        logger.info(f"Retrieving documents for: {query}")
        
        # STEP 1: Detect query intent (which stat modules to retrieve)
        query_lower = query.lower()
        
        # Module keywords mapping
        module_keywords = {
            'shooting': ['goal', 'shot', 'finish', 'score', 'xg', 'striker', 'forward', 'npxg'],
            'passing': ['pass', 'assist', 'creative', 'chance', 'playmaker', 'link', 'xag', 'key pass'],
            'progression': ['dribble', 'carry', 'progress', 'touch', 'beat', 'run', 'take-on'],
            'defensive': ['tackle', 'press', 'intercept', 'block', 'defend', 'win back', 'defensive'],
            'goalkeeper': ['goalkeeper', 'save', 'clean sheet', 'distribution', 'gk', 'shot-stopping']
        }
        
        # Detect relevant modules
        relevant_modules = ['identity']  # Always include identity
        for module, keywords in module_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                relevant_modules.append(module)
        
        # If no specific module detected, include all outfield modules (general scout report)
        if len(relevant_modules) == 1:  # Only identity
            relevant_modules.extend(['shooting', 'passing', 'progression', 'defensive'])
        
        logger.info(f" Query intent detected - retrieving modules: {relevant_modules}")
        
        # Check if this is a comparison query
        import re
        query_lower = query.lower()
        is_comparison = any(word in query_lower for word in ['compare', 'vs', 'versus', 'better than', 'or'])
        
        if is_comparison:
            # Extract player names for comparison
            players = self._extract_multiple_players(query)
            if len(players) >= 2:
                logger.info(f"Comparison query detected: {players}")
                docs = self._retrieve_comparison(players, query)
                state["retrieved_docs"] = docs
                logger.info(f"Retrieved {len(docs)} documents for comparison")
                return state
        
        # Extract specific player name if mentioned (for single player queries)
        player_name = None
        
        # Strategy 0: Check if query is JUST a name (1-3 words)
        query_words = query.strip().split()
        if 1 <= len(query_words) <= 3:
            # Check if these look like player names (no common query words)
            common_words = {'scout', 'report', 'analysis', 'player', 'stats', 'show', 'find', 
                          'compare', 'vs', 'versus', 'how', 'was', 'this', 'season', 'generate',
                          'analyze', 'evaluate', 'assess', 'for', 'the', 'a', 'an', 'in', 'on'}
            cleaned_words = [w for w in query_words if w.lower() not in common_words]
            
            if len(cleaned_words) >= 1:
                # Likely a player name
                potential_name = ' '.join(cleaned_words)
                # Capitalize properly for database matching
                player_name = ' '.join(word.capitalize() for word in cleaned_words)
                logger.info(f" Detected direct player name query: {player_name}")
        
        # Strategy 1: Try known players database first (handles nicknames and variations like "allison" -> "Alisson Becker")
        # ALWAYS check known players to avoid ambiguous last-name matches (e.g., "Alisson Becker" vs "Sheraldo Becker")
        if True:  # Always run this check for better disambiguation
            from src.agents.position_prompts import KNOWN_PLAYERS
            query_lower = query.lower()
            
            # Try exact substring matching first
            best_match = None
            best_match_score = 0
            best_match_position = None
            
            # Check if query mentions position (helps with ambiguous names like "Becker")
            query_lower = query.lower()
            position_hint = None
            if 'goalkeeper' in query_lower or 'gk' in query_lower or 'keeper' in query_lower:
                position_hint = 'gk'
            elif 'defender' in query_lower or 'defence' in query_lower or 'df' in query_lower or 'defense' in query_lower:
                position_hint = 'df'
            elif 'midfielder' in query_lower or 'midfield' in query_lower or 'mf' in query_lower or 'attacking midfielder' in query_lower:
                position_hint = 'mf'
            elif 'forward' in query_lower or 'striker' in query_lower or 'attacker' in query_lower or 'fw' in query_lower:
                position_hint = 'fw'
            
            # Stopwords to exclude from matching (common query terms)
            stopwords = {
                'scout', 'report', 'analysis', 'season', 'performance', 'stats', 
                'compare', 'comparison', 'versus', 'worth', 'latest', 'news',
                'transfer', 'this', 'that', 'the', 'and', 'for', 'from'
            }
            
            for position, players in KNOWN_PLAYERS.items():
                for known_player in players:
                    known_lower = known_player.lower()
                    
                    # Simple substring matching with scoring
                    name_parts = known_lower.split()
                    match_score = 0
                    matched_parts = set()  # Track which parts of the name matched
                    
                    # Split query into words (avoid matching substrings like "son" in "season")
                    query_words = set(query_lower.split())
                    
                    for part in name_parts:
                        if len(part) >= 3:  # Consider parts with 3+ chars
                            # Only match if player name part is a standalone word in query
                            if part in query_words:
                                match_score += len(part) * 3  # Whole word match = higher score
                                matched_parts.add(part)
                    
                    # Also check if query words match player name (exclude stopwords)
                    query_words_lower = query_lower.split()
                    for word in query_words_lower:
                        if len(word) >= 3 and word not in stopwords:
                            for part in name_parts:
                                if len(part) >= 3:
                                    # Exact match = highest score
                                    if word == part:
                                        match_score += len(word) * 3
                                        matched_parts.add(part)
                                    # Prefix match for misspellings (e.g., "allison" starts with "alisson"[:7])
                                    elif word[:min(5, len(word), len(part))] == part[:min(5, len(word), len(part))] and abs(len(word) - len(part)) <= 1:
                                        match_score += len(word) * 2
                                        matched_parts.add(part)
                                    # Last name exact match (e.g., "becker" matches "Becker")
                                    elif word == part and len(word) >= 4:
                                        match_score += len(word) * 2  # REDUCED from 4 to 2
                                        matched_parts.add(part)
                    
                    # CRITICAL FIX: Bonus for matching multiple name parts (first + last)
                    # This ensures "Alisson Becker" beats "Sheraldo Becker" when query is "allison becker"
                    if len(matched_parts) >= 2:
                        match_score += 20  # Big bonus for matching multiple parts
                    
                    # Position-based bonus: If query mentions a position and player matches that position
                    if position_hint and position == position_hint:
                        match_score += 15  # Strong bonus for position match (helps with ambiguous surnames)
                    
                    # Tie-breaker: For ambiguous surnames, prefer more famous players
                    # Goalkeepers are typically more unique and requested
                    if position == 'gk':
                        match_score += 3
                    
                    if match_score > best_match_score:
                        best_match_score = match_score
                        best_match = known_player
                        best_match_position = position
            
            # CRITICAL FIX: Only lock to player if VERY confident
            # Scores 12-50 are moderate confidence and can cause wrong player matches
            # Only lock if: score >= 70 (high confidence) OR it's an exact name match
            if best_match:
                query_lower = query.lower()
                is_exact_match = best_match.lower() in query_lower
                is_high_confidence = best_match_score >= 70
                
                if is_exact_match or is_high_confidence:
                    player_name = best_match
                    match_reason = "exact match" if is_exact_match else f"high-confidence fuzzy match (score: {best_match_score})"
                    logger.info(f" ✓ Matched known player: {best_match} ({match_reason})")
                elif best_match_score >= 12:
                    # Moderate confidence match (score 12-70) - log but DON'T lock
                    # This prevents "Hugo Ekitike" from being locked to "Hugo Lloris" (score 24)
                    logger.info(f"Moderate fuzzy match: {best_match} (score: {best_match_score}) - skipping lock for accuracy")
        
        # Strategy 2: Regex pattern for capitalized names (if not found in known players)
        if not player_name:
            name_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'
            name_matches = re.findall(name_pattern, query)
            if name_matches:
                player_name = name_matches[0]
                logger.info(f" Detected player name from capitalization: {player_name}")
        
        # Store detected player in state for later use (e.g., club lock)
        if player_name:
            state['detected_player'] = player_name
            
            # FUZZY NAME MATCHING: Try database with accent variations
            # Convert "Jeremy Doku" -> try ["Jeremy Doku", "Jérémy Doku", "Jeremy Doku"]
            # This handles cases where user types without accents but DB has accents
            import unicodedata
            
            def remove_accents(text):
                """Remove accents/diacritics from text for fuzzy matching"""
                return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
            
            # Create accent-free version for fuzzy search
            player_name_no_accents = remove_accents(player_name)
            state['player_name_no_accents'] = player_name_no_accents
            logger.info(f"🔍 Fuzzy matching: '{player_name}' (also trying without accents: '{player_name_no_accents}')")
        
        # Extract filters from query
        import re
        query_lower = query.lower()
        where_filter = {}
        target_seasons = []
        
        # Position filter (explicit from UI: "position:GK", "position:DF", etc.)
        position_match = re.search(r'position:(GK|DF|MF|FW)', query, re.IGNORECASE)
        if position_match:
            position_filter = position_match.group(1).upper()
            where_filter["pos"] = position_filter
            logger.info(f" Position filter applied: {position_filter}")
            # Remove the position filter from query for cleaner processing
            query = re.sub(r'\s*position:(GK|DF|MF|FW)\s*', ' ', query, flags=re.IGNORECASE).strip()
            query_lower = query.lower()
        
        # Season filter - MOST IMPORTANT for accuracy
        season_match = re.search(r'(\d{4})[/-](\d{4})', query)  # Match "2024-2025" or "2024/2025"
        requested_specific_season = False
        requested_season_value = None
        if season_match:
            target_season = f"{season_match.group(1)}-{season_match.group(2)}"
            requested_specific_season = True
            requested_season_value = target_season
            target_seasons = [target_season, "2025-2026", "2024-2025", "2023-2024"]  # Try requested + recent seasons
            logger.info(f"Detected season filter: {target_season}, will try fallback to recent seasons")
        elif any(word in query_lower for word in ['this season', 'current season']):
            target_seasons = ["2025-2026", "2024-2025", "2023-2024"]  # Current is 2025-2026
            logger.info(f"Detected 'this season' query, trying: {target_seasons}")
        elif 'last season' in query_lower:
            target_seasons = ["2024-2025", "2023-2024", "2022-2023"]
            logger.info(f"Detected 'last season' query, trying: {target_seasons}")
        
        # Age filter
        age_match = re.search(r'under (\d+)', query_lower)
        if age_match:
            max_age = int(age_match.group(1))
            logger.info(f"Detected age filter: under {max_age}")
        
        # Position filter
        if any(word in query_lower for word in ['striker', 'forward']):
            where_filter['position'] = {'$in': ['FW', 'FW,MF']}
        elif 'midfielder' in query_lower:
            where_filter['position'] = {'$in': ['MF', 'MF,FW', 'MF,DF']}
        elif 'defender' in query_lower:
            where_filter['position'] = {'$in': ['DF', 'DF,MF']}
        
        # Try to query with season filters (fallback to no filter if nothing found)
        results = None
        successful_season = None  # Track which season worked
        for attempt_season in (target_seasons if target_seasons else [None]):
            # Build ChromaDB where filter with proper $and operator
            query_where_filter = None
            
            if where_filter or attempt_season or player_name:
                filter_conditions = []
                
                # CRITICAL FIX: Only lock to player if we're CONFIDENT about the match
                # If score < 12, it's a low-confidence match - skip the lock and let semantic search handle it
                # This prevents "Hugo Ekitike" from being locked to "Hugo Lloris"
                if player_name:
                    # FUZZY MATCHING: Don't use strict WHERE filter for player names
                    # ChromaDB WHERE requires exact match, which fails for accents
                    # Instead, rely on semantic search + post-filtering
                    # ONLY use WHERE filter if we have HIGH confidence from KNOWN_PLAYERS
                    
                    use_strict_filter = best_match_score >= 70  # Only lock if confident match from KNOWN_PLAYERS
                    
                    if use_strict_filter and best_match:
                        # High confidence from known players - use WHERE filter
                        name_parts = player_name.split()
                        player_name_variants = [player_name]  # Full name first
                        if len(name_parts) > 1:
                            player_name_variants.append(name_parts[0])  # First name only (e.g., "Alisson")
                        
                        # Use $or to match either full name OR first name
                        if len(player_name_variants) > 1:
                            filter_conditions.append({'$or': [{'player': variant} for variant in player_name_variants]})
                            logger.info(f"✓ Strict WHERE filter: {player_name} (high-confidence from KNOWN_PLAYERS)")
                        else:
                            filter_conditions.append({'player': player_name})
                            logger.info(f"✓ Strict WHERE filter: {player_name} (high-confidence from KNOWN_PLAYERS)")
                    else:
                        # Low/medium confidence - use semantic search + post-filtering
                        # Don't add WHERE filter, but enhance query with player name
                        logger.info(f" Semantic search mode: '{player_name}' (will post-filter results for fuzzy name match)")
                else:
                    # No player locked - let vector DB do semantic search
                    logger.info(f" No player detected - using semantic search")
                
                # Add position filter if present
                if 'pos' in where_filter:
                    filter_conditions.append({'pos': where_filter['pos']})
                
                # Add position filter (old key) if present
                if 'position' in where_filter:
                    filter_conditions.append({'position': where_filter['position']})
                
                # Add season filter
                if attempt_season:
                    filter_conditions.append({'season': attempt_season})
                    logger.info(f"Attempting retrieval with season: {attempt_season}")
                
                # Construct proper where clause
                if len(filter_conditions) == 1:
                    query_where_filter = filter_conditions[0]
                elif len(filter_conditions) > 1:
                    query_where_filter = {'$and': filter_conditions}
                
                if not attempt_season:
                    logger.info(f"Attempting retrieval WITHOUT season filter")
            
            # Enhance query text with player name for better semantic matching
            search_query = query
            if player_name and player_name.lower() not in query.lower():
                search_query = f"{player_name} {query}"
                logger.info(f"Enhanced search query: {search_query}")
            
            results = self.vector_db.query(
                query_texts=[search_query],
                n_results=50 if player_name else 15,  # Get more docs for player-specific queries
                where=query_where_filter
            )
            
            # Check if we got good results
            if results and results.get('documents') and results['documents'][0]:
                doc_count = len(results['documents'][0])
                logger.info(f"Retrieved {doc_count} documents with season={attempt_season}")
                if doc_count >= 3:  # Good enough
                    successful_season = attempt_season  # Save the season that worked
                    break
            
            # If no season filter specified and we got nothing, break
            if not target_seasons:
                break
        
        # MODULE-AWARE FILTERING: After initial retrieval, filter by stat_module
        if results and results.get('documents') and results['documents'][0]:
            logger.info(f" Filtering {len(results['documents'][0])} docs by modules: {relevant_modules}")
            
            filtered_docs = []
            filtered_metas = []
            filtered_dists = []
            
            module_counts = {}  # Track how many of each module we keep
            
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i] if results.get('metadatas') else {}
                stat_module = metadata.get('stat_module', 'identity')
                chunk_type = metadata.get('chunk_type', '')
                
                # Keep blogs (for tactical context) OR relevant stat modules
                if chunk_type == 'blog_article' or stat_module in relevant_modules:
                    filtered_docs.append(doc)
                    filtered_metas.append(metadata)
                    if results.get('distances'):
                        filtered_dists.append(results['distances'][0][i])
                    
                    # Track module counts
                    if chunk_type == 'blog_article':
                        module_counts['blog'] = module_counts.get('blog', 0) + 1
                    else:
                        module_counts[stat_module] = module_counts.get(stat_module, 0) + 1
            
            # Update results with filtered data
            if filtered_docs:
                # DIVERSITY FIX: Ensure we get docs from ALL modules for player queries
                # If we're querying a specific player and missing modules, do targeted retrieval
                missing_modules = [m for m in relevant_modules if m not in module_counts]
                
                if player_name and missing_modules:
                    logger.warning(f" Missing modules in retrieval: {missing_modules}")
                    logger.info(f" Doing targeted retrieval for missing modules...")
                    
                    # For each missing module, do a targeted query
                    for module in missing_modules:
                        module_query = f"{player_name} {module} stats"
                        try:
                            # Build where filter for module (include season if we have it)
                            module_where = {'stat_module': module}
                            if successful_season:
                                module_where = {'$and': [
                                    {'stat_module': module},
                                    {'season': successful_season}
                                ]}
                                logger.info(f"  Retrieving {module} with season={successful_season}")
                            
                            module_results = self.vector_db.query(
                                query_texts=[module_query],
                                n_results=3,
                                where=module_where
                            )
                            
                            if module_results and module_results.get('documents') and module_results['documents'][0]:
                                # Add module docs to filtered results
                                for i, doc in enumerate(module_results['documents'][0]):
                                    meta = module_results['metadatas'][0][i] if module_results.get('metadatas') else {}
                                    # Only add if same player (fuzzy match - check if names overlap)
                                    doc_player = meta.get('player', '')
                                    player_name_parts = set(player_name.lower().split())
                                    doc_player_parts = set(doc_player.lower().split())
                                    # Match if at least 2 name parts overlap (first + last name typically)
                                    if len(player_name_parts & doc_player_parts) >= 2 or player_name.lower() in doc_player.lower() or doc_player.lower() in player_name.lower():
                                        filtered_docs.append(doc)
                                        filtered_metas.append(meta)
                                        if module_results.get('distances'):
                                            filtered_dists.append(module_results['distances'][0][i])
                                        module_counts[module] = module_counts.get(module, 0) + 1
                                        logger.info(f" Added {module} module doc for {doc_player}")
                        except Exception as e:
                            logger.error(f"Failed to retrieve {module} module: {e}")
                    
                    logger.info(f" After diversity fix - module counts: {module_counts}")
                
                # REORDER for diversity: Ensure first docs include all modules
                # Group docs by module, then interleave
                module_docs = {m: [] for m in relevant_modules}
                for i, meta in enumerate(filtered_metas):
                    mod = meta.get('stat_module', 'identity')
                    if mod in module_docs:
                        module_docs[mod].append((filtered_docs[i], meta, filtered_dists[i] if i < len(filtered_dists) else 0))
                
                # Interleave: pick 1 from each module in rotation
                reordered_docs = []
                reordered_metas = []
                reordered_dists = []
                max_per_module = max(len(docs) for docs in module_docs.values())
                
                for round_num in range(max_per_module):
                    for module in relevant_modules:
                        if round_num < len(module_docs[module]):
                            doc, meta, dist = module_docs[module][round_num]
                            reordered_docs.append(doc)
                            reordered_metas.append(meta)
                            reordered_dists.append(dist)
                
                filtered_docs = reordered_docs
                filtered_metas = reordered_metas
                filtered_dists = reordered_dists
                logger.info(f"Reordered docs for module diversity (first 10 will include all modules)")
                
                # CRITICAL FIX: Limit to 2 docs PER MODULE instead of 3 total
                # This ensures ALL modules (identity, shooting, passing, defensive, progression) are included
                final_docs = []
                final_metas = []
                final_dists = []
                module_doc_counts = {}
                
                for i, meta in enumerate(filtered_metas):
                    mod = meta.get('stat_module', 'identity')
                    count = module_doc_counts.get(mod, 0)
                    
                    # Keep up to 2 docs per module (ensures 10 docs max for 5 modules)
                    if count < 2:
                        final_docs.append(filtered_docs[i])
                        final_metas.append(meta)
                        if i < len(filtered_dists):
                            final_dists.append(filtered_dists[i])
                        module_doc_counts[mod] = count + 1
                
                filtered_docs = final_docs
                filtered_metas = final_metas
                filtered_dists = final_dists
                
                logger.info(f"Final doc count: {len(filtered_docs)} docs across {len(module_doc_counts)} modules: {module_doc_counts}")
                
                # NO MORE ARTIFICIAL LIMITS - keep all module docs!
                results['documents'] = [filtered_docs]
                results['metadatas'] = [filtered_metas]
                if results.get('distances'):
                    results['distances'] = [filtered_dists]
                logger.info(f" Filtered to {len(filtered_docs)} module-relevant docs (max 3 for tokens): {module_counts}")
            else:
                logger.warning(f"No docs matched modules {relevant_modules}, keeping all")
        
        # Format documents and apply additional filters
        docs = []
        if results and results.get('documents'):
            # Helper function for accent-insensitive comparison
            import unicodedata
            def normalize_name(text):
                """Remove accents and normalize to lowercase for comparison"""
                if not text:
                    return ''
                # Remove accents
                text = ''.join(c for c in unicodedata.normalize('NFD', str(text)) if unicodedata.category(c) != 'Mn')
                return text.lower().strip()
            
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i] if results.get('metadatas') else {}
                
                # FUZZY NAME MATCHING: Post-filter results if we didn't use strict WHERE filter
                # This handles accent variations like "Jeremy Doku" vs "Jérémy Doku"
                match_found = True  # Default to True
                
                if player_name and best_match_score < 70:
                    # We're in semantic search mode - need to post-filter for player name
                    doc_player = metadata.get('player') or metadata.get('Player') or metadata.get('name') or ''
                    
                    # Normalize both query name and doc name (remove accents, lowercase)
                    query_name_normalized = normalize_name(player_name)
                    doc_name_normalized = normalize_name(doc_player)
                    
                    match_found = False
                    
                    # Strategy 1: Exact match after normalization (handles accents)
                    if query_name_normalized and doc_name_normalized:
                        if query_name_normalized in doc_name_normalized or doc_name_normalized in query_name_normalized:
                            match_found = True
                            logger.info(f"✓ Fuzzy match: '{player_name}' → '{doc_player}'")
                        
                        # Strategy 2: Last name match (most reliable)
                        query_parts = query_name_normalized.split()
                        doc_parts = doc_name_normalized.split()
                        if len(query_parts) >= 2 and len(doc_parts) >= 2:
                            if query_parts[-1] == doc_parts[-1]:
                                match_found = True
                                logger.info(f"✓ Last name match: '{player_name}' → '{doc_player}'")
                        
                        # Strategy 3: First name match (for unique names)
                        if len(query_parts) >= 1 and len(doc_parts) >= 1:
                            if len(query_parts[0]) >= 4 and query_parts[0] == doc_parts[0]:
                                match_found = True
                                logger.info(f"✓ First name match: '{player_name}' → '{doc_player}'")
                    
                    if not match_found:
                        logger.info(f"✗ Skipping non-matching doc: '{doc_player}' (query: '{player_name}')")
                        continue  # Skip this doc
                else:
                    # Strict WHERE filter was used - trust ChromaDB results
                    pass
                
                # Age filtering (if age constraint in query)
                if age_match:
                    # Try to get age from metadata (stored as string like "18")
                    age_str = metadata.get('age') or metadata.get('Age')
                    if age_str and age_str != 'unknown':
                        try:
                            age = int(age_str)
                            if age >= max_age:
                                continue  # Skip this document
                        except ValueError:
                            # Try parsing format "18-283" if still in that format
                            age_parts = str(age_str).split('-')
                            if age_parts:
                                try:
                                    age = int(age_parts[0])
                                    if age >= max_age:
                                        continue
                                except ValueError:
                                    pass
                
                docs.append({
                    'content': doc,
                    'metadata': metadata,
                    'distance': results['distances'][0][i] if results.get('distances') else 0.5
                })
        
        # Limit to top results after filtering
        # Increase to top 10 to preserve one doc per module (prevents losing progression/docs)
        docs = docs[:10]

        # If user requested a SPECIFIC season, enforce season hard-gating:
        # If no DB documents match the requested season, block generation and return a clear message
        # EXCEPTION: If we used player WHERE filter and got documents, trust them (skip validation)
        # ChromaDB WHERE filter already ensured correct player+season combination
        player_filter_was_used = player_name is not None
        
        if requested_specific_season and not player_filter_was_used:
            seasons_found = set()
            for d in docs:
                meta_season = d.get('metadata', {}).get('season')
                if meta_season:
                    seasons_found.add(meta_season)

            if requested_season_value not in seasons_found:
                # If the exact requested season isn't found, do NOT hard-block generation.
                # Instead, mark retrieval as partial/missing facts but allow generation
                # using available DB documents (better than falling back to web-only).
                logger.warning(f"Requested season {requested_season_value} not found in DB for player; proceeding with available seasons: {sorted(list(seasons_found))}")
                state["grade"] = "context_missing_facts"
                # Add a user-visible note but keep retrieved docs for generation
                state.setdefault("notes", [])
                state["notes"].append(f"Requested season {requested_season_value} not found in DB; using available seasons: {', '.join(sorted(list(seasons_found))) if seasons_found else 'None'}")
                # Do not block generation; return to continue workflow

        state["retrieved_docs"] = docs
        
        # TACTICAL BLOG ENRICHMENT: Add relevant blog articles for context
        # Limit to 2 blogs (instead of 3) and truncate to save tokens while keeping tactical insights
        if player_name and len(docs) > 0:
            # Extract player position for blog query
            first_meta = docs[0].get('metadata', {})
            player_pos = first_meta.get('pos', '')
            
            # Build blog query based on position
            blog_query_terms = []
            if 'MF' in player_pos:
                blog_query_terms = ['midfielder', 'playmaker', 'pressing', 'ball progression', 'midfield']
            elif 'FW' in player_pos:
                blog_query_terms = ['striker', 'forward', 'attacking', 'finishing', 'movement']
            elif 'DF' in player_pos:
                blog_query_terms = ['defender', 'defensive', 'pressing', 'positioning', 'coverage']
            elif 'GK' in player_pos:
                blog_query_terms = ['goalkeeper', 'shot-stopping', 'distribution', 'sweeper-keeper']
            
            if blog_query_terms:
                blog_query = ' '.join(blog_query_terms[:3])  # Use top 3 terms
                try:
                    logger.info(f"Retrieving tactical blogs for context: {blog_query}")
                    blog_results = self.vector_db.query(
                        query_texts=[blog_query],
                        n_results=2,  # Limit to 2 blogs (was 3) for token efficiency
                        where={'chunk_type': 'blog_article'}
                    )
                    
                    if blog_results and blog_results.get('documents') and blog_results['documents'][0]:
                        blog_count = len(blog_results['documents'][0])
                        logger.info(f"Added {blog_count} tactical blog articles for context")
                        
                        # Add blog docs, truncate to 500 chars each to save tokens
                        for i, blog_doc in enumerate(blog_results['documents'][0]):
                            blog_meta = blog_results['metadatas'][0][i] if blog_results.get('metadatas') else {}
                            docs.append({
                                'content': blog_doc[:500],  # Truncate to 500 chars
                                'metadata': blog_meta
                            })
                except Exception as e:
                    logger.warning(f"[WARNING] Blog retrieval failed: {e}")
        
        # Record primary DB club/team for club-lock assertions
        # Use the player-matched doc's team, not just first doc (which might be wrong player)
        if docs:
            primary_db_club = ''
            # Try to use detected player name's team if available
            detected_player = state.get('detected_player') or player_name
            
            # Find first doc matching the detected player name (if any)
            if detected_player:
                for doc in docs:
                    doc_meta = doc.get('metadata', {})
                    doc_player = doc_meta.get('player', '')
                    if doc_player:
                        player_parts = set(detected_player.lower().split())
                        doc_parts = set(doc_player.lower().split())
                        # Match if names overlap (at least 2 parts or substring match)
                        if len(player_parts & doc_parts) >= 2 or detected_player.lower() in doc_player.lower():
                            primary_db_club = doc_meta.get('team') or doc_meta.get('club') or doc_meta.get('Squad') or ''
                            break
            
            # Fallback: use first doc if no player match found
            if not primary_db_club:
                first_meta = docs[0].get('metadata', {})
                primary_db_club = first_meta.get('team') or first_meta.get('club') or first_meta.get('Squad') or ''
            
            state['club_lock'] = primary_db_club
            logger.info(f"DB club lock: {primary_db_club}")

        logger.info(f"Retrieved {len(docs)} documents (after filtering)")
        return state
    
    def _grade_node(self, state: CRAGState) -> CRAGState:
        """Node: Grade retrieval quality using LLM"""
        query = state["query"]
        docs = state["retrieved_docs"]
        
        # If no documents retrieved, immediately mark as missing facts
        if not docs or len(docs) == 0:
            state["grade"] = "context_missing_facts"
            logger.info("Grade: context_missing_facts (no documents retrieved)")
            return state
        
        # OPTIMISTIC GRADING: If we have docs, assume they're sufficient
        # The _route_after_grade will verify has_player_data anyway
        state["grade"] = "context_sufficient"
        logger.info(f"✓ Grade: context_sufficient ({len(docs)} documents found)")
        return state
        
        # (LLM grading removed - optimistic approach with verification in routing)
        return state
    
    def _route_after_grade(self, state: CRAGState) -> str:
        """Conditional routing based on grade"""
        grade = state.get("grade", "context_sufficient")
        docs = state.get("retrieved_docs", [])
        
        # Check if we have player-specific data in retrieved docs
        has_player_data = False
        stats_found = []
        
        if docs:
            for i, doc in enumerate(docs):
                content = doc.get('content', '').lower()
                metadata = doc.get('metadata', {})
                
                # IMPROVED detection: Check for player data or statistical content
                player_name = metadata.get('player', '')
                has_player_meta = bool(player_name)
                
                # Look for statistical/football-specific keywords
                stat_keywords = ['goal', 'assist', 'pass', 'tackle', 'dribble', 'shot', 'xg', 'xga', 
                                'minute', 'season', 'stat', 'performance', 'match', 'player']
                has_stats = any(term in content for term in stat_keywords)
                
                # Also check for position/role indicators (shows player data, not generic)
                position_indicators = ['forward', 'midfielder', 'defender', 'goalkeeper', 'fw', 'mf', 'df', 'gk']
                has_position = any(term in content for term in position_indicators)
                
                if has_player_meta or (has_stats and has_position):
                    has_player_data = True
                    if has_stats:
                        stats_found.append(player_name or f"Doc{i+1}")
                    break
        
        # REASONING TRACE
        reasoning = f"Grade={grade}, DocsCount={len(docs)}, HasPlayerData={has_player_data}"
        if stats_found:
            reasoning += f", StatsFound={stats_found[0]}"
        state["reasoning_trace"] = reasoning
        logger.info(f"Routing Logic: {reasoning}")
        
        # ROUTING DECISION
        if grade == "context_sufficient" and docs:
            logger.info("✓ Using DB data (sufficient grade + documents found)")
            return "generate"
        elif has_player_data:
            logger.info("✓ Using DB data (player data detected despite any grading)")
            return "generate"
        else:
            # Only web search if truly no useful data
            logger.info("Web search needed (no player data in DB)")
            return "web_search"
    
    def _web_search_node(self, state: CRAGState) -> CRAGState:
        """Node: Fallback to Tavily web search"""
        query = state["query"]
        
        if not self.tavily:
            logger.warning("Tavily not available, skipping web search")
            state["web_results"] = []
            return state
        
        logger.info(f"Performing web search for: {query}")
        
        try:
            # Search for recent football news
            search_results = self.tavily.search(
                query=f"{query} football soccer 2024 2025",
                max_results=3
            )
            
            web_docs = []
            for result in search_results.get('results', []):
                web_docs.append({
                    'content': result.get('content', ''),
                    'url': result.get('url', ''),
                    'title': result.get('title', '')
                })
            
            state["web_results"] = web_docs
            logger.info(f"Found {len(web_docs)} web results")
        except Exception as e:
            logger.error(f"Web search error: {e}")
            state["web_results"] = []
        
        return state
    
    def _rank_documents(self, docs: List[Dict], query: str) -> List[Dict]:
        """Rank documents by metrics for 'top/best' queries"""
        import re
        
        query_lower = query.lower()
        
        # Determine ranking metric
        if any(word in query_lower for word in ['striker', 'forward', 'goals', 'scorer']):
            metric = 'goals'
        elif 'assist' in query_lower:
            metric = 'assists'
        elif 'young' in query_lower or 'age' in query_lower:
            metric = 'age'
            reverse = False  # Lower age is better for young players
        else:
            metric = 'goals'  # Default
        
        # Extract metrics from content
        ranked_docs = []
        for doc in docs:
            content = doc.get('content', '')
            metadata = doc.get('metadata', {})
            
            # Try to extract metric value
            value = 0
            
            if metric == 'goals':
                match = re.search(r'(\d+)\s*goals?', content, re.IGNORECASE)
                if match:
                    value = int(match.group(1))
            elif metric == 'assists':
                match = re.search(r'(\d+)\s*assists?', content, re.IGNORECASE)
                if match:
                    value = int(match.group(1))
            elif metric == 'age':
                match = re.search(r'(\d+)-year-old', content)
                if match:
                    value = int(match.group(1))
            
            doc['_rank_value'] = value
            ranked_docs.append(doc)
        
        # Sort (higher is better, except for age in young queries)
        reverse = True if metric != 'age' else False
        ranked_docs.sort(key=lambda x: x.get('_rank_value', 0), reverse=reverse)
        
        return ranked_docs
    
    def _generate_node(self, state: CRAGState) -> CRAGState:
        """Node: Generate final answer"""
        query = state["query"]
        docs = state.get("retrieved_docs", [])
        web_results = state.get("web_results") or []
        grade = state.get("grade", "")

        # If earlier step flagged missing facts that should block generation,
        # return the prepared final_answer immediately.
        if state.get("block_on_missing_facts"):
            logger.info("Generation blocked due to missing DB facts (season/gating)")
            # Ensure sources/confidence are set
            state.setdefault("sources", [])
            state.setdefault("confidence", 0.0)
            return state
        
        # Check if ranking needed
        query_lower = query.lower()
        needs_ranking = any(word in query_lower for word in ['top', 'best', 'elite', 'greatest'])
        
        # Apply ranking if needed
        if needs_ranking and docs:
            docs = self._rank_documents(docs, query)
        
        # Combine all sources with relevance scores
        all_content = []
        sources = []
        seen_sources = set()  # Track unique sources
        
        # Add retrieved docs with relevance
        if docs:
            for doc in docs[:3]:  # Use 3 player docs for token efficiency
                content = doc.get('content', '')
                if content:
                    all_content.append(content)
                    metadata = doc.get('metadata', {})
                    distance = doc.get('distance', 0.5)
                    
                    # Calculate relevance from distance (lower distance = higher relevance)
                    # ChromaDB uses L2 distance, typical range 0-2
                    base_relevance = max(0, min(1, 1 - (distance / 2)))
                    
                    # Boost exact name matches
                    doc_player = str(metadata.get('player', '')).lower()
                    query_lower = query.lower()
                    
                    # Extract player name from query if present
                    import re
                    query_name_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'
                    query_names = re.findall(query_name_pattern, query)
                    
                    relevance = base_relevance
                    if query_names:
                        query_player = query_names[0].lower()
                        # Exact match gets big boost
                        if query_player == doc_player:
                            relevance = min(1.0, base_relevance + 0.3)
                        # Partial match (contains) gets small boost
                        elif query_player in doc_player or doc_player in query_player:
                            relevance = min(1.0, base_relevance + 0.1)
                    
                    # Check if it's a blog/tactical article or player stats
                    doc_type = metadata.get('type', '')
                    
                    if doc_type == 'blog_article' or 'tactical' in content.lower()[:200]:
                        # It's a tactical article
                        title = metadata.get('title', 'Tactical Article')
                        source = metadata.get('source', 'Blog')
                        source_key = f"{title}_{source}"
                        if source_key not in seen_sources:
                            sources.append({
                                'player': title,
                                'team': source,
                                'season': 'Article',
                                'type': 'blog',
                                'relevance': relevance,
                                'content': content[:200]
                            })
                            seen_sources.add(source_key)
                    else:
                        # It's player stats
                        player = metadata.get('player') or metadata.get('Player') or metadata.get('name') or 'Player'
                        team = metadata.get('team') or metadata.get('Squad') or metadata.get('club') or ''
                        season = metadata.get('season') or metadata.get('Season') or ''
                        stat_module = metadata.get('stat_module', 'identity')  # Extract module info
                        
                        source_key = f"{player}_{team}_{season}"
                        if source_key not in seen_sources:
                            sources.append({
                                'player': player,
                                'team': team,
                                'season': season,
                                'type': 'stats',
                                'relevance': relevance,
                                'stat_module': stat_module,
                                'content': content[:200]
                            })
                            seen_sources.add(source_key)
        
        # Add web results
        if web_results:
            # Enforce club-lock: if a primary DB club exists, ignore web results that conflict
            primary_db_club = state.get('club_lock') or ''
            filtered_web = []
            conflicting_found = False
            for web_doc in web_results:
                content = web_doc.get('content', '')
                url = web_doc.get('url', 'Web source')
                if not content:
                    continue

                if primary_db_club:
                    # If web content doesn't mention the DB club, treat as potential conflict and ignore
                    if primary_db_club.lower() not in content.lower():
                        conflicting_found = True
                        continue

                # Accept web doc
                filtered_web.append(web_doc)

            # Add only filtered web results
            for web_doc in filtered_web:
                content = web_doc.get('content', '')
                all_content.append(content)
                url = web_doc.get('url', 'Web source')
                if url not in seen_sources:
                    sources.append({
                        'player': 'Web Source',
                        'team': url,
                        'season': 'Live',
                        'type': 'web',
                        'relevance': 0.6,
                        'content': content[:200]
                    })
                    seen_sources.add(url)

            if conflicting_found:
                # Expose club conflict to state/Self-Check
                state['club_conflict_detected'] = True
                logger.warning('Conflicting web results found for club; ignored web sources to enforce DB club lock')
        
        if not self.llm:
            # Simple fallback answer
            db_note = "[DB] " if docs else ""
            web_note = "[Web] " if web_results else ""
            answer = f"{db_note}{web_note}Based on the data:\n\n{all_content[0][:500] if all_content else 'No data found'}"
            state["final_answer"] = answer
            state["sources"] = sources
            state["confidence"] = 0.5
            return state
        
        # Determine season context
        query_lower = query.lower()
        season_specific = any(word in query_lower for word in ['this season', 'current season', '2025', '2024'])
        is_comparison = any(word in query_lower for word in ['compare', 'vs', 'versus', 'better'])
        
        # Build context with tags
        db_context = ""
        web_context = ""
        
        if docs:
            # Group docs by player for comparisons
            if is_comparison:
                player_groups = {}
                for doc in docs:
                    player = doc.get('metadata', {}).get('comparison_player') or doc.get('metadata', {}).get('player', 'Unknown')
                    if player not in player_groups:
                        player_groups[player] = []
                    player_groups[player].append(doc.get('content', '')[:500])  # Truncate to 500 chars
                
                db_facts = []
                for player, contents in player_groups.items():
                    db_facts.append(f"\n=== {player} ===\n" + "\n".join(contents[:2]))
                db_context = f"[DATABASE FACTS]:\n{''.join(db_facts)}"
            else:
                # Use 2 player docs (600 chars) + 1 blog (400 chars) for balanced optimization
                player_docs = [doc.get('content', '')[:600] for doc in docs[:2]]  # 2 player docs x 600 chars
                blog_docs = [doc.get('content', '')[:400] for doc in docs[2:3]]   # 1 blog doc x 400 chars
                db_facts = "\n\n".join(player_docs + blog_docs)
                db_context = f"[DATABASE FACTS]:\n{db_facts}"
        
        if web_results:
            web_facts = "\n\n".join([web['content'] for web in web_results[:3]])
            web_context = f"\n\n[WEB FACTS (Recent/Live)]:\n{web_facts}"
        
        # Build column-specific context FIRST before combining
        column_specific_context = ""
        if docs and not is_comparison:
            # MERGE metadata from all modules (identity + shooting + passing + etc.)
            merged_meta = {}
            player_name = None
            modules_found = []
            
            logger.info(f" Starting metadata merge from {len(docs)} retrieved docs")
            
            # Merge from ALL docs (not just first 10) to ensure we capture all stat modules
            for i, doc in enumerate(docs):
                doc_meta = doc.get('metadata', {})
                stat_module = doc_meta.get('stat_module', '')
                doc_player = doc_meta.get('player')
                
                # Track player name for consistency
                if not player_name:
                    player_name = doc_player
                elif player_name != doc_player:
                    # Skip docs from different players (comparison guard)
                    if i > 0:  # But still check first doc
                        continue
                
                if stat_module:
                    modules_found.append(stat_module)
                
                # Merge ALL columns from this module (no filtering)
                for key, val in doc_meta.items():
                    if key not in ['chunk_id', 'parent_id', 'source', 'chunk_index', 'total_chunks', 
                                  'token_count', 'content_hash', 'stat_module', 'chunk_type', 
                                  'document_id', 'comparison_player', 'distance']:
                        if key not in merged_meta:  # Don't overwrite (keep first occurrence)
                            merged_meta[key] = val
            
            logger.info(f" Merged {len(merged_meta)} unique columns from modules: {list(set(modules_found))}")
            
            # FALLBACK: Load player data from CSV to fill in missing columns
            # This ensures all stats are available even if not in Chroma documents
            try:
                if player_name and season:
                    import pandas as pd
                    import os
                    
                    csv_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed', 'player_stats_unified_FINAL_DEDUPED.csv')
                    if os.path.exists(csv_path):
                        df = pd.read_csv(csv_path)
                        # Find matching player record
                        player_mask = (df['player'].str.contains(player_name, na=False, case=False, regex=False)) & (df['season'] == season)
                        if len(player_mask) > 0 and player_mask.sum() > 0:
                            player_row = df[player_mask].iloc[0]
                            # Fill in missing columns from CSV
                            for col in player_row.index:
                                if col not in merged_meta and pd.notna(player_row[col]) and player_row[col] != '':
                                    merged_meta[col] = player_row[col]
                            logger.info(f" Filled {len(merged_meta)} columns total after CSV fallback (added {len([c for c in player_row.index if c not in merged_meta])} new columns from CSV)")
            except Exception as e:
                logger.debug(f"CSV fallback lookup failed: {e}")  # Silent fail, continue with Chroma data
            
            # Log first 20 column names for debugging
            first_columns = list(merged_meta.keys())[:20]
            logger.debug(f"Sample columns in merged_meta: {first_columns}")
            
            logger.info(f" Merged {len(merged_meta)} total columns from modules: {modules_found}")

            # NORMALIZE COLUMN NAMES: add common aliases so prompts can find expected keys
            # e.g., Progression_PrgC_std -> PrgC_poss / progressive_carries
            def add_alias(key, alias):
                if key in merged_meta and alias not in merged_meta:
                    merged_meta[alias] = merged_meta[key]

            # Map progressive passes and carries, dribble stats
            for k in list(merged_meta.keys()):
                kl = k.lower()
                if 'prgp' in kl or 'prg_p' in kl or 'prgp_pass' in kl or kl == 'prgp_pass' or 'prgp' in kl:
                    add_alias(k, 'PrgP_pass')
                    add_alias(k, 'progressive_passes')
                if ('prgc' in kl or 'prg_c' in kl or 'progression_prgc' in kl or ('progression' in kl and 'prg' in kl)):
                    add_alias(k, 'PrgC_poss')
                    add_alias(k, 'progressive_carries')
                if 'succ_drib' in kl or 'succdrib' in kl or 'successful_dribbles' in kl:
                    add_alias(k, 'Succ_drib')
                    add_alias(k, 'successful_dribbles')
                if 'att_drib' in kl or 'attdrib' in kl or ('dribble' in kl and 'att' in kl):
                    add_alias(k, 'Att_drib')
                    add_alias(k, 'dribble_attempts')
                # Defensive stats aliases
                if 'tklw' in kl and '%' in k:
                    add_alias(k, 'Tackles Won %')
                    add_alias(k, 'TackleWonPct')
                if 'clr' in kl or 'clearance' in kl:
                    add_alias(k, 'Clearances')
                    add_alias(k, 'Clr')
                # AGGRESSIVE: Match any aerial/duel variations - try multiple patterns
                if ('aerial' in kl or 'duel' in kl or 'aerialduels' in kl or 'aeriald' in kl):
                    if 'won' in kl or 'win' in kl or 'w' in k.split('_')[-1].lower():
                        add_alias(k, 'Aerial Duels Won')
                        add_alias(k, 'AerialDuelsWon')
                        add_alias(k, 'AerialWon')
                        add_alias(k, 'Aerial_Duels')
                        add_alias(k, 'Aerial Duels')
                # Also try looking for columns with 'Aerial' exactly
                if k.startswith('Aerial') and 'won' in kl:
                    add_alias(k, 'Aerial Duels Won')
                    add_alias(k, 'AerialDuelsWon')
                if 'dribbled' in kl and 'past' in kl:
                    add_alias(k, 'Times Dribbled Past')
                    add_alias(k, 'Dribbled')
                if 'err' in kl or 'error' in kl:
                    add_alias(k, 'Errors Leading to Shot/Goal')
                    add_alias(k, 'Errors')
                # AGGRESSIVE: Match any fouls variation - try multiple patterns
                if 'foul' in kl or ('fls' in kl) or ('fld' in kl):
                    if 'commit' in kl or 'cmt' in kl or (kl == 'fls'):
                        add_alias(k, 'Fouls Committed')
                        add_alias(k, 'Fls')
                        add_alias(k, 'Fouls')
                    if 'drawn' in kl or (kl == 'fld'):
                        add_alias(k, 'Fouls Drawn')
                        add_alias(k, 'Fld')
                # Direct mapping for Fls and Fld if they exist
                if k == 'Fls' or kl == 'fls':
                    add_alias(k, 'Fouls Committed')
                if k == 'Fld' or kl == 'fld':
                    add_alias(k, 'Fouls Drawn')
                # Pass completion - AGGRESSIVE matching
                if ('cmp' in kl and '%' in k) or ('pass' in kl and '%' in k):
                    add_alias(k, 'Cmp%_pass')
                    add_alias(k, 'Pass Completion')
                    add_alias(k, 'Pass Cmp%')
                # Long pass accuracy
                if 'long' in kl and 'cmp' in kl and '%' in k:
                    add_alias(k, 'Long_Cmp%_pass')
                    add_alias(k, 'Long Pass Accuracy')
                # Goalkeeper stats
                if 'save' in kl and '%' in k:
                    add_alias(k, 'Save%')
                    add_alias(k, 'SavePct')
                    add_alias(k, 'Performance_Save%_gk')
                if 'psxg' in kl or ('psxg' in kl and 'ga' in kl):
                    add_alias(k, 'PSxG-GA')
                    add_alias(k, 'PSxG')
                if 'launch' in kl and '%' in k:
                    add_alias(k, 'Launch%')
                    add_alias(k, 'LaunchPct')
                if 'cs' in kl and ('clean' in kl or 'sheet' in kl):
                    add_alias(k, 'Clean Sheets')
                    add_alias(k, 'CS')
            
            # CALCULATE DERIVED STATS if they don't exist
            # Pass Completion % from Cmp_pass and Att_pass
            if 'Cmp%_pass' not in merged_meta and 'Cmp_pass' in merged_meta and 'Att_pass' in merged_meta:
                try:
                    cmp = float(merged_meta['Cmp_pass'])
                    att = float(merged_meta['Att_pass'])
                    if att > 0:
                        pass_pct = (cmp / att) * 100
                        merged_meta['Cmp%_pass'] = f"{pass_pct:.1f}%"
                        merged_meta['Pass Completion'] = f"{pass_pct:.1f}% ({int(cmp)}/{int(att)})"
                        logger.info(f" Calculated Pass Completion: {pass_pct:.1f}%")
                except (ValueError, TypeError):
                    pass
            
            # Aerial Duels per 90 from AerialDuelsWon and minutes
            if 'AerialDuels_Won_per_90' not in merged_meta and 'AerialDuelsWon' in merged_meta:
                try:
                    aerial_won = float(merged_meta['AerialDuelsWon'])
                    minutes = merged_meta.get('Playing Time_Min_std') or merged_meta.get('Playing Time_Min_gk') or 0
                    if minutes and float(minutes) > 0:
                        per_90 = (aerial_won / (float(minutes) / 90))
                        merged_meta['AerialDuels_Won_per_90'] = f"{per_90:.2f}"
                        merged_meta['Aerial Duels per 90'] = f"{per_90:.2f}"
                        logger.info(f"Calculated Aerial Duels per 90: {per_90:.2f}")
                except (ValueError, TypeError):
                    pass
            
            # IMPROVED STAT EXTRACTION: Show all available stats (don't over-filter)
            # Extract stats from metadata - include ALL available stats
            stat_dict = {}
            
            PRIORITY_BY_POSITION = {
                'GK': [
                    'player', 'pos', 'age', 'team', 'season', 'market_value',
                    # GK-specific columns
                    'Playing Time_MP_gk', 'Playing Time_Min_gk', 'Playing Time_90s_gk',
                    'Performance_GA_gk', 'Performance_GA90_gk', 'Performance_Saves_gk',
                    'Performance_Save%_gk', 'Performance_CS_gk', 'Performance_CS%_gk',
                    'Penalty Kicks_PKatt_gk', 'Penalty Kicks_PKsv_gk',
                    # Passing for GK
                    'Total_Cmp%_pass', 'Total_Cmp_pass', 'Total_Att_pass',
                    'Short_Cmp%_pass', 'Medium_Cmp%_pass', 'Long_Cmp%_pass',
                    'PrgP_pass', 'PrgC_poss'
                ],
                'DF': [
                    'player', 'pos', 'age', 'team', 'season', 'market_value',
                    # Playing time
                    'Playing Time_MP_std', 'Playing Time_Min_std', 'Playing Time_90s_std',
                    # Defensive core stats
                    'Tackles_Tkl_def', 'Tackles_TklW_def', 'Challenges_Tkl%_def',
                    'Int_def', 'Blocks_Blocks_def', 'Clr_def', 'Err_def',
                    # Attacking contribution
                    'Performance_Gls_std', 'Performance_Ast_std',
                    # Passing stats
                    'Total_Cmp%_pass', 'Total_Cmp_pass', 'Total_Att_pass',
                    'PrgP_pass', 'Progression_PrgP_std', '1/3_pass',
                    'Long_Cmp%_pass', 'CrsPA_pass'
                ],
                'MF': [
                    'player', 'pos', 'age', 'team', 'season', 'market_value',
                    # Playing time and attacking
                    'Playing Time_MP_std', 'Playing Time_Min_std', 'Playing Time_90s_std',
                    'Performance_Gls_std', 'Performance_Ast_std', 'Performance_G+A_std',
                    # Expected stats
                    'Expected_xG_std', 'Expected_xAG_std', 'Expected_npxG+xAG_std',
                    # Passing and progression
                    'Total_Cmp%_pass', 'Total_Cmp_pass', 'Total_Att_pass', 'Total_PrgDist_pass',
                    'KP_pass', 'PPA_pass', '1/3_pass', 'CrsPA_pass',
                    'PrgP_pass', 'Progression_PrgP_std', 'Progression_PrgC_std',
                    # Shooting stats
                    'Standard_Sh_shoot', 'Standard_SoT_shoot', 'Standard_SoT%_shoot',
                    'Standard_G/Sh_shoot', 'Standard_G/SoT_shoot',
                    # Dribbling and touches
                    'Challenges_Tkl_def', 'Challenges_Tkl%_def',
                    # Defensive contribution
                    'Tackles_Tkl_def', 'Int_def', 'Blocks_Blocks_def', 'Clr_def'
                ],
                'FW': [
                    'player', 'pos', 'age', 'team', 'season', 'market_value',
                    # Playing time
                    'Playing Time_MP_std', 'Playing Time_Min_std', 'Playing Time_90s_std',
                    # Attacking stats
                    'Performance_Gls_std', 'Performance_Ast_std', 'Performance_G+A_std',
                    'Performance_G-PK_std', 'Performance_PK_std',
                    # Expected stats
                    'Expected_xG_std', 'Expected_npxG_std', 'Expected_xAG_std', 
                    'Expected_npxG+xAG_std',
                    # Shooting
                    'Standard_Sh_shoot', 'Standard_SoT_shoot', 'Standard_SoT%_shoot',
                    'Standard_G/Sh_shoot', 'Standard_G/SoT_shoot',
                    'Standard_Dist_shoot', 'Standard_FK_shoot',
                    # Passing
                    'KP_pass', 'Total_Cmp%_pass', 'Total_Cmp_pass', 'Total_Att_pass',
                    'PPA_pass', '1/3_pass', 'CrsPA_pass', 'PrgP_pass'
                ],
                'DEFAULT': [
                    'player', 'pos', 'age', 'team', 'season', 'market_value',
                    'Playing Time_MP_std', 'Playing Time_Min_std', 'Performance_Gls_std', 
                    'Performance_Ast_std', 'Expected_xG_std', 'Expected_xAG_std', 
                    'Tackles_Tkl_def', 'Int_def', 'Blocks_Blocks_def'
                ]
            }
            
            # Determine detected_position (reuse detected_position logic if present)
            detected_position = None
            try:
                pos_field = merged_meta.get('pos') or merged_meta.get('position') or ''
                if pos_field:
                    primary_position = pos_field.split(',')[0].strip()
                    if 'GK' in primary_position:
                        detected_position = 'GK'
                    elif 'FW' in primary_position:
                        detected_position = 'FW'
                    elif 'DF' in primary_position:
                        detected_position = 'DF'
                    elif 'MF' in primary_position or 'AM' in primary_position:
                        detected_position = 'MF'
            except Exception:
                detected_position = None

            # Select priority list for the detected position
            prio_list = PRIORITY_BY_POSITION.get(detected_position or 'DEFAULT', PRIORITY_BY_POSITION['DEFAULT'])

            # Add priority stats first (show numeric zeros as valid values)
            # Look for EXACT column names in merged_meta
            for key in prio_list:
                if key in merged_meta:
                    val = merged_meta[key]
                    # Only mark as unavailable if truly missing (None or 'nan'/'none' strings)
                    if val is None or (isinstance(val, str) and val.strip().lower() in ['nan', 'none']):
                        stat_dict[key] = "Data not available"
                    else:
                        stat_dict[key] = val
                else:
                    # Key not found in merged_meta - don't add to stat_dict (will show as "not in database")
                    pass
            
            # POSITION-SPECIFIC FILTERING: Skip columns inappropriate for detected position
            # This prevents showing defender stats (Tackles, Blocks) for goalkeepers, etc.
            POSITION_SKIP_PATTERNS = {
                'GK': ['_def', '_std', 'Tackles', 'Interceptions', 'Int_def', 'Blocks_', 'Clr', 'Fls', 'Fld',
                       'Pressures', 'Pressure', 'Duels', 'Dribble', 'Challenges'],  # Skip defender/field player stats for GK
                'DF': ['_gk', 'Performance_Saves', 'PSxG', 'Save%', 'Launch%'],  # Skip GK-specific stats
                'MF': ['_gk', 'Performance_Saves', 'Save%', 'Launch%'],  # Skip GK stats
                'FW': ['_gk', 'Performance_Saves', 'Save%', 'Launch%', 'Clearance', 'Blocks_']  # Skip GK/DEF stats
            }
            
            skip_patterns = POSITION_SKIP_PATTERNS.get(detected_position, [])
            
            # Add remaining stats from merged_meta (non-metadata fields only)
            SKIP_KEYS = ['chunk_id', 'parent_id', 'source', 'chunk_index', 'total_chunks', 
                        'token_count', 'content_hash', 'stat_module', 'chunk_type', 
                        'document_id', 'comparison_player']
            
            for key, val in sorted(merged_meta.items()):
                # Skip if already in priority list
                if key in stat_dict or key in SKIP_KEYS:
                    continue
                
                # Skip position-inappropriate columns
                skip_this = False
                for pattern in skip_patterns:
                    if pattern.lower() in key.lower():
                        skip_this = True
                        break
                
                if skip_this:
                    continue  # Don't show this column for this position
                
                # Add to display
                if val is None or (isinstance(val, str) and val.strip().lower() in ['nan', 'none']):
                    stat_dict[key] = "Data not available"
                else:
                    stat_dict[key] = val
            
            # Format compactly - show all available stats
            stat_lines = ["=" * 60]
            stat_lines.append("DATABASE STATISTICS (Use EXACT column names below)")
            stat_lines.append("=" * 60)
            stat_lines.append("")
            stat_lines.append(" CRITICAL INSTRUCTIONS FOR USING THESE STATS:")
            stat_lines.append("- Use EXACT column names shown below (e.g., 'Tackles_Tkl_def = 39', 'PrgP_pass = 32')")
            stat_lines.append("- If you see a column with an equals sign and value, USE THAT EXACT VALUE")
            stat_lines.append("- For example: 'Tackles_Tkl_def = 39' → write 'Tackles: 39' in report")
            stat_lines.append("- If column is NOT shown in the lists below, then say 'Data not available'")
            stat_lines.append("- NUMERIC ZEROS (0, 0.0) ARE VALID VALUES - example: 'Yellow Cards: 0'")
            stat_lines.append("- COMMON COLUMN NAME PATTERNS:")
            stat_lines.append("  * 'Tkl_Won%_def' = Tackles Won Percentage")
            stat_lines.append("  * 'AerialDuelsWon' or 'Aerial_Duels' = Aerial Duels Won")
            stat_lines.append("  * '1/3_pass' or 'PassInto1/3' = Passes into Final Third")
            stat_lines.append("  * 'Long_Cmp%_pass' = Long Pass Accuracy")
            stat_lines.append("  * 'Err' = Errors Leading to Shot/Goal")
            stat_lines.append("  * 'Fls' = Fouls Committed, 'Fld' = Fouls Drawn")
            stat_lines.append("  * 'Pressures_Press_press' = Pressures Applied")
            stat_lines.append("")
            
            # Show ALL available stats from merged_meta (complete transparency)
            # Priority stats first
            stat_lines.append("\n PRIORITY STATS (Position-Specific):\n")
            available_count = 0
            for key in prio_list:
                if key in stat_dict:
                    val = stat_dict[key]
                    if val != "Data not available":
                        stat_lines.append(f"{key} = {val}")
                        available_count += 1
            
            # Then ALL remaining stats from merged_meta (comprehensive coverage)
            stat_lines.append("\n ALL OTHER AVAILABLE STATS (Raw Database Columns):\n")
            other_count = 0
            SKIP_KEYS_DISPLAY = ['chunk_id', 'parent_id', 'source', 'chunk_index', 'total_chunks', 
                        'token_count', 'content_hash', 'stat_module', 'chunk_type', 
                        'document_id', 'comparison_player']
            for key, val in sorted(merged_meta.items()):
                # Skip if already shown in priority or in skip list
                if key in stat_dict or key in SKIP_KEYS_DISPLAY or key in prio_list:
                    continue
                
                # Skip position-inappropriate columns in other stats too
                skip_this = False
                for pattern in skip_patterns:
                    if pattern.lower() in key.lower():
                        skip_this = True
                        break
                
                if skip_this:
                    continue  # Don't show position-inappropriate column
                
                if val is None or (isinstance(val, str) and val.strip().lower() in ['nan', 'none']):
                    continue  # Skip missing values in "other" section
                stat_lines.append(f"{key} = {val}")
                other_count += 1
            
            stat_lines.append("")
            stat_lines.append(f"[Total: {available_count} priority stats + {other_count} other stats = {available_count + other_count} total from database]")
            stat_lines.append("=" * 60)
            stat_lines.append("")
            
            logger.info(f" Showing {available_count + other_count} available stats ({available_count} priority + {other_count} other)")
            
            column_specific_context = "\n".join(stat_lines)
        
        combined_context = column_specific_context + db_context + web_context
        
        # If no context at all, return error message
        if not combined_context.strip():
            state["final_answer"] = "No information found. The database doesn't contain relevant data for this query, and web search didn't return results."
            state["sources"] = []
            state["confidence"] = 0.0
            return state
        
        # Use LLM to generate answer with position-aware prompts
        
        # Detect player position for appropriate template
        detected_position = None
        if POSITION_PROMPTS_AVAILABLE:
            # PRIORITY 1: Use 'pos' field from merged_meta (most reliable: GK/DF/MF/FW)
            # CRITICAL: Use merged_meta which aggregates data from all modules, not just first doc
            if column_specific_context and merged_meta:
                pos_field = merged_meta.get('pos') or merged_meta.get('position') or ''
                if pos_field:
                    # Map DB position codes to template positions
                    # For multi-position players (e.g., "FW,MF"), prioritize PRIMARY position (first one)
                    primary_position = pos_field.split(',')[0].strip()
                    
                    if 'GK' in primary_position:
                        detected_position = 'GK'
                    elif 'FW' in primary_position:
                        detected_position = 'FW'
                    elif 'DF' in primary_position:
                        detected_position = 'DF'
                    elif 'MF' in primary_position or 'AM' in primary_position:
                        detected_position = 'MF'
                    logger.info(f" Position from merged_meta 'pos' field: {pos_field} -> primary: {primary_position} -> template: {detected_position}")
            
            # FALLBACK: Try first doc if merged_meta didn't have position
            if not detected_position and docs:
                first_meta = docs[0].get('metadata', {})
                pos_field = first_meta.get('pos') or first_meta.get('position') or ''
                if pos_field:
                    primary_position = pos_field.split(',')[0].strip()
                    
                    if 'GK' in primary_position:
                        detected_position = 'GK'
                    elif 'FW' in primary_position:
                        detected_position = 'FW'
                    elif 'DF' in primary_position:
                        detected_position = 'DF'
                    elif 'MF' in primary_position or 'AM' in primary_position:
                        detected_position = 'MF'
                    logger.info(f" Position from first doc fallback 'pos' field: {pos_field} -> primary: {primary_position} -> template: {detected_position}")

            # PRIORITY 2: Try to detect from query
            if not detected_position or detected_position == 'unknown':
                detected_position = detect_position_from_query(query)
                if detected_position and detected_position != 'unknown':
                    logger.info(f" Position from query detection: {detected_position}")

            # PRIORITY 3: Try legacy metadata detection
            if (not detected_position or detected_position == 'unknown') and docs:
                for doc in docs[:3]:  # Check first 3 docs
                    metadata = doc.get('metadata', {})
                    detected_position = detect_position_from_metadata(metadata)
                    if detected_position != 'unknown':
                        logger.info(f" Position from metadata detection: {detected_position}")
                        break

            # Final fallback: try to match known players from prompts
            if (detected_position == 'unknown' or detected_position is None) and docs:
                try:
                    from src.agents.position_prompts import KNOWN_PLAYERS
                    # Attempt to extract a capitalized name from query
                    name_match = None
                    m = re.search(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", query)
                    if m:
                        name_match = m.group(1)
                    # Also check first doc metadata for player
                    if not name_match and docs[0].get('metadata'):
                        name_match = docs[0]['metadata'].get('player')

                    if name_match:
                        nm_lower = name_match.lower()
                        for pos, players in KNOWN_PLAYERS.items():
                            for p in players:
                                if nm_lower in p.lower() or p.lower() in nm_lower:
                                    detected_position = pos
                                    logger.info(f" Position from KNOWN_PLAYERS match: {detected_position}")
                                    break
                            if detected_position and detected_position != 'unknown':
                                break
                except Exception:
                    # If any issue, leave detected_position as unknown
                    detected_position = detected_position or 'unknown'

            # Log final position determination
            final_position = detected_position if detected_position else 'unknown'
            logger.info(f" FINAL Position detected: {final_position} - Using {final_position}-specific prompt template")

            # Get appropriate prompt template
            position_specific_prompt = get_prompt_for_position(final_position)

            # Use position-specific template
            answer_prompt = ChatPromptTemplate.from_template(position_specific_prompt)
        else:
            # Position prompts not available - use detailed fallback
            logger.warning(" POSITION_PROMPTS_AVAILABLE=False - using fallback template")
            answer_prompt = ChatPromptTemplate.from_template(
                """You are a professional football scout writing a detailed scouting report.

Query: {question}

Data:
{context}

Write a comprehensive SCOUT REPORT using this structure:

## PLAYER SNAPSHOT
- Name, Position, Age, Club, Season
- Market Value (if available)
- Matches and Minutes

## DATA QUALITY
- Minutes Played: [assess based on minutes]
- Metric Coverage: [assess based on available stats]
- Overall Reliability: [Low/Medium/High]

## EXECUTIVE SUMMARY
[4-5 sentences: Player archetype, key strength with stat, stat evidence, weakness, market value assessment]

## SEASON-SPECIFIC PERFORMANCE
### Attacking Output
Goals, Assists, xG, xAG, Key Passes, Shot Accuracy
Tactical Analysis: [4-5 sentences on chance creation, positioning, movement, decision-making]

### Ball Progression  
Progressive Passes, Progressive Carries, Passes into Final Third
Tactical Analysis: [2-3 sentences on progression methods]

### Defensive Contribution
Tackles, Interceptions, Blocks, Pressures
Tactical Analysis: [2-3 sentences on defensive behavior]

### Passing & Retention
Pass Completion %, Short/Medium/Long Pass %, Passes per 90
Tactical Analysis: [2-3 sentences on passing style]

## STRENGTHS (Evidence-Based)
- [Strength 1 with stat]
- [Strength 2 with stat]

## DEVELOPMENT AREAS (Evidence-Based)
- [Weakness 1 with stat]
- [Weakness 2 with stat]

## GAME MODEL FIT
Ideal Systems, Risk Factors, Optimal Role

## TACTICAL FIT
[1-2 sentences on best tactical fit]

## SCOUTING RECOMMENDATION
Profile Level, Value Assessment, Decision, Rationale

CRITICAL RULES:
- Use ONLY provided data from {context}
- Write player's actual name, not "Player"
- Be specific with stats and examples
- If data missing, say "Data not available"
"""
            )

        try:
            # Extract player info from docs for similar players feature and market value check
            player_name = None
            player_position = None
            player_season = "2024-2025"  # default
            has_market_value = False
            
            if docs and not is_comparison:
                # Try to extract from first doc metadata
                first_meta = docs[0].get('metadata', {})
                player_name = first_meta.get('player') or first_meta.get('Player')
                player_position = first_meta.get('position')
                player_season = first_meta.get('season', '2024-2025')
                
                # Check if market value exists in any doc
                for doc in docs[:3]:  # Check first 3 docs
                    meta = doc.get('metadata', {})
                    market_val = meta.get('market_value', '')
                    if market_val and str(market_val).lower() not in ['nan', 'none', '']:
                        has_market_value = True
                        break
                
                # If no market value, search Tavily for current market value
                if player_name and not has_market_value:
                    logger.info(f"No market value in DB for {player_name}, attempting web search for market value...")
                    try:
                        market_query = f"{player_name} current market value transfermarkt 2024 2025"
                        tavily_client = getattr(self, 'tavily', None)
                        if tavily_client is None:
                            logger.warning("Tavily client not configured - skipping web market value lookup")
                        else:
                            market_results = tavily_client.search(
                                query=market_query,
                                max_results=2,
                                search_depth="basic"
                            )

                            if market_results and market_results.get('results'):
                                market_context = "\n\n[CURRENT MARKET VALUE - FROM WEB SEARCH]:\n"
                                for result in market_results['results'][:2]:
                                    market_context += f"Source: {result.get('url', 'N/A')}\n"
                                    market_context += f"Info: {result.get('content', '')[:300]}...\n\n"

                                # Add market value context to combined context
                                combined_context += market_context
                                logger.success(f" Found market value via web search for {player_name}")
                    except Exception as e:
                        logger.warning(f"Could not fetch market value from web: {e}")
            
            # Find similar players if we have player info
            # DISABLED for token optimization - similar players can add 500+ tokens
            similar_players_context = ""
            
            # Combine all context (stats already prepended at top)
            full_context = combined_context + similar_players_context

            # --- MINIMAL ROW-LOCK for player identity (reduced tokens) ---
            source_of_truth = ""
            if docs and not is_comparison:
                first_meta = docs[0].get('metadata', {})
                player_lock_name = first_meta.get('player') or first_meta.get('Player') or 'Unknown'
                club_lock = first_meta.get('team') or first_meta.get('club') or first_meta.get('Squad') or ''
                position_lock = first_meta.get('pos') or first_meta.get('position') or ''

                source_of_truth = (
                    f"PLAYER: {player_lock_name} | CLUB: {club_lock} | POS: {position_lock}\n"
                )

                # Prepend source of truth to context
                full_context = source_of_truth + full_context
            
            # ========== INTENT-AWARE TEMPLATE SELECTION ==========
            # Import intent templates
            try:
                from src.agents.intent_templates import get_template_for_intent
                from src.agents.intent_classifier import QueryIntent
                
                # Get intent from state (passed from EnhancedCRAGAgent)
                intent = state.get('intent')
                
                # Use intent-specific template if available, otherwise use position-specific
                if intent and intent != QueryIntent.SCOUT_REPORT and intent != QueryIntent.UNKNOWN:
                    intent_template = get_template_for_intent(intent)
                    if intent_template:
                        logger.info(f" Using intent-specific template: {intent.value}")
                        answer_prompt = ChatPromptTemplate.from_template(intent_template)
                    else:
                        logger.info(f" No template for intent {intent.value}, using position-specific")
                else:
                    logger.info("Using full scout report template (default)")
            except Exception as e:
                logger.warning(f"Could not load intent templates: {e}, using position-specific")
            # ========== END INTENT-AWARE TEMPLATE SELECTION ==========
            
            answer_chain = answer_prompt | self.llm | StrOutputParser()
            
            # Log the context being sent to help debug  
            # Note: Using 2 player docs (600 chars) + 1 blog (400 chars) = ~1600 chars, safe margin under 6K tokens
            logger.info(f"Generating answer with {len(all_content)} context chunks (~1600 chars approx)")
            
            answer = answer_chain.invoke({
                "question": query,
                "context": full_context,
                "season": player_season
            })
            
            # Validate answer isn't empty or nonsensical
            if not answer or len(answer.strip()) < 10 or answer.strip().lower() in ['football', 'soccer', 'player', 'no', 'yes', 'query', 'answer']:
                logger.error(f"Generated answer is too short or invalid: '{answer}'")
                
                # Build a basic answer from available data
                if sources and len(sources) > 0:
                    first_source = sources[0]
                    player = first_source.get('player', 'Unknown')
                    team = first_source.get('team', 'Unknown')
                    season = first_source.get('season', 'Unknown')
                    answer = f"Unable to generate detailed analysis. Available data shows {player} playing for {team} in {season}. Please try a more specific query or check if the player exists in our database."
                else:
                    answer = f"No data found in database for this query. The player may not exist in our {query_lower} database, or the query needs to be more specific (e.g., include player's full name)."

            state["final_answer"] = answer
            state["sources"] = sources

            # INTELLIGENT CONFIDENCE: Factor in minutes, metric completeness, sources
            base_confidence = 0.95  # Start high for DB data
            
            # Only adjust for sample size if we have merged_meta (DB documents exist)
            minutes_played = 0
            available_stats = 0
            
            if 'merged_meta' in locals() and merged_meta:
                # Adjust for sample size (minutes played)
                minutes_played = merged_meta.get('Playing Time_Min_std') or merged_meta.get('Playing Time_Min_gk') or 0
                try:
                    minutes_played = float(minutes_played) if minutes_played else 0
                except:
                    minutes_played = 0
                
                if minutes_played < 300:  # Less than ~3 matches
                    base_confidence *= 0.65  # Drop to 62% (0.95 * 0.65)
                    logger.warning(f" Low sample size ({minutes_played} min) - confidence reduced to ~62%")
                elif minutes_played < 900:  # Less than ~10 matches
                    base_confidence *= 0.85  # Drop to 81% (0.95 * 0.85)
                    logger.info(f" Limited sample size ({minutes_played} min) - confidence at ~81%")
                else:
                    logger.info(f" Strong sample size ({minutes_played} min) - full confidence")
                
                # Adjust for metric completeness
                available_stats = sum(1 for v in merged_meta.values() if v and str(v).lower() not in ['nan', 'none', ''])
                if available_stats < 15:
                    base_confidence *= 0.80  # Drop for sparse data
                    logger.warning(f" Sparse metrics ({available_stats} stats) - confidence reduced")
                elif available_stats < 30:
                    base_confidence *= 0.90  # Slight drop for partial data
                    logger.info(f" Partial metrics ({available_stats} stats) - slight confidence reduction")
                else:
                    logger.info(f" Comprehensive metrics ({available_stats} stats)")
            else:
                # No DB data - using only web results
                base_confidence = 0.70  # Lower baseline for web-only
                logger.warning(" No database data - using web results only, confidence reduced to 70%")
            
            # Adjust for source types
            if web_results and docs:
                base_confidence *= 0.95  # Slight drop if we needed web + DB
                logger.info("Web + DB results - minor confidence adjustment")
            elif web_results and not docs:
                # Already handled above with base_confidence = 0.70
                pass
            
            # Final confidence clamping
            confidence = max(0.50, min(0.96, base_confidence))  # Keep between 50-96%
            
            logger.info(f"Final Confidence: {confidence:.0%} (minutes: {minutes_played}, metrics: {available_stats}, sources: {'DB+Web' if docs and web_results else 'Web' if web_results else 'DB'})")
            
            state["confidence"] = confidence
        except Exception as e:
            logger.error(f"Generation error: {e}")
            state["final_answer"] = f"Error generating answer: {str(e)}"
            state["sources"] = sources
            state["confidence"] = 0.3

        return state
    
    def find_similar_players(self, player_name: str, position: str, season: str = "2024-2025", n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Find players similar to the queried player based on statistical profile
        
        Args:
            player_name: Name of the reference player
            position: Position of the player (FW, MF, DF, GK)
            season: Season to analyze (default: 2024-2025)
            n_results: Number of similar players to return
            
        Returns:
            List of similar players with their stats and similarity scores
        """
        try:
            # First, get the reference player's stats
            # ChromaDB only supports one operator at root level
            where_filter = {'season': season}
            
            # Query for reference player
            ref_query = f"{player_name} stats {season}"
            ref_results = self.vector_db.query(
                query_texts=[ref_query],
                n_results=20,  # Get more to filter by position in Python
                where=where_filter
            )
            
            # Extract reference player's stats (filter by position in Python)
            ref_stats = None
            if ref_results and ref_results.get('documents'):
                for i, doc in enumerate(ref_results['documents'][0]):
                    metadata = ref_results['metadatas'][0][i]
                    doc_player = metadata.get('player', '').lower()
                    doc_position = metadata.get('position', '')
                    
                    # Check if position matches
                    position_match = position in doc_position if doc_position else False
                    
                    if player_name.lower() in doc_player and position_match:
                        ref_stats = metadata
                        break
            
            if not ref_stats:
                logger.warning(f"Could not find reference stats for {player_name}")
                return []
            
            # Query for similar players using semantic search
            similar_query = f"player similar to {player_name} {position} {season}"
            similar_results = self.vector_db.query(
                query_texts=[similar_query],
                n_results=n_results * 5,  # Get more candidates to filter by position
                where=where_filter
            )
            
            similar_players = []
            if similar_results and similar_results.get('documents'):
                for i, doc in enumerate(similar_results['documents'][0]):
                    metadata = similar_results['metadatas'][0][i]
                    candidate_name = metadata.get('player', '')
                    candidate_position = metadata.get('position', '')
                    
                    # Skip the reference player himself
                    if candidate_name.lower() == player_name.lower():
                        continue
                    
                    # Filter by position in Python
                    if position not in candidate_position:
                        continue
                    
                    # Calculate similarity score based on key stats
                    similarity_score = self._calculate_stat_similarity(ref_stats, metadata)
                    
                    similar_players.append({
                        'player': candidate_name,
                        'team': metadata.get('team', 'Unknown'),
                        'age': metadata.get('age', 'Unknown'),
                        'similarity_score': similarity_score,
                        'goals': metadata.get('goals', 0),
                        'assists': metadata.get('assists', 0),
                        'xg': metadata.get('xg', 0),
                        'xa': metadata.get('xa', 0),
                        'minutes': metadata.get('minutes', 0)
                    })
            
            # Sort by similarity score and return top N
            similar_players.sort(key=lambda x: x['similarity_score'], reverse=True)
            return similar_players[:n_results]
            
        except Exception as e:
            logger.error(f"Error finding similar players: {e}")
            return []
    
    def _calculate_stat_similarity(self, ref_stats: Dict, candidate_stats: Dict) -> float:
        """Calculate statistical similarity between two players (0-1 score)"""
        try:
            # Key stats to compare (adjust based on position)
            stat_keys = ['goals', 'assists', 'xg', 'xa', 'shots', 'passes_completed', 'tackles', 'interceptions']
            
            similarity_sum = 0
            count = 0
            
            for key in stat_keys:
                ref_val = float(ref_stats.get(key, 0))
                cand_val = float(candidate_stats.get(key, 0))
                
                # Skip if both are 0
                if ref_val == 0 and cand_val == 0:
                    continue
                
                # Calculate percentage difference
                max_val = max(ref_val, cand_val)
                min_val = min(ref_val, cand_val)
                
                if max_val > 0:
                    stat_similarity = min_val / max_val
                    similarity_sum += stat_similarity
                    count += 1
            
            # Return average similarity
            return similarity_sum / count if count > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def query(self, query_text: str, intent: Optional[Any] = None, intent_metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process query through CRAG workflow with intent awareness
        
        Args:
            query_text: User query
            intent: Query intent (from IntentClassifier)
            intent_metadata: Intent metadata dict
            
        Returns:
            Dictionary with answer, sources, confidence
        """
        initial_state = {
            "query": query_text,
            "retrieved_docs": [],
            "grade": "",
            "web_results": None,
            "final_answer": "",
            "sources": [],
            "confidence": 0.0,
            "intent": intent,
            "intent_metadata": intent_metadata or {},
            "reasoning_trace": None
        }
        
        try:
            # Run workflow
            final_state = self.workflow.invoke(initial_state)
            
            # Determine data source
            used_web = final_state.get("web_results") is not None and len(final_state.get("web_results", [])) > 0
            data_source = "Web Search" if used_web else "Database"
            if used_web and len(final_state.get("retrieved_docs", [])) > 0:
                data_source = "Web+DB"
            
            # Extract intent as string (handle enum)
            intent_value = final_state.get("intent")
            if hasattr(intent_value, 'value'):
                intent_str = intent_value.value
            elif intent_value is not None:
                intent_str = str(intent_value)
            else:
                intent_str = "unknown"
            
            # Get intent confidence from metadata
            intent_conf = 0.0
            if intent_metadata:
                intent_conf = intent_metadata.get('confidence', 0.0)
            
            return {
                "answer": final_state["final_answer"],
                "sources": final_state["sources"],
                "confidence": final_state["confidence"],
                "grade": final_state["grade"],
                "used_web_search": used_web,
                "data_source": data_source,
                "reasoning_trace": final_state.get("reasoning_trace", ""),
                # Include raw retrieved docs (content + metadata) so callers can render full stat tables
                "retrieved_docs": final_state.get("retrieved_docs", []),
                # Include intent for proper UI rendering
                "intent": intent_str,
                "intent_confidence": intent_conf,
                "intent_metadata": final_state.get("intent_metadata", {})
            }
        except Exception as e:
            logger.error(f"CRAG workflow error: {e}")
            return {
                "answer": f"Error processing query: {str(e)}",
                "sources": [],
                "confidence": 0.0,
                "grade": "error",
                "used_web_search": False,
                "data_source": "Error",
                "reasoning_trace": str(e),
                "retrieved_docs": []
            }
