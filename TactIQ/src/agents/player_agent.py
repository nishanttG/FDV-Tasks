"""
Player Agent
============

Specialized agent for handling player-specific queries:
- Player search and recommendations
- Statistical comparisons
- Performance analysis
- Transfer market insights
"""

from typing import Dict, List, Any, Optional
import re
from loguru import logger

from .base_agent import BaseAgent, AgentResponse


class PlayerAgent(BaseAgent):
    """Agent specialized in player queries and analysis"""
    
    # Keywords that indicate player queries
    PLAYER_KEYWORDS = [
        'player', 'striker', 'midfielder', 'defender', 'goalkeeper', 'winger',
        'forward', 'playmaker', 'talent', 'prospect', 'veteran',
        'goals', 'assists', 'passes', 'tackles', 'saves',
        'young', 'experienced', 'under', 'over', 'age',
        'premier league', 'la liga', 'serie a', 'bundesliga', 'ligue 1',
        'market value', 'transfer', 'squad', 'team'
    ]
    
    # Temporal keywords for recency detection
    TEMPORAL_KEYWORDS = [
        'this season', 'current season', 'recent', 'recently', 'latest',
        'now', 'today', 'current', 'so far', '2024', '2025'
    ]
    
    def __init__(self, vector_db, llm=None):
        super().__init__(vector_db, llm)
        logger.info("PlayerAgent initialized with hybrid search capabilities")
    
    def can_handle(self, query: str) -> bool:
        """
        Check if query is about players
        
        Args:
            query: User query string
            
        Returns:
            True if query contains player-related keywords or player names
        """
        query_lower = query.lower()
        
        # Check for player keywords
        if any(keyword in query_lower for keyword in self.PLAYER_KEYWORDS):
            return True
        
        # Check for temporal keywords + capitalized words (player names)
        if any(keyword in query_lower for keyword in self.TEMPORAL_KEYWORDS):
            words = query.split()
            capitalized_words = [w for w in words if w and w[0].isupper() and len(w) > 1]
            if len(capitalized_words) >= 1:
                return True
        
        # Check for proper nouns (potential player names)
        words = query.split()
        capitalized_words = [w for w in words if w and w[0].isupper() and len(w) > 2]
        if len(capitalized_words) >= 2:
            return True
            
        return False
    
    def _extract_player_name(self, query: str) -> str:
        """Extract potential player name from query."""
        # Look for patterns like "Mo. Salah", "Mohamed Salah", "Cristiano Ronaldo"
        words = query.split()
        potential_names = []
        
        i = 0
        while i < len(words):
            word = words[i]
            # Check if word starts with capital and is not a common word
            if word and word[0].isupper() and len(word) > 1:
                # Skip common non-name words
                if word.lower() not in ['how', 'what', 'who', 'which', 'when', 'where', 'why', 'is', 'are', 'been', 'the']:
                    name_parts = [word]
                    # Look ahead for more capitalized words
                    j = i + 1
                    while j < len(words) and words[j] and words[j][0].isupper():
                        if words[j].lower() not in ['how', 'what', 'who', 'which', 'when', 'where', 'why']:
                            name_parts.append(words[j])
                            j += 1
                        else:
                            break
                    if len(name_parts) >= 1:
                        potential_names.append(' '.join(name_parts))
                        i = j
                        continue
            i += 1
        
        # Return the longest potential name found
        return max(potential_names, key=len) if potential_names else ''
    
    def _extract_filters(self, query: str) -> Dict[str, Any]:
        """
        Extract metadata filters from query
        
        Args:
            query: User query string
            
        Returns:
            Dictionary of filters (position, league, age constraints)
        """
        filters = {}
        query_lower = query.lower()
        
        # Position detection
        positions = []
        if any(word in query_lower for word in ['striker', 'forward', 'attacker']):
            positions.extend(['FW', 'FW,MF'])
        if any(word in query_lower for word in ['midfielder', 'playmaker']):
            positions.extend(['MF', 'MF,FW', 'MF,DF'])
        if any(word in query_lower for word in ['defender', 'centre-back', 'full-back']):
            positions.extend(['DF', 'DF,MF'])
        if any(word in query_lower for word in ['goalkeeper', 'keeper']):
            positions.append('GK')
        
        if positions:
            filters['position'] = {'$in': positions}
        
        # League detection
        league_map = {
            'premier league': 'ENG-Premier League',
            'la liga': 'ESP-La Liga',
            'serie a': 'ITA-Serie A',
            'bundesliga': 'GER-Bundesliga',
            'ligue 1': 'FRA-Ligue 1'
        }
        
        for league_name, league_code in league_map.items():
            if league_name in query_lower:
                filters['league'] = league_code
                break
        
        # Age detection
        age_match = re.search(r'under (\d+)', query_lower)
        if age_match:
            filters['max_age'] = int(age_match.group(1))
        
        age_match = re.search(r'over (\d+)', query_lower)
        if age_match:
            filters['min_age'] = int(age_match.group(1))
        
        return filters
    
    def retrieve(self, query: str, top_k: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieve relevant player documents
        
        Args:
            query: User query string
            top_k: Number of results to retrieve
            **kwargs: Additional parameters
            
        Returns:
            List of retrieved documents
        """
        # Extract potential player name
        player_name = self._extract_player_name(query)
        if player_name:
            logger.info(f"Detected player name: {player_name}")
            # Enhance query with player name
            query = f"{player_name} {query}"
        
        # Extract filters
        filters = self._extract_filters(query)
        
        # Build where clause
        where_clauses = [{"type": "player_stats"}]
        
        if 'position' in filters:
            where_clauses.append({"position": filters['position']})
        
        if 'league' in filters:
            where_clauses.append({"league": filters['league']})
        
        where = {"$and": where_clauses} if len(where_clauses) > 1 else where_clauses[0]
        
        # Query vector database
        results = self.vector_db.query(
            query_text=query,
            n_results=top_k,
            where=where
        )
        
        # Deduplicate
        results = self._deduplicate_results(results)
        
        # Apply age filtering
        if 'min_age' in filters or 'max_age' in filters:
            results = self._apply_age_filter(
                results,
                min_age=filters.get('min_age'),
                max_age=filters.get('max_age')
            )
        
        # Format as sources
        sources = self._format_sources(results)
        
        logger.info(f"Retrieved {len(sources)} player documents")
        return sources
    
    def _rank_by_attribute(self, sources: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """
        Re-rank results based on query intent (e.g., "high market value", "top scorer")
        
        Args:
            sources: Retrieved sources
            query: User query
            
        Returns:
            Re-ranked sources
        """
        query_lower = query.lower()
        
        # Detect ranking intent
        if any(phrase in query_lower for phrase in ['high value', 'expensive', 'market value']):
            # Sort by market value (would need to extract from metadata)
            logger.info("Re-ranking by market value")
            # TODO: Implement when market_value is in metadata
        
        elif any(phrase in query_lower for phrase in ['top scorer', 'goal', 'goals']):
            logger.info("Re-ranking by goals")
            # TODO: Implement when goals are in metadata
        
        return sources
    
    def process(self, query: str, **kwargs) -> AgentResponse:
        """
        Process player query and generate response
        
        Args:
            query: User query string
            **kwargs: Additional parameters
            
        Returns:
            AgentResponse with player recommendations
        """
        logger.info(f"Processing player query: {query}")
        
        # Retrieve relevant players
        sources = self.retrieve(query, top_k=10)
        
        # Apply intent-based re-ranking
        sources = self._rank_by_attribute(sources, query)
        
        # Limit to top 5
        sources = sources[:5]
        
        # Generate answer (simple version without LLM for now)
        if not sources:
            answer = "No players found matching your criteria."
            confidence = 0.0
        else:
            answer = self._format_player_answer(sources, query)
            confidence = sum(s['similarity'] for s in sources) / len(sources)
        
        return AgentResponse(
            agent_name=self.agent_name,
            query=query,
            answer=answer,
            sources=sources,
            metadata={'filters_applied': self._extract_filters(query)},
            confidence=confidence
        )
    
    def _format_player_answer(self, sources: List[Dict[str, Any]], query: str) -> str:
        """
        Format player sources into readable answer
        
        Args:
            sources: Retrieved player sources
            query: Original query
            
        Returns:
            Formatted answer string
        """
        answer_parts = [f"Based on your query '{query}', here are the top players:\n"]
        
        for i, source in enumerate(sources, 1):
            meta = source['metadata']
            sim = source['similarity']
            
            answer_parts.append(
                f"\n{i}. {meta.get('player', 'Unknown')} "
                f"({meta.get('position', 'N/A')}, Age {meta.get('age', 'N/A')})\n"
                f"   Team: {meta.get('team', 'N/A')} | {meta.get('league', 'N/A')}\n"
                f"   Season: {meta.get('season', 'N/A')} | Relevance: {sim:.1%}"
            )
        
        return "".join(answer_parts)
