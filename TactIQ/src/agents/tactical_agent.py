"""
Tactical Agent
==============

Specialized agent for handling tactical and strategic queries:
- Formation analysis
- Playing styles and strategies
- Defensive/offensive tactics
- Coaching insights
"""

from typing import Dict, List, Any
from loguru import logger

from .base_agent import BaseAgent, AgentResponse


class TacticalAgent(BaseAgent):
    """Agent specialized in tactical queries and strategic analysis"""
    
    # Keywords that indicate tactical queries
    TACTICAL_KEYWORDS = [
        'formation', 'tactics', 'strategy', 'system',
        'pressing', 'counter', 'attack', 'defense', 'possession',
        'build-up', 'transition', 'shape', 'structure',
        'high press', 'low block', 'offside trap',
        'full-back', 'wingback', 'pivot', 'false nine',
        'how to', 'why', 'explain', 'analysis'
    ]
    
    def __init__(self, vector_db, llm=None):
        super().__init__(vector_db, llm)
        logger.info("TacticalAgent initialized with blog article search")
    
    def can_handle(self, query: str) -> bool:
        """
        Check if query is about tactics/strategy
        
        Args:
            query: User query string
            
        Returns:
            True if query contains tactical keywords
        """
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in self.TACTICAL_KEYWORDS)
    
    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieve relevant tactical articles/chunks
        
        Args:
            query: User query string
            top_k: Number of results to retrieve
            **kwargs: Additional parameters
            
        Returns:
            List of retrieved article chunks
        """
        # Query vector database (no type filter - let it return best matches)
        results = self.vector_db.query(
            query_text=query,
            n_results=top_k
        )
        
        # Format as sources
        sources = self._format_sources(results)
        
        logger.info(f"Retrieved {len(sources)} tactical documents")
        return sources
    
    def process(self, query: str, **kwargs) -> AgentResponse:
        """
        Process tactical query and generate response
        
        Args:
            query: User query string
            **kwargs: Additional parameters
            
        Returns:
            AgentResponse with tactical insights
        """
        logger.info(f"Processing tactical query: {query}")
        
        # Retrieve relevant articles
        sources = self.retrieve(query, top_k=5)
        
        # Filter to only blog articles if available
        blog_sources = [s for s in sources if s['type'] == 'blog_article']
        if blog_sources:
            sources = blog_sources
        
        # Generate answer
        if not sources:
            answer = "No tactical articles found for your query."
            confidence = 0.0
        else:
            answer = self._format_tactical_answer(sources, query)
            confidence = sum(s['similarity'] for s in sources) / len(sources)
        
        return AgentResponse(
            agent_name=self.agent_name,
            query=query,
            answer=answer,
            sources=sources,
            metadata={'article_count': len(sources)},
            confidence=confidence
        )
    
    def _format_tactical_answer(self, sources: List[Dict[str, Any]], query: str) -> str:
        """
        Format tactical sources into readable answer
        
        Args:
            sources: Retrieved tactical sources
            query: Original query
            
        Returns:
            Formatted answer string
        """
        answer_parts = [f"Tactical insights for: '{query}'\n"]
        
        for i, source in enumerate(sources, 1):
            meta = source['metadata']
            content = source['content']
            sim = source['similarity']
            
            if source['type'] == 'blog_article':
                answer_parts.append(
                    f"\n{i}. {meta.get('title', 'Unknown Article')[:60]}...\n"
                    f"   Source: {meta.get('source', 'N/A')}\n"
                    f"   Relevance: {sim:.1%}\n"
                    f"   Excerpt: {content[:200]}...\n"
                )
            else:
                # Player stat result (less relevant for tactical query)
                answer_parts.append(
                    f"\n{i}. {meta.get('player', 'N/A')} ({meta.get('position', 'N/A')})\n"
                    f"   Context: {content[:150]}...\n"
                )
        
        return "".join(answer_parts)
