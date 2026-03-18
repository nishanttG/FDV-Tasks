"""
Base Agent Class
================

Abstract base class for all TactIQ agents.
Provides common functionality for query processing, retrieval, and response generation.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class AgentResponse:
    """Structured response from an agent"""
    agent_name: str
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    confidence: float


class BaseAgent(ABC):
    """Abstract base class for TactIQ agents"""
    
    def __init__(self, vector_db, llm=None):
        """
        Initialize base agent
        
        Args:
            vector_db: VectorDatabase instance for retrieval
            llm: Optional LLM instance for answer generation
        """
        self.vector_db = vector_db
        self.llm = llm
        self.agent_name = self.__class__.__name__
        
        logger.info(f"Initialized {self.agent_name}")
    
    @abstractmethod
    def can_handle(self, query: str) -> bool:
        """
        Determine if this agent can handle the query
        
        Args:
            query: User query string
            
        Returns:
            bool: True if agent can handle, False otherwise
        """
        pass
    
    @abstractmethod
    def retrieve(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for the query
        
        Args:
            query: User query string
            **kwargs: Additional retrieval parameters
            
        Returns:
            List of retrieved documents with metadata
        """
        pass
    
    @abstractmethod
    def process(self, query: str, **kwargs) -> AgentResponse:
        """
        Process query and generate response
        
        Args:
            query: User query string
            **kwargs: Additional processing parameters
            
        Returns:
            AgentResponse with answer and sources
        """
        pass
    
    def _deduplicate_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deduplicate retrieved results based on (player, team, season)
        
        Args:
            results: ChromaDB query results
            
        Returns:
            Deduplicated results
        """
        if not results or not results.get('documents'):
            return results
        
        seen = {}
        
        for doc, meta, dist in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            # Create unique key based on document type
            if meta.get('type') == 'player_stats':
                key = (meta.get('player'), meta.get('team'), meta.get('season'))
            else:
                key = (meta.get('url'), meta.get('chunk_id'))
            
            # Keep result with lowest distance (highest similarity)
            if key not in seen or dist < seen[key][2]:
                seen[key] = (doc, meta, dist)
        
        # Convert back to lists
        deduped = list(seen.values())
        results['documents'][0] = [x[0] for x in deduped]
        results['metadatas'][0] = [x[1] for x in deduped]
        results['distances'][0] = [x[2] for x in deduped]
        
        return results
    
    def _apply_age_filter(self, results: Dict[str, Any], min_age: Optional[int] = None, 
                         max_age: Optional[int] = None) -> Dict[str, Any]:
        """
        Apply age filtering to results
        
        Args:
            results: ChromaDB query results
            min_age: Minimum age threshold
            max_age: Maximum age threshold
            
        Returns:
            Filtered results
        """
        if not results or not results.get('documents'):
            return results
        
        if min_age is None and max_age is None:
            return results
        
        filtered_docs = []
        filtered_metas = []
        filtered_dists = []
        
        for doc, meta, dist in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            try:
                age = int(meta.get('age', 0))
                
                if min_age is not None and age < min_age:
                    continue
                if max_age is not None and age > max_age:
                    continue
                
                filtered_docs.append(doc)
                filtered_metas.append(meta)
                filtered_dists.append(dist)
                
            except (ValueError, TypeError):
                pass
        
        results['documents'][0] = filtered_docs
        results['metadatas'][0] = filtered_metas
        results['distances'][0] = filtered_dists
        
        return results
    
    def _format_sources(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Format retrieved results as source citations
        
        Args:
            results: ChromaDB query results
            
        Returns:
            List of formatted source dictionaries
        """
        sources = []
        
        if not results or not results.get('documents'):
            return sources
        
        for doc, meta, dist in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            similarity = 1 - dist
            
            source = {
                'content': doc,
                'metadata': meta,
                'similarity': similarity,
                'type': meta.get('type', 'unknown')
            }
            
            sources.append(source)
        
        return sources
