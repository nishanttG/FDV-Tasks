"""
TactIQ Agent Framework
======================

Multi-agent system for football scouting with RAG capabilities.

Agents:
- PlayerAgent: Handles player-specific queries (stats, comparisons, recommendations)
- TacticalAgent: Handles tactical queries (formations, strategies, analysis)
- OrchestratorAgent: Routes queries and coordinates multi-agent responses
"""

from .player_agent import PlayerAgent
from .tactical_agent import TacticalAgent
from .orchestrator import OrchestratorAgent

__all__ = ['PlayerAgent', 'TacticalAgent', 'OrchestratorAgent']
