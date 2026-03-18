"""
Intent-Based UI Renderers
=========================

Streamlit renderers for each intent type to deliver adaptive UX.
"""

import streamlit as st
from typing import Dict, Any, List
from loguru import logger

try:
    from src.agents.intent_classifier import QueryIntent
except ImportError:
    logger.warning("IntentClassifier not available")
    QueryIntent = None


def render_scout_report(result: Dict[str, Any], query: str):
    """
    Render full scout report (flagship format)
    Time: ~20s | Length: 2000+ words
    """
    st.markdown("### 📋 Scout Report")
    st.caption(f"Query: {query}")
    
    # Main report (no duplicate metrics - they're shown above by app)
    answer = result.get('answer', 'No answer generated')
    st.markdown(answer)
    
    # Sources section with excerpts
    if result.get('sources'):
        st.markdown("---")
        st.markdown("### 📚 Sources")
        for i, source in enumerate(result['sources'][:10], 1):
            st.markdown(f"**{i}. {source.get('player', 'Unknown')}**")
            st.caption(f"🏟️ {source.get('team', 'N/A')} | 📅 {source.get('season', 'N/A')} | Relevance: {source.get('relevance', 0.0):.0%}")
            
            # Show excerpt/content if available
            excerpt = source.get('content', '')
            if excerpt:
                st.text_area(
                    "Excerpt",
                    value=excerpt,
                    height=80,
                    key=f"excerpt_{i}",
                    label_visibility="collapsed"
                )
            st.markdown("")  # spacing


def render_evaluation(result: Dict[str, Any], query: str):
    """
    Render quick evaluation (150 words)
    Time: ~5s | Format: 4 sections with 3-5 stats
    """
    st.markdown("### ⚡ Quick Evaluation")
    st.caption(f"Query: {query}")
    
    # Confidence badge
    confidence = result.get('confidence', 0.0)
    if confidence >= 0.9:
        st.success(f"High confidence ({confidence:.0%})")
    elif confidence >= 0.7:
        st.info(f"Medium confidence ({confidence:.0%})")
    else:
        st.warning(f"Low confidence ({confidence:.0%})")
    
    # Answer
    answer = result.get('answer', 'No evaluation available')
    st.markdown(answer)
    
    # Compact sources
    sources = result.get('sources', [])
    if sources:
        st.caption(f"📊 Based on {len(sources)} source(s)")


def render_comparison(result: Dict[str, Any], query: str):
    """
    Render side-by-side comparison (table format)
    Time: ~8s | Format: Table with tactical differences
    """
    st.markdown("### ⚖️ Player Comparison")
    st.caption(f"Query: {query}")
    
    # Answer (should contain markdown table)
    answer = result.get('answer', 'No comparison available')
    st.markdown(answer)
    
    # Show sources grouped by player
    sources = result.get('sources', [])
    if sources:
        st.markdown("---")
        st.markdown("**Data Sources:**")
        
        # Group by player
        player_sources = {}
        for source in sources:
            player = source.get('player', 'Unknown')
            if player not in player_sources:
                player_sources[player] = []
            player_sources[player].append(source)
        
        cols = st.columns(len(player_sources))
        for i, (player, srcs) in enumerate(player_sources.items()):
            with cols[i]:
                st.markdown(f"**{player}**")
                for src in srcs[:3]:
                    st.caption(f"📅 {src.get('season', 'N/A')} @ {src.get('team', 'N/A')}")


def render_tactical_fit(result: Dict[str, Any], query: str):
    """
    Render tactical fit analysis (system compatibility)
    Time: ~7s | Format: Requirements + fit score
    """
    st.markdown("### 🎯 Tactical Fit Analysis")
    st.caption(f"Query: {query}")
    
    # Extract fit score from answer if present (pattern: X/10)
    answer = result.get('answer', 'No analysis available')
    import re
    fit_match = re.search(r'(\d+(?:\.\d+)?)\s*/\s*10', answer)
    
    if fit_match:
        fit_score = float(fit_match.group(1))
        st.metric("Fit Score", f"{fit_score}/10", delta=f"{fit_score - 5:.1f} vs average")
    
    st.markdown(answer)
    
    # Sources
    sources_count = len(result.get('sources', []))
    if sources_count > 0:
        st.caption(f"📊 Analysis based on {sources_count} data point(s)")


def render_stat_query(result: Dict[str, Any], query: str):
    """
    Render direct stat answer (minimal format)
    Time: ~3s | Format: Direct answer + 1 context sentence
    """
    st.markdown("### 📊 Stat Answer")
    
    # Answer in large font
    answer = result.get('answer', 'No data found')
    st.markdown(f"#### {answer}")
    
    # Show sources inline
    sources = result.get('sources', [])
    if sources:
        source = sources[0]
        st.caption(f"📅 {source.get('season', 'N/A')} | 🏟️ {source.get('team', 'N/A')}")


def render_trend_analysis(result: Dict[str, Any], query: str):
    """
    Render multi-season trend (table with indicators)
    Time: ~10s | Format: Season-by-season table with ↑↓→
    """
    st.markdown("### 📈 Trend Analysis")
    st.caption(f"Query: {query}")
    
    # Answer (should contain table with trend indicators)
    answer = result.get('answer', 'No trend data available')
    st.markdown(answer)
    
    # Show season range
    sources = result.get('sources', [])
    if sources:
        seasons = [s.get('season', '') for s in sources if s.get('season')]
        if seasons:
            unique_seasons = sorted(set(seasons))
            st.info(f"📅 Seasons analyzed: {', '.join(unique_seasons)}")


def render_team_analysis(result: Dict[str, Any], query: str):
    """
    Render team/squad analysis
    Time: ~12s | Format: Squad overview with key players
    """
    st.markdown("### 🏟️ Team Analysis")
    st.caption(f"Query: {query}")
    
    answer = result.get('answer', 'No team data available')
    st.markdown(answer)
    
    # Show sources by player
    sources = result.get('sources', [])
    if sources:
        st.markdown("---")
        st.markdown("**Players Analyzed:**")
        
        players = list(set([s.get('player', 'Unknown') for s in sources]))
        cols = st.columns(min(4, len(players)))
        for i, player in enumerate(players[:8]):
            with cols[i % 4]:
                st.caption(f"• {player}")


def render_transfer_value(result: Dict[str, Any], query: str):
    """
    Render market valuation analysis
    Time: ~8s | Format: Fair value range + factors
    """
    st.markdown("### 💰 Transfer Valuation")
    st.caption(f"Query: {query}")
    
    # Extract value range if present (pattern: €XM - €YM)
    answer = result.get('answer', 'No valuation available')
    import re
    value_match = re.search(r'€(\d+(?:\.\d+)?)[Mm]\s*-\s*€(\d+(?:\.\d+)?)[Mm]', answer)
    
    if value_match:
        low = float(value_match.group(1))
        high = float(value_match.group(2))
        mid = (low + high) / 2
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Low End", f"€{low}M")
        with col2:
            st.metric("Fair Value", f"€{mid}M")
        with col3:
            st.metric("High End", f"€{high}M")
    
    st.markdown(answer)
    
    # Show confidence
    confidence = result.get('confidence', 0.0)
    st.caption(f"💡 Confidence: {confidence:.0%}")


def render_unknown(result: Dict[str, Any], query: str):
    """
    Fallback renderer for unclassified queries
    """
    st.warning("⚠️ Could not classify query intent. Using generic format.")
    st.markdown("### 💬 Answer")
    st.caption(f"Query: {query}")
    
    answer = result.get('answer', 'No answer generated')
    st.markdown(answer)
    
    # Sources
    if result.get('sources'):
        with st.expander("📚 Sources"):
            for i, source in enumerate(result['sources'][:5], 1):
                st.markdown(f"{i}. **{source.get('player', 'Unknown')}** ({source.get('season', 'N/A')})")


# ========== INTENT ROUTER ==========
def render_result(result: Dict[str, Any], query: str):
    """
    Main router: dispatches to appropriate renderer based on intent
    
    Args:
        result: Query result dict with 'intent' field
        query: Original user query
    """
    intent = result.get('intent', 'unknown')
    
    logger.info(f"🎨 Rendering intent: {intent}")
    
    # Map intent to renderer
    RENDERER_MAP = {
        'scout_report': render_scout_report,
        'evaluation': render_evaluation,
        'comparison': render_comparison,
        'tactical_fit': render_tactical_fit,
        'stat_query': render_stat_query,
        'trend_analysis': render_trend_analysis,
        'team_analysis': render_team_analysis,
        'transfer_value': render_transfer_value,
        'unknown': render_unknown
    }
    
    # Get renderer (default to scout report for backwards compatibility)
    renderer = RENDERER_MAP.get(intent, render_scout_report)
    
    # Render
    try:
        renderer(result, query)
    except Exception as e:
        logger.error(f"Rendering error: {e}")
        st.error(f"Error rendering result: {e}")
        # Fallback to basic display
        st.markdown("### Answer")
        st.markdown(result.get('answer', 'No answer available'))


# ========== CONFIDENCE INDICATORS ==========
def show_query_insights(result: Dict[str, Any]):
    """
    Show query classification insights (debug/transparency)
    """
    with st.expander("🔍 Query Analysis", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            intent = result.get('intent', 'unknown')
            intent_conf = result.get('intent_confidence', 0.0)
            st.metric("Detected Intent", intent, delta=f"{intent_conf:.0%} confidence")
        
        with col2:
            reasoning = result.get('used_web_search', False)
            reasoning_label = "Used Web Search" if reasoning else "DB Only"
            st.metric("Reasoning", reasoning_label)
        
        # Show metadata
        metadata = result.get('intent_metadata', {})
        if metadata:
            st.json(metadata)
