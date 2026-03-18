"""
Football Player Visualization Utilities
======================================

Create charts and plots for player performance analysis
"""

import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any
import pandas as pd
from loguru import logger


def create_radar_chart(player_stats: Dict[str, Any], player_name: str) -> go.Figure:
    """
    Create a radar chart for player performance metrics
    
    Args:
        player_stats: Dictionary with player statistics
        player_name: Name of the player
        
    Returns:
        Plotly figure object
    """
    try:
        # Define categories for radar chart based on position
        position = player_stats.get('position', 'FW')
        
        if 'FW' in position:
            # Forward metrics
            categories = ['Goals', 'xG', 'Assists', 'xA', 'Shots', 'Dribbles']
            values = [
                float(player_stats.get('goals', 0)),
                float(player_stats.get('xg', 0)),
                float(player_stats.get('assists', 0)),
                float(player_stats.get('xa', 0)),
                float(player_stats.get('shots', 0)) / 10,  # Normalize
                float(player_stats.get('dribbles_completed', 0)) / 5  # Normalize
            ]
        elif 'MF' in position:
            # Midfielder metrics
            categories = ['Goals', 'Assists', 'Passes', 'Key Passes', 'Tackles', 'Interceptions']
            values = [
                float(player_stats.get('goals', 0)),
                float(player_stats.get('assists', 0)),
                float(player_stats.get('passes_completed', 0)) / 100,  # Normalize
                float(player_stats.get('passes_key', 0)),
                float(player_stats.get('tackles', 0)) / 10,  # Normalize
                float(player_stats.get('interceptions', 0)) / 5  # Normalize
            ]
        else:
            # Defender metrics
            categories = ['Tackles', 'Interceptions', 'Clearances', 'Blocks', 'Passes', 'Duels Won']
            values = [
                float(player_stats.get('tackles', 0)) / 10,
                float(player_stats.get('interceptions', 0)) / 5,
                float(player_stats.get('clearances', 0)) / 10,
                float(player_stats.get('blocks', 0)) / 5,
                float(player_stats.get('passes_completed', 0)) / 100,
                float(player_stats.get('duels_won_pct', 50)) / 10  # Normalize
            ]
        
        # Create radar chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=player_name,
            line=dict(color='#1f77b4', width=2),
            fillcolor='rgba(31, 119, 180, 0.3)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(values) * 1.2] if max(values) > 0 else [0, 10]
                )
            ),
            showlegend=True,
            title=f"{player_name} Performance Radar",
            height=500
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating radar chart: {e}")
        return None


def create_comparison_chart(players_data: List[Dict[str, Any]], metric: str = 'goals') -> go.Figure:
    """
    Create a bar chart comparing multiple players on a specific metric
    
    Args:
        players_data: List of player dictionaries with stats
        metric: Metric to compare (e.g., 'goals', 'assists', 'xg')
        
    Returns:
        Plotly figure object
    """
    try:
        # Extract names and values
        names = [p.get('player', 'Unknown') for p in players_data]
        values = [float(p.get(metric, 0)) for p in players_data]
        
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=names,
                y=values,
                marker=dict(
                    color=values,
                    colorscale='Blues',
                    showscale=True
                ),
                text=values,
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title=f"Player Comparison: {metric.replace('_', ' ').title()}",
            xaxis_title="Player",
            yaxis_title=metric.replace('_', ' ').title(),
            height=400,
            showlegend=False
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating comparison chart: {e}")
        return None


def create_trend_chart(player_seasons: List[Dict[str, Any]], player_name: str) -> go.Figure:
    """
    Create a line chart showing player performance trends across seasons
    
    Args:
        player_seasons: List of season stats for a player
        player_name: Name of the player
        
    Returns:
        Plotly figure object
    """
    try:
        if not player_seasons:
            return None
        
        # Sort by season
        player_seasons = sorted(player_seasons, key=lambda x: x.get('season', ''))
        
        seasons = [p.get('season', '') for p in player_seasons]
        goals = [float(p.get('goals', 0)) for p in player_seasons]
        assists = [float(p.get('assists', 0)) for p in player_seasons]
        xg = [float(p.get('xg', 0)) for p in player_seasons]
        xa = [float(p.get('xa', 0)) for p in player_seasons]
        
        # Create multi-line chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=seasons,
            y=goals,
            mode='lines+markers',
            name='Goals',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=seasons,
            y=xg,
            mode='lines+markers',
            name='xG (Expected Goals)',
            line=dict(color='#ff7f0e', width=2, dash='dash'),
            marker=dict(size=6)
        ))
        
        fig.add_trace(go.Scatter(
            x=seasons,
            y=assists,
            mode='lines+markers',
            name='Assists',
            line=dict(color='#2ca02c', width=3),
            marker=dict(size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=seasons,
            y=xa,
            mode='lines+markers',
            name='xA (Expected Assists)',
            line=dict(color='#d62728', width=2, dash='dash'),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title=f"{player_name} Performance Trend",
            xaxis_title="Season",
            yaxis_title="Count",
            height=500,
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating trend chart: {e}")
        return None


def create_similar_players_table(similar_players: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Create a formatted DataFrame for similar players comparison
    
    Args:
        similar_players: List of similar player dictionaries
        
    Returns:
        Pandas DataFrame
    """
    try:
        if not similar_players:
            return None
        
        df = pd.DataFrame(similar_players)
        
        # Format columns
        display_df = df[[
            'player', 'team', 'age', 'goals', 'assists', 
            'xg', 'xa', 'similarity_score'
        ]].copy()
        
        # Rename for display
        display_df.columns = [
            'Player', 'Team', 'Age', 'Goals', 'Assists',
            'xG', 'xA', 'Similarity %'
        ]
        
        # Format similarity as percentage
        display_df['Similarity %'] = (display_df['Similarity %'] * 100).round(1)
        
        # Round float columns
        for col in ['xG', 'xA']:
            if col in display_df.columns:
                display_df[col] = display_df[col].round(2)
        
        return display_df
        
    except Exception as e:
        logger.error(f"Error creating similar players table: {e}")
        return None
