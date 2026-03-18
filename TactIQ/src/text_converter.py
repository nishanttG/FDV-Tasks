"""
Table-to-Text Conversion Utilities
Converts structured data (tables) into natural language descriptions for embedding
"""

import pandas as pd
from typing import List, Dict, Any, Optional
from loguru import logger


class TableToTextConverter:
    """Converts structured player/team data into natural language descriptions"""
    
    @staticmethod
    def create_player_description(row: pd.Series, include_stats: bool = True) -> str:
        """
        Convert a player stats row into natural language description
        
        Args:
            row: Pandas Series with player data
            include_stats: Whether to include detailed stats
            
        Returns:
            Natural language description string
        """
        parts = []
        
        # Basic info
        player_name = row.get('Player', 'Unknown Player')
        age = row.get('Age', 'N/A')
        position = row.get('Pos', 'player')
        squad = row.get('Squad', 'Unknown Club')
        league = row.get('Comp', 'Unknown League')
        
        parts.append(f"{player_name} is a {age} year old {position} playing for {squad} in the {league}")
        
        if include_stats:
            # Performance stats
            if pd.notna(row.get('Gls')):
                parts.append(f"with {row.get('Gls', 0)} goals")
            if pd.notna(row.get('Ast')):
                parts.append(f"and {row.get('Ast', 0)} assists")
            if pd.notna(row.get('90s')):
                parts.append(f"in {row.get('90s', 0):.1f} matches (90s)")
            
            # Additional stats
            if pd.notna(row.get('xG')):
                parts.append(f"xG: {row.get('xG', 0):.2f}")
            if pd.notna(row.get('xAG')):
                parts.append(f"xAG: {row.get('xAG', 0):.2f}")
        
        # Market value
        if pd.notna(row.get('market_value_eur')):
            value_m = row.get('market_value_eur', 0) / 1_000_000
            parts.append(f"Market value: €{value_m:.1f}M")
        
        # Nationality
        if pd.notna(row.get('Nation')):
            parts.append(f"Nationality: {row.get('Nation', 'N/A')}")
        
        # Season
        if pd.notna(row.get('Season')):
            parts.append(f"Season: {row.get('Season', 'N/A')}")
        
        description = ", ".join(parts) + "."
        return description
    
    @staticmethod
    def create_team_description(row: pd.Series) -> str:
        """
        Convert team stats into natural language description
        
        Args:
            row: Pandas Series with team data
            
        Returns:
            Natural language description string
        """
        parts = []
        
        team_name = row.get('Squad', 'Unknown Team')
        league = row.get('Comp', 'Unknown League')
        season = row.get('Season', 'N/A')
        
        parts.append(f"{team_name} competed in {league} during the {season} season")
        
        if pd.notna(row.get('Pts')):
            parts.append(f"earning {row.get('Pts', 0)} points")
        
        if pd.notna(row.get('GF')):
            parts.append(f"with {row.get('GF', 0)} goals scored")
        
        if pd.notna(row.get('GA')):
            parts.append(f"and {row.get('GA', 0)} goals conceded")
        
        description = ", ".join(parts) + "."
        return description
    
    @staticmethod
    def convert_player_stats(
        df: pd.DataFrame,
        stat_type: str = "standard",
        max_rows: Optional[int] = None
    ) -> List[str]:
        """
        Convert entire player stats dataframe to list of descriptions
        
        Args:
            df: Player stats dataframe
            stat_type: Type of stats (standard, shooting, passing, etc.)
            max_rows: Maximum number of rows to convert (None for all)
            
        Returns:
            List of natural language descriptions
        """
        logger.info(f"Converting {len(df)} {stat_type} player stats to text...")
        
        if max_rows:
            df = df.head(max_rows)
        
        descriptions = []
        for idx, row in df.iterrows():
            try:
                desc = TableToTextConverter.create_player_description(row, include_stats=True)
                descriptions.append(desc)
            except Exception as e:
                logger.warning(f"Failed to convert row {idx}: {e}")
                continue
        
        logger.info(f"✓ Converted {len(descriptions)} descriptions")
        return descriptions
    
    @staticmethod
    def convert_team_stats(
        df: pd.DataFrame,
        max_rows: Optional[int] = None
    ) -> List[str]:
        """
        Convert team stats dataframe to list of descriptions
        
        Args:
            df: Team stats dataframe
            max_rows: Maximum number of rows to convert (None for all)
            
        Returns:
            List of natural language descriptions
        """
        logger.info(f"Converting {len(df)} team stats to text...")
        
        if max_rows:
            df = df.head(max_rows)
        
        descriptions = []
        for idx, row in df.iterrows():
            try:
                desc = TableToTextConverter.create_team_description(row)
                descriptions.append(desc)
            except Exception as e:
                logger.warning(f"Failed to convert row {idx}: {e}")
                continue
        
        logger.info(f"✓ Converted {len(descriptions)} descriptions")
        return descriptions
    
    @staticmethod
    def create_comprehensive_player_profile(
        player_data: Dict[str, Any],
        include_tactical: bool = True
    ) -> str:
        """
        Create comprehensive player profile combining multiple data sources
        
        Args:
            player_data: Dictionary with player information
            include_tactical: Whether to include tactical analysis
            
        Returns:
            Comprehensive player description
        """
        parts = []
        
        # Basic info
        name = player_data.get('name', 'Unknown Player')
        age = player_data.get('age', 'N/A')
        position = player_data.get('position', 'Unknown')
        
        parts.append(f"Player Profile: {name}")
        parts.append(f"Age: {age}, Position: {position}")
        
        # Club & league
        if 'club' in player_data:
            parts.append(f"Current Club: {player_data['club']}")
        if 'league' in player_data:
            parts.append(f"League: {player_data['league']}")
        
        # Performance stats
        if 'stats' in player_data:
            stats = player_data['stats']
            stat_parts = []
            
            if 'goals' in stats:
                stat_parts.append(f"{stats['goals']} goals")
            if 'assists' in stats:
                stat_parts.append(f"{stats['assists']} assists")
            if 'matches' in stats:
                stat_parts.append(f"{stats['matches']} matches")
            
            if stat_parts:
                parts.append(f"Season Performance: {', '.join(stat_parts)}")
        
        # Market value
        if 'market_value' in player_data:
            value = player_data['market_value']
            parts.append(f"Market Value: €{value/1_000_000:.1f}M")
        
        # Tactical attributes
        if include_tactical and 'tactical' in player_data:
            tactical = player_data['tactical']
            parts.append(f"Tactical Profile: {tactical.get('style', 'N/A')}")
            if 'strengths' in tactical:
                parts.append(f"Strengths: {', '.join(tactical['strengths'])}")
        
        description = ". ".join(parts) + "."
        return description


# Convenience functions
def player_to_text(row: pd.Series) -> str:
    """Convert player row to text (convenience wrapper)"""
    return TableToTextConverter.create_player_description(row)


def team_to_text(row: pd.Series) -> str:
    """Convert team row to text (convenience wrapper)"""
    return TableToTextConverter.create_team_description(row)
