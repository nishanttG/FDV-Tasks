"""
Table-to-Text Converter
Converts structured football statistics into descriptive text for better RAG retrieval
"""

import pandas as pd
from typing import Dict, List, Optional
from loguru import logger


class TableToTextConverter:
    """Converts statistical tables into natural language descriptions"""
    
    def __init__(self):
        """Initialize converter with formatting templates"""
        self.player_templates = {
            "standard": self._describe_standard_stats,
            "shooting": self._describe_shooting_stats,
            "passing": self._describe_passing_stats,
            "defense": self._describe_defense_stats,
        }
    
    def _safe_get(self, row: pd.Series, col: str, default="N/A") -> str:
        """Safely get column value with fallback"""
        try:
            val = row.get(col, default)
            if pd.isna(val):
                return default
            return str(val)
        except:
            return default
    
    def _describe_standard_stats(self, row: pd.Series) -> str:
        """
        Convert standard player stats to descriptive text
        
        Args:
            row: DataFrame row with player stats
            
        Returns:
            Natural language description
        """
        player = self._safe_get(row, 'Player', row.name if hasattr(row, 'name') else 'Unknown')
        team = self._safe_get(row, 'Squad', 'Unknown Team')
        position = self._safe_get(row, 'Pos', 'Unknown')
        age = self._safe_get(row, 'Age', 'Unknown')
        
        matches = self._safe_get(row, 'MP', '0')
        starts = self._safe_get(row, 'Starts', '0')
        minutes = self._safe_get(row, 'Min', '0')
        
        goals = self._safe_get(row, 'Gls', '0')
        assists = self._safe_get(row, 'Ast', '0')
        xg = self._safe_get(row, 'xG', '0')
        xag = self._safe_get(row, 'xAG', '0')
        
        description = (
            f"{player} is a {age}-year-old {position} playing for {team}. "
            f"This season, {player} has appeared in {matches} matches, starting {starts} games "
            f"and playing {minutes} minutes. "
            f"{player} has scored {goals} goals and provided {assists} assists, "
            f"with an expected goals (xG) of {xg} and expected assisted goals (xAG) of {xag}."
        )
        
        return description
    
    def _describe_shooting_stats(self, row: pd.Series) -> str:
        """Convert shooting stats to descriptive text"""
        player = self._safe_get(row, 'Player', row.name if hasattr(row, 'name') else 'Unknown')
        
        goals = self._safe_get(row, 'Gls', '0')
        shots = self._safe_get(row, 'Sh', '0')
        shots_on_target = self._safe_get(row, 'SoT', '0')
        shot_accuracy = self._safe_get(row, 'SoT%', '0%')
        goals_per_shot = self._safe_get(row, 'G/Sh', '0')
        xg = self._safe_get(row, 'xG', '0')
        
        # Determine performance vs xG
        try:
            performance = 'overperformance' if (xg not in ['0', 'N/A'] and float(goals) > float(xg)) else 'alignment with expected output'
        except (ValueError, TypeError):
            performance = 'alignment with expected output'
        
        description = (
            f"{player}'s shooting profile: {goals} goals from {shots} total shots, "
            f"with {shots_on_target} shots on target (accuracy: {shot_accuracy}). "
            f"Shot conversion rate is {goals_per_shot} goals per shot, "
            f"compared to an expected {xg} xG, showing {performance}."
        )
        
        return description
    
    def _describe_passing_stats(self, row: pd.Series) -> str:
        """Convert passing stats to descriptive text"""
        player = self._safe_get(row, 'Player', row.name if hasattr(row, 'name') else 'Unknown')
        
        passes_completed = self._safe_get(row, 'Cmp', '0')
        passes_attempted = self._safe_get(row, 'Att', '0')
        pass_completion = self._safe_get(row, 'Cmp%', '0%')
        
        total_distance = self._safe_get(row, 'TotDist', '0')
        progressive_distance = self._safe_get(row, 'PrgDist', '0')
        
        assists = self._safe_get(row, 'Ast', '0')
        key_passes = self._safe_get(row, 'KP', '0')
        passes_into_final_third = self._safe_get(row, 'Final_Third', '0')
        
        description = (
            f"{player}'s passing statistics: Completed {passes_completed} of {passes_attempted} passes "
            f"({pass_completion} completion rate). Total passing distance of {total_distance} meters, "
            f"with {progressive_distance} meters in progressive passes. "
            f"Created {assists} assists from {key_passes} key passes, "
            f"with {passes_into_final_third} passes into the final third."
        )
        
        return description
    
    def _describe_defense_stats(self, row: pd.Series) -> str:
        """Convert defensive stats to descriptive text"""
        player = self._safe_get(row, 'Player', row.name if hasattr(row, 'name') else 'Unknown')
        
        tackles = self._safe_get(row, 'Tkl', '0')
        tackles_won = self._safe_get(row, 'TklW', '0')
        interceptions = self._safe_get(row, 'Int', '0')
        blocks = self._safe_get(row, 'Blocks', '0')
        clearances = self._safe_get(row, 'Clr', '0')
        
        # Determine defensive intensity
        try:
            intensity = 'strong' if (tackles not in ['N/A', '0'] and int(tackles) > 50) else 'moderate'
        except (ValueError, TypeError):
            intensity = 'moderate'
        
        description = (
            f"{player}'s defensive contributions: {tackles} tackles attempted with {tackles_won} won, "
            f"{interceptions} interceptions, {blocks} blocks, and {clearances} clearances. "
            f"Demonstrates {intensity} defensive activity."
        )
        
        return description
    
    def convert_player_stats(
        self, 
        df: pd.DataFrame, 
        stat_type: str = "standard"
    ) -> List[Dict[str, str]]:
        """
        Convert player stats DataFrame to text descriptions
        
        Args:
            df: DataFrame with player statistics
            stat_type: Type of stats (standard, shooting, passing, defense)
            
        Returns:
            List of dictionaries with player_id and description
        """
        descriptions = []
        
        converter_func = self.player_templates.get(stat_type, self._describe_standard_stats)
        
        for idx, row in df.iterrows():
            try:
                description = converter_func(row)
                
                # Extract player identifier
                player_id = f"{self._safe_get(row, 'Player', str(idx))}_{self._safe_get(row, 'Squad', 'unknown')}"
                
                descriptions.append({
                    "player_id": player_id,
                    "player_name": self._safe_get(row, 'Player', str(idx)),
                    "team": self._safe_get(row, 'Squad', 'Unknown'),
                    "stat_type": stat_type,
                    "description": description,
                    "metadata": {
                        "position": self._safe_get(row, 'Pos', 'Unknown'),
                        "age": self._safe_get(row, 'Age', 'Unknown'),
                        "league": self._safe_get(row, 'Comp', 'Unknown'),
                    }
                })
            except Exception as e:
                logger.warning(f"Error converting row {idx}: {e}")
                continue
        
        logger.info(f"Converted {len(descriptions)} player records to text")
        return descriptions
    
    def convert_team_stats(self, df: pd.DataFrame) -> List[Dict[str, str]]:
        """
        Convert team stats DataFrame to text descriptions
        
        Args:
            df: DataFrame with team statistics
            
        Returns:
            List of dictionaries with team_id and description
        """
        descriptions = []
        
        for idx, row in df.iterrows():
            try:
                team = self._safe_get(row, 'Squad', str(idx))
                
                matches = self._safe_get(row, 'MP', '0')
                wins = self._safe_get(row, 'W', '0')
                draws = self._safe_get(row, 'D', '0')
                losses = self._safe_get(row, 'L', '0')
                
                goals_for = self._safe_get(row, 'GF', '0')
                goals_against = self._safe_get(row, 'GA', '0')
                goal_diff = self._safe_get(row, 'GD', '0')
                
                points = self._safe_get(row, 'Pts', '0')
                
                description = (
                    f"{team}'s season performance: Played {matches} matches with "
                    f"{wins} wins, {draws} draws, and {losses} losses, earning {points} points. "
                    f"Scored {goals_for} goals and conceded {goals_against} goals "
                    f"(goal difference: {goal_diff})."
                )
                
                descriptions.append({
                    "team_id": team,
                    "team_name": team,
                    "stat_type": "team_overview",
                    "description": description,
                    "metadata": {
                        "league": self._safe_get(row, 'Comp', 'Unknown'),
                        "points": points,
                        "position": self._safe_get(row, 'Rk', 'Unknown'),
                    }
                })
            except Exception as e:
                logger.warning(f"Error converting team row {idx}: {e}")
                continue
        
        logger.info(f"Converted {len(descriptions)} team records to text")
        return descriptions
    
    def convert_match_results(self, df: pd.DataFrame) -> List[Dict[str, str]]:
        """
        Convert match results to text descriptions
        
        Args:
            df: DataFrame with match results
            
        Returns:
            List of dictionaries with match_id and description
        """
        descriptions = []
        
        for idx, row in df.iterrows():
            try:
                date = self._safe_get(row, 'Date', 'Unknown date')
                home = self._safe_get(row, 'Home', 'Home team')
                away = self._safe_get(row, 'Away', 'Away team')
                score = self._safe_get(row, 'Score', '0-0')
                
                description = (
                    f"Match on {date}: {home} vs {away}, final score {score}."
                )
                
                descriptions.append({
                    "match_id": f"{home}_{away}_{date}",
                    "description": description,
                    "stat_type": "match_result",
                    "metadata": {
                        "date": date,
                        "home_team": home,
                        "away_team": away,
                        "score": score,
                    }
                })
            except Exception as e:
                logger.warning(f"Error converting match row {idx}: {e}")
                continue
        
        logger.info(f"Converted {len(descriptions)} match records to text")
        return descriptions


def main():
    """Test conversion with sample data"""
    # Create sample data
    sample_data = {
        'Player': ['Mohamed Salah', 'Erling Haaland'],
        'Squad': ['Liverpool', 'Manchester City'],
        'Pos': ['FW', 'FW'],
        'Age': [31, 23],
        'MP': [38, 35],
        'Starts': [38, 35],
        'Min': [3420, 3150],
        'Gls': [30, 36],
        'Ast': [16, 11],
        'xG': [28.5, 35.2],
        'xAG': [14.3, 9.8],
    }
    
    df = pd.DataFrame(sample_data)
    
    converter = TableToTextConverter()
    descriptions = converter.convert_player_stats(df, stat_type="standard")
    
    for desc in descriptions:
        print(f"\n{desc['player_name']}:")
        print(desc['description'])


if __name__ == "__main__":
    main()
