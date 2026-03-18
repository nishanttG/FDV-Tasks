"""
FBref Stats Scraper using soccerdata library
⚠️ IMPORTANT: Only collects from Top 5 European DOMESTIC leagues
Includes: Premier League, La Liga, Bundesliga, Serie A, Ligue 1
⚠️ UEFA competitions (Champions/Europa/Conference League) are NOT available
"""

import soccerdata as sd
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Optional
from loguru import logger
import os
from dotenv import load_dotenv

load_dotenv()


class FBrefScraper:
    """Scrapes football statistics from FBref via soccerdata library"""
    
    def __init__(
        self, 
        leagues: Optional[List[str]] = None,
        seasons: Optional[List[str]] = None,
        include_uefa: bool = False,  # UEFA not available in soccerdata
        data_dir: str = "./data/raw"
    ):
        """
        Initialize FBref scraper
        
        Args:
            leagues: List of league codes (ENG-Premier League, ESP-La Liga, etc.)
            seasons: List of season strings (e.g., ['2025-2026', '2024-2025'])
            include_uefa: NOT SUPPORTED - UEFA club competitions unavailable in soccerdata
            data_dir: Directory to save raw data
        """
        self.leagues = leagues or os.getenv("TOP_LEAGUES", "ENG-Premier League,ESP-La Liga,GER-Bundesliga,ITA-Serie A,FRA-Ligue 1").split(",")
        
        # UEFA club competitions are NOT available in soccerdata's FBref integration
        self.include_uefa = False  # Forced to False - not supported
        self.uefa_competitions = []
        
        # Default: Current season (2025-2026) + previous 4 seasons
        self.seasons = seasons or [
            "2025-2026", "2024-2025", "2023-2024", "2022-2023", "2021-2022"
        ]
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized FBref scraper for leagues: {self.leagues}, seasons: {self.seasons}")
    
    def fetch_player_stats(self, stat_type: str = "standard") -> pd.DataFrame:
        """
        Fetch player statistics from FBref across multiple seasons
        Includes both domestic leagues and UEFA competitions
        
        Args:
            stat_type: Type of stats - 'standard', 'shooting', 'passing', 'defense', etc.
            
        Returns:
            DataFrame with player statistics from domestic leagues and UEFA competitions
        """
        all_dfs = []
        
        # Combine domestic leagues and UEFA competitions
        all_competitions = self.leagues + self.uefa_competitions
        
        for season in self.seasons:
            try:
                logger.info(f"Fetching {stat_type} player stats for {all_competitions}, season {season}")
                
                # Initialize FBref reader for this season
                fbref = sd.FBref(leagues=all_competitions, seasons=season)
                
                # Fetch stats based on type
                if stat_type == "standard":
                    df = fbref.read_player_season_stats(stat_type="standard")
                elif stat_type == "shooting":
                    df = fbref.read_player_season_stats(stat_type="shooting")
                elif stat_type == "passing":
                    df = fbref.read_player_season_stats(stat_type="passing")
                elif stat_type == "defense":
                    df = fbref.read_player_season_stats(stat_type="defense")
                else:
                    df = fbref.read_player_season_stats(stat_type=stat_type)
                
                if not df.empty:
                    df['season'] = season  # Add season column
                    all_dfs.append(df)
                    
                    # Save individual season
                    output_path = self.data_dir / f"player_stats_{stat_type}_{season.replace('-', '_')}.csv"
                    df.to_csv(output_path)
                    logger.info(f"Saved {len(df)} player records for {season}")
                
            except Exception as e:
                logger.error(f"Error fetching player stats for {season}: {e}")
                continue
        
        # Combine all seasons
        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=False)
            logger.info(f"Combined {len(combined_df)} total player records across {len(self.seasons)} seasons")
            logger.info(f"  → Domestic leagues: {', '.join(self.leagues)}")
            if self.uefa_competitions:
                logger.info(f"  → UEFA competitions: {', '.join(self.uefa_competitions)}")
            return combined_df
        else:
            return pd.DataFrame()
    
    def fetch_team_stats(self) -> pd.DataFrame:
        """
        Fetch team-level statistics from FBref (domestic leagues + UEFA)
        
        Returns:
            DataFrame with team statistics
        """
        try:
            all_competitions = self.leagues + self.uefa_competitions
            logger.info(f"Fetching team stats for {all_competitions}")
            
            fbref = sd.FBref(leagues=self.leagues, seasons=self.season)
            df = fbref.read_team_season_stats()
            
            # Save to CSV
            output_path = self.data_dir / f"team_stats_{self.season.replace('-', '_')}.csv"
            df.to_csv(output_path)
            logger.info(f"Saved {len(df)} team records to {output_path}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching team stats: {e}")
            return pd.DataFrame()
    
    def fetch_match_results(self) -> pd.DataFrame:
        """
        Fetch match results and scores
        
        Returns:
            DataFrame with match results
        """
        try:
            logger.info(f"Fetching match results for {self.leagues}")
            
            fbref = sd.FBref(leagues=self.leagues, seasons=self.season)
            df = fbref.read_schedule()
            
            # Save to CSV
            output_path = self.data_dir / f"match_results_{self.season.replace('-', '_')}.csv"
            df.to_csv(output_path)
            logger.info(f"Saved {len(df)} match records to {output_path}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching match results: {e}")
            return pd.DataFrame()
    
    def fetch_all_stats(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch all available statistics
        
        Returns:
            Dictionary mapping stat type to DataFrame
        """
        logger.info("Starting comprehensive stats collection...")
        
        stats = {}
        
        # Fetch different stat categories
        stat_types = ["standard", "shooting", "passing", "defense", "keeper"]
        
        for stat_type in stat_types:
            logger.info(f"Fetching {stat_type} stats...")
            stats[f"player_{stat_type}"] = self.fetch_player_stats(stat_type)
        
        # Team stats
        stats["team_stats"] = self.fetch_team_stats()
        
        # Match results
        stats["match_results"] = self.fetch_match_results()
        
        logger.info(f"Completed fetching {len(stats)} stat categories")
        return stats
    
    def create_summary_json(self, stats: Dict[str, pd.DataFrame]) -> None:
        """
        Create a JSON summary of collected data
        
        Args:
            stats: Dictionary of DataFrames
        """
        summary = {
            "seasons": self.seasons,
            "leagues": self.leagues,
            "collections": {}
        }
        
        for name, df in stats.items():
            if not df.empty:
                summary["collections"][name] = {
                    "records": len(df),
                    "columns": list(df.columns),
                    "sample_teams": df.index.get_level_values(0).unique().tolist()[:5] if hasattr(df.index, 'levels') else []
                }
        
        output_path = self.data_dir / f"collection_summary_{'_'.join([s.replace('-', '_') for s in self.seasons])}.json"
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Created summary at {output_path}")


def main():
    """Main execution function"""
    # Initialize scraper for all Top 5 European leagues + UCL
    # Seasons: 2024-2025 (current) + previous 4 seasons
    scraper = FBrefScraper(
        leagues=["EPL", "La_Liga", "Bundesliga", "Serie_A", "Ligue_1"],
        seasons=["2024-2025", "2023-2024", "2022-2023", "2021-2022", "2020-2021"]
    )
    
    # Fetch all stats
    stats = scraper.fetch_all_stats()
    
    # Create summary
    scraper.create_summary_json(stats)
    
    logger.info("FBref data collection complete!")


if __name__ == "__main__":
    main()
