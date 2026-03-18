"""
Transfermarkt Data Scraper
Collects market values, transfer history, and player profiles
Uses soccerdata library for reliable Transfermarkt access
"""

from soccerdata.transfermarkt import Transfermarkt
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Optional
from loguru import logger
import os
from dotenv import load_dotenv

load_dotenv()


class TransfermarktScraper:
    """Scrapes player market values and transfer data from Transfermarkt"""
    
    def __init__(
        self, 
        leagues: Optional[List[str]] = None,
        seasons: Optional[List[str]] = None,
        data_dir: str = "./data/raw/transfermarkt"
    ):
        """
        Initialize Transfermarkt scraper
        
        Args:
            leagues: List of league codes (EPL, La_Liga, Bundesliga, Serie_A, Ligue_1)
            seasons: List of season strings (e.g., ['2024-2025', '2023-2024'])
            data_dir: Directory to save raw data
        """
        self.leagues = leagues or os.getenv("TOP_LEAGUES", "EPL,La_Liga,Bundesliga,Serie_A,Ligue_1").split(",")
        self.seasons = seasons or [
            "2025-2026", "2024-2025", "2023-2024", "2022-2023", "2021-2022"
        ]
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized Transfermarkt scraper for leagues: {self.leagues}")
    
    def fetch_market_values(self) -> pd.DataFrame:
        """
        Fetch current market values for players in specified leagues
        
        Returns:
            DataFrame with player market values
        """
        all_dfs = []
        
        for league in self.leagues:
            try:
                logger.info(f"Fetching market values for {league}...")
                
                # Initialize Transfermarkt reader
                tm = Transfermarkt(leagues=league, seasons=self.seasons[0])  # Use current season
                
                # Fetch player market values
                df = tm.read_player_values()
                
                if not df.empty:
                    df['league'] = league
                    all_dfs.append(df)
                    logger.info(f"Retrieved {len(df)} player valuations from {league}")
                
            except Exception as e:
                logger.error(f"Error fetching market values for {league}: {e}")
                continue
        
        # Combine all leagues
        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            
            # Save to CSV
            output_path = self.data_dir / f"market_values_{self.seasons[0].replace('-', '_')}.csv"
            combined_df.to_csv(output_path)
            
            logger.info(f"Combined {len(combined_df)} total market values")
            return combined_df
        else:
            return pd.DataFrame()
    
    def fetch_transfer_history(self, season: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch transfer history for a specific season
        
        Args:
            season: Season string (e.g., '2024-2025'). Defaults to current season.
            
        Returns:
            DataFrame with transfer records
        """
        season = season or self.seasons[0]
        all_dfs = []
        
        for league in self.leagues:
            try:
                logger.info(f"Fetching transfers for {league}, season {season}...")
                
                # Initialize Transfermarkt reader
                tm = Transfermarkt(leagues=league, seasons=season)
                
                # Fetch transfers
                df = tm.read_transfers()
                
                if not df.empty:
                    df['league'] = league
                    df['season'] = season
                    all_dfs.append(df)
                    logger.info(f"Retrieved {len(df)} transfers from {league}")
                
            except Exception as e:
                logger.error(f"Error fetching transfers for {league}: {e}")
                continue
        
        # Combine all leagues
        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            
            # Save to CSV
            output_path = self.data_dir / f"transfers_{season.replace('-', '_')}.csv"
            combined_df.to_csv(output_path)
            
            logger.info(f"Combined {len(combined_df)} total transfers")
            return combined_df
        else:
            return pd.DataFrame()
    
    def fetch_squad_values(self) -> pd.DataFrame:
        """
        Fetch total squad market values for teams
        
        Returns:
            DataFrame with team squad values
        """
        all_dfs = []
        
        for league in self.leagues:
            try:
                logger.info(f"Fetching squad values for {league}...")
                
                # Initialize Transfermarkt reader
                tm = Transfermarkt(leagues=league, seasons=self.seasons[0])
                
                # Fetch team values
                df = tm.read_team_values()
                
                if not df.empty:
                    df['league'] = league
                    all_dfs.append(df)
                    logger.info(f"Retrieved squad values for {len(df)} teams in {league}")
                
            except Exception as e:
                logger.error(f"Error fetching squad values for {league}: {e}")
                continue
        
        # Combine all leagues
        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            
            # Save to CSV
            output_path = self.data_dir / f"squad_values_{self.seasons[0].replace('-', '_')}.csv"
            combined_df.to_csv(output_path)
            
            logger.info(f"Combined squad values for {len(combined_df)} teams")
            return combined_df
        else:
            return pd.DataFrame()
    
    def create_summary_json(self, market_values_df: pd.DataFrame, transfers_df: pd.DataFrame) -> List[Dict]:
        """
        Convert Transfermarkt data into JSON summaries for RAG ingestion
        
        Args:
            market_values_df: DataFrame with player market values
            transfers_df: DataFrame with transfer records
            
        Returns:
            List of dictionaries with player/transfer summaries
        """
        summaries = []
        
        # Player market value summaries
        if not market_values_df.empty:
            for _, row in market_values_df.iterrows():
                try:
                    summary = {
                        "type": "player_market_value",
                        "player": row.get('player', 'Unknown'),
                        "team": row.get('team', 'Unknown'),
                        "league": row.get('league', 'Unknown'),
                        "market_value": row.get('market_value', 0),
                        "currency": row.get('currency', 'EUR'),
                        "age": row.get('age', None),
                        "position": row.get('position', 'Unknown'),
                        "nationality": row.get('nationality', 'Unknown'),
                        "text": f"{row.get('player', 'Unknown')} ({row.get('team', 'Unknown')}) - Market Value: €{row.get('market_value', 0)}M"
                    }
                    summaries.append(summary)
                except Exception as e:
                    logger.debug(f"Error creating market value summary: {e}")
                    continue
        
        # Transfer summaries
        if not transfers_df.empty:
            for _, row in transfers_df.iterrows():
                try:
                    summary = {
                        "type": "transfer",
                        "player": row.get('player', 'Unknown'),
                        "from_team": row.get('from', 'Unknown'),
                        "to_team": row.get('to', 'Unknown'),
                        "fee": row.get('fee', 0),
                        "season": row.get('season', 'Unknown'),
                        "transfer_type": row.get('type', 'Unknown'),
                        "text": f"Transfer: {row.get('player', 'Unknown')} from {row.get('from', 'Unknown')} to {row.get('to', 'Unknown')} for €{row.get('fee', 0)}M ({row.get('season', 'Unknown')})"
                    }
                    summaries.append(summary)
                except Exception as e:
                    logger.debug(f"Error creating transfer summary: {e}")
                    continue
        
        # Save to JSON
        if summaries:
            output_path = self.data_dir / f"transfermarkt_summaries.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(summaries, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Created {len(summaries)} Transfermarkt summaries")
        
        return summaries
    
    def scrape_all(self) -> Dict[str, pd.DataFrame]:
        """
        Scrape all Transfermarkt data types
        
        Returns:
            Dictionary with all scraped data
        """
        logger.info("=" * 70)
        logger.info("STARTING TRANSFERMARKT DATA COLLECTION")
        logger.info("=" * 70)
        
        results = {}
        
        # Market values
        logger.info("\n[1/3] Fetching market values...")
        results['market_values'] = self.fetch_market_values()
        
        # Transfers (current season)
        logger.info("\n[2/3] Fetching transfer history...")
        results['transfers'] = self.fetch_transfer_history()
        
        # Squad values
        logger.info("\n[3/3] Fetching squad values...")
        results['squad_values'] = self.fetch_squad_values()
        
        # Create JSON summaries
        if not results['market_values'].empty or not results['transfers'].empty:
            logger.info("\nCreating JSON summaries for RAG...")
            self.create_summary_json(results['market_values'], results['transfers'])
        
        logger.info("\n" + "=" * 70)
        logger.info("TRANSFERMARKT SCRAPING COMPLETE!")
        logger.info("=" * 70)
        logger.info(f"✓ Market values: {len(results['market_values'])} players")
        logger.info(f"✓ Transfers: {len(results['transfers'])} records")
        logger.info(f"✓ Squad values: {len(results['squad_values'])} teams")
        
        return results


def main():
    """Main execution function"""
    scraper = TransfermarktScraper()
    
    try:
        # Scrape all data
        results = scraper.scrape_all()
        
        logger.success("Transfermarkt data collection completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during Transfermarkt scraping: {e}")
        raise


if __name__ == "__main__":
    main()
