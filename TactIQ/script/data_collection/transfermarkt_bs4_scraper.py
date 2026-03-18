"""
Transfermarkt BS4 Scraper
Direct web scraping using BeautifulSoup for player market values, 
transfer history, and profile data
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import json
from pathlib import Path
from typing import List, Dict, Optional
from loguru import logger
import re
from datetime import datetime


class TransfermarktBS4Scraper:
    """Direct web scraper for Transfermarkt using BeautifulSoup"""
    
    BASE_URL = "https://www.transfermarkt.com"
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    # League URLs on Transfermarkt
    LEAGUE_URLS = {
        'Premier League': '/premier-league/startseite/wettbewerb/GB1',
        'La Liga': '/laliga/startseite/wettbewerb/ES1',
        'Bundesliga': '/bundesliga/startseite/wettbewerb/L1',
        'Serie A': '/serie-a/startseite/wettbewerb/IT1',
        'Ligue 1': '/ligue-1/startseite/wettbewerb/FR1'
    }
    
    def __init__(
        self, 
        data_dir: str = "./data/transfermarkt",
        delay: float = 2.0
    ):
        """
        Initialize BS4 scraper
        
        Args:
            data_dir: Directory to save scraped data
            delay: Delay between requests (seconds) to be respectful
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
        
        logger.info("Initialized Transfermarkt BS4 scraper")
    
    def _make_request(self, url: str, max_retries: int = 3) -> Optional[BeautifulSoup]:
        """
        Make HTTP request with retry logic
        
        Args:
            url: URL to fetch
            max_retries: Maximum number of retries
            
        Returns:
            BeautifulSoup object or None if failed
        """
        full_url = url if url.startswith('http') else self.BASE_URL + url
        
        for attempt in range(max_retries):
            try:
                time.sleep(self.delay)  # Rate limiting
                response = self.session.get(full_url, timeout=30)
                response.raise_for_status()
                
                return BeautifulSoup(response.content, 'html.parser')
                
            except requests.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(self.delay * 2)  # Longer delay on retry
                else:
                    logger.error(f"Failed to fetch {full_url} after {max_retries} attempts")
                    return None
    
    def _clean_value(self, value_str: str) -> float:
        """
        Convert Transfermarkt value string to float (in millions)
        
        Examples:
            '€50.00m' -> 50.0
            '€2.50bn' -> 2500.0
            '€750Th.' -> 0.75
        """
        if not value_str or value_str == '-':
            return 0.0
        
        # Remove currency symbol and spaces
        value_str = value_str.replace('€', '').replace(' ', '').strip()
        
        try:
            # Handle billions
            if 'bn' in value_str.lower():
                return float(value_str.lower().replace('bn', '')) * 1000
            # Handle millions
            elif 'm' in value_str.lower():
                return float(value_str.lower().replace('m', ''))
            # Handle thousands
            elif 'th' in value_str.lower():
                return float(value_str.lower().replace('th.', '').replace('th', '')) / 1000
            # Plain number
            else:
                return float(value_str)
        except ValueError:
            logger.debug(f"Could not parse value: {value_str}")
            return 0.0
    
    def scrape_league_players(self, league_name: str) -> List[Dict]:
        """
        Scrape all players from a league
        
        Args:
            league_name: League name (must be in LEAGUE_URLS)
            
        Returns:
            List of player dictionaries
        """
        if league_name not in self.LEAGUE_URLS:
            logger.error(f"Unknown league: {league_name}")
            return []
        
        logger.info(f"Scraping players from {league_name}...")
        league_url = self.LEAGUE_URLS[league_name]
        
        # Get league page to find clubs
        soup = self._make_request(league_url)
        if not soup:
            return []
        
        players = []
        
        # Find all club links
        club_links = soup.select('table.items tbody tr td.hauptlink a')
        logger.info(f"Found {len(club_links)} clubs in {league_name}")
        
        for idx, club_link in enumerate(club_links, 1):  # Scrape ALL clubs
            club_name = club_link.text.strip()
            club_url = club_link['href']
            
            logger.info(f"  [{idx}/{len(club_links)}] Scraping {club_name}...")
            
            # Get club page
            club_soup = self._make_request(club_url)
            if not club_soup:
                continue
            
            # Find player rows in squad table
            player_rows = club_soup.select('table.items tbody tr')
            
            seen_player_ids = set()  # Track seen player IDs to avoid duplicates
            
            for row in player_rows:
                try:
                    # Skip header rows
                    if 'thead' in row.get('class', []):
                        continue
                    
                    # Skip rows that are detail/spacing rows (they have class 'bg_blau_20' or 'bg_grau')
                    row_classes = row.get('class', [])
                    if 'bg_blau_20' in row_classes or 'bg_grau' in row_classes:
                        continue
                    
                    # Player name and link
                    player_link = row.select_one('td.hauptlink a')
                    if not player_link:
                        continue
                    
                    player_name = player_link.text.strip()
                    player_url = player_link['href']
                    player_id = re.search(r'/spieler/(\d+)', player_url)
                    player_id = player_id.group(1) if player_id else None
                    
                    # Skip if we've already seen this player (handles duplicate rows)
                    if player_id and player_id in seen_player_ids:
                        continue
                    
                    # Position
                    position_td = row.select('td.posrela table tr')[1] if len(row.select('td.posrela table tr')) > 1 else None
                    position = position_td.text.strip() if position_td else 'Unknown'
                    
                    # Age
                    age_td = row.select_one('td.zentriert:nth-of-type(3)')
                    age = age_td.text.strip() if age_td else None
                    
                    # Nationality
                    nationality_img = row.select_one('td.zentriert img.flaggenrahmen')
                    nationality = nationality_img['title'] if nationality_img and nationality_img.get('title') else 'Unknown'
                    
                    # Market value
                    value_td = row.select_one('td.rechts.hauptlink')
                    market_value_str = value_td.text.strip() if value_td else '0'
                    market_value = self._clean_value(market_value_str)
                    
                    # Only add if we have meaningful data (position or market value)
                    if position != 'Unknown' or market_value > 0:
                        player_data = {
                            'player_id': player_id,
                            'player_name': player_name,
                            'club': club_name,
                            'league': league_name,
                            'position': position,
                            'age': age,
                            'nationality': nationality,
                            'market_value_millions': market_value,
                            'market_value_original': market_value_str,
                            'player_url': self.BASE_URL + player_url,
                            'scraped_date': datetime.now().strftime('%Y-%m-%d')
                        }
                        
                        players.append(player_data)
                        if player_id:
                            seen_player_ids.add(player_id)
                    
                except Exception as e:
                    logger.debug(f"Error parsing player row: {e}")
                    continue
            
            club_players_count = sum(1 for p in players if p['club'] == club_name)
            logger.info(f"    Scraped {club_players_count} players from {club_name}")
        
        logger.success(f"Scraped {len(players)} total players from {league_name}")
        return players
    
    def scrape_player_profile(self, player_url: str) -> Dict:
        """
        Scrape detailed player profile
        
        Args:
            player_url: Full URL to player profile
            
        Returns:
            Dictionary with player details
        """
        soup = self._make_request(player_url)
        if not soup:
            return {}
        
        try:
            profile = {}
            
            # Player name
            name_header = soup.select_one('h1.data-header__headline-wrapper')
            profile['name'] = name_header.text.strip() if name_header else 'Unknown'
            
            # Current club
            club_link = soup.select_one('span.data-header__club a')
            profile['current_club'] = club_link.text.strip() if club_link else 'Unknown'
            
            # Market value
            value_div = soup.select_one('div.data-header__market-value-wrapper')
            if value_div:
                value_text = value_div.text.strip()
                profile['current_market_value'] = self._clean_value(value_text)
            
            # Info table
            info_items = soup.select('span.info-table__content')
            for item in info_items:
                label = item.find_previous_sibling('span', class_='info-table__content--bold')
                if label:
                    label_text = label.text.strip().replace(':', '')
                    value_text = item.text.strip()
                    profile[label_text.lower().replace(' ', '_')] = value_text
            
            return profile
            
        except Exception as e:
            logger.error(f"Error scraping player profile: {e}")
            return {}
    
    def scrape_player_transfer_history(self, player_url: str) -> List[Dict]:
        """
        Scrape player's transfer history
        
        Args:
            player_url: Full URL to player profile
            
        Returns:
            List of transfer records
        """
        # Modify URL to get transfer history page
        transfer_url = player_url.replace('/profil/', '/transfers/')
        
        soup = self._make_request(transfer_url)
        if not soup:
            return []
        
        transfers = []
        
        try:
            # Find transfer table
            transfer_rows = soup.select('div.grid.tm-player-transfer-history-grid')
            
            for row in transfer_rows:
                try:
                    # Season
                    season = row.select_one('div.grid__cell--center')
                    season_text = season.text.strip() if season else 'Unknown'
                    
                    # From club
                    from_club = row.select('div.grid__cell.tm-player-transfer-history-grid__old-club img')
                    from_club_name = from_club[0]['alt'] if from_club else 'Unknown'
                    
                    # To club
                    to_club = row.select('div.grid__cell.tm-player-transfer-history-grid__new-club img')
                    to_club_name = to_club[0]['alt'] if to_club else 'Unknown'
                    
                    # Transfer fee
                    fee = row.select_one('div.grid__cell.tm-player-transfer-history-grid__fee')
                    fee_text = fee.text.strip() if fee else '0'
                    fee_value = self._clean_value(fee_text)
                    
                    transfer_data = {
                        'season': season_text,
                        'from_club': from_club_name,
                        'to_club': to_club_name,
                        'transfer_fee_millions': fee_value,
                        'transfer_fee_original': fee_text,
                        'transfer_date': datetime.now().strftime('%Y-%m-%d')
                    }
                    
                    transfers.append(transfer_data)
                    
                except Exception as e:
                    logger.debug(f"Error parsing transfer row: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"Error scraping transfer history: {e}")
        
        return transfers
    
    def scrape_all_leagues(self, leagues: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Scrape player data from multiple leagues
        
        Args:
            leagues: List of league names to scrape (None = all leagues)
            
        Returns:
            DataFrame with all player data
        """
        leagues = leagues or list(self.LEAGUE_URLS.keys())
        all_players = []
        
        logger.info("=" * 70)
        logger.info("STARTING TRANSFERMARKT BS4 SCRAPING")
        logger.info("=" * 70)
        
        for idx, league in enumerate(leagues, 1):
            logger.info(f"\n[{idx}/{len(leagues)}] Scraping {league}...")
            players = self.scrape_league_players(league)
            all_players.extend(players)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_players)
        
        if not df.empty:
            # Save to CSV
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = self.data_dir / 'player_latest_market_value'
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f'player_latest_market_value_bs4_{timestamp}.csv'
            
            df.to_csv(output_path, index=False)
            logger.success(f"Saved {len(df)} player records to {output_path}")
            
            # Save summary JSON for RAG
            self._create_rag_summaries(df)
        
        logger.info("\n" + "=" * 70)
        logger.info("SCRAPING COMPLETE!")
        logger.info("=" * 70)
        logger.info(f"✓ Total players scraped: {len(df)}")
        logger.info(f"✓ Leagues covered: {', '.join(df['league'].unique())}")
        logger.info(f"✓ Clubs covered: {df['club'].nunique()}")
        
        return df
    
    def _create_rag_summaries(self, df: pd.DataFrame):
        """Create JSON summaries for RAG ingestion"""
        summaries = []
        
        for _, row in df.iterrows():
            summary = {
                "type": "player_market_value",
                "player": row['player_name'],
                "club": row['club'],
                "league": row['league'],
                "position": row['position'],
                "age": row['age'],
                "nationality": row['nationality'],
                "market_value": row['market_value_millions'],
                "scraped_date": row['scraped_date'],
                "text": f"{row['player_name']} ({row['club']}, {row['league']}) - {row['position']} - Market Value: €{row['market_value_millions']}M - Age: {row['age']} - {row['nationality']}"
            }
            summaries.append(summary)
        
        # Save JSON
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = self.data_dir / f'player_market_values_rag_{timestamp}.json'
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summaries, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Created {len(summaries)} RAG summaries")


def main():
    """Main execution function"""
    scraper = TransfermarktBS4Scraper()
    
    try:
        # Scrape all leagues (or specify: ['Premier League', 'La Liga'])
        df = scraper.scrape_all_leagues()
        
        logger.success(f"Successfully scraped {len(df)} players!")
        
    except Exception as e:
        logger.error(f"Error during scraping: {e}")
        raise


if __name__ == "__main__":
    main()
