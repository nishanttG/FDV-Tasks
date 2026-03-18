"""
Test script for Transfermarkt BS4 scraper
Quick test to verify the scraper works
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_collection.transfermarkt_bs4_scraper import TransfermarktBS4Scraper
from loguru import logger


def test_single_league():
    """Test scraping a single league"""
    logger.info("Testing BS4 scraper with Premier League...")
    
    scraper = TransfermarktBS4Scraper(
        data_dir="./data/transfermarkt",
        delay=2.0  # Be respectful with rate limiting
    )
    
    # Test with just Premier League
    df = scraper.scrape_all_leagues(leagues=['Premier League'])
    
    if not df.empty:
        logger.success(f"✓ Successfully scraped {len(df)} players")
        logger.info(f"✓ Columns: {', '.join(df.columns)}")
        logger.info("\nSample data:")
        print(df.head(10))
        
        # Show top 10 most valuable players
        top_players = df.nlargest(10, 'market_value_millions')
        logger.info("\nTop 10 most valuable players:")
        for idx, row in top_players.iterrows():
            logger.info(f"  {row['player_name']} ({row['club']}) - €{row['market_value_millions']}M")
    else:
        logger.warning("No data scraped")


def test_player_profile():
    """Test scraping a specific player profile"""
    logger.info("\nTesting player profile scraping...")
    
    scraper = TransfermarktBS4Scraper()
    
    # Test with a known player URL (example: Erling Haaland)
    # You can replace this with any player URL
    player_url = "https://www.transfermarkt.com/erling-haaland/profil/spieler/418560"
    
    profile = scraper.scrape_player_profile(player_url)
    
    if profile:
        logger.success("✓ Profile scraped successfully")
        logger.info(f"Player: {profile.get('name', 'Unknown')}")
        logger.info(f"Club: {profile.get('current_club', 'Unknown')}")
        logger.info(f"Market Value: €{profile.get('current_market_value', 0)}M")


def test_transfer_history():
    """Test scraping transfer history"""
    logger.info("\nTesting transfer history scraping...")
    
    scraper = TransfermarktBS4Scraper()
    
    # Test with a known player URL
    player_url = "https://www.transfermarkt.com/erling-haaland/profil/spieler/418560"
    
    transfers = scraper.scrape_player_transfer_history(player_url)
    
    if transfers:
        logger.success(f"✓ Found {len(transfers)} transfers")
        for transfer in transfers:
            logger.info(f"  {transfer['season']}: {transfer['from_club']} → {transfer['to_club']} (€{transfer['transfer_fee_millions']}M)")
    else:
        logger.warning("No transfers found")


if __name__ == "__main__":
    # Run all tests
    try:
        # Test 1: Scrape league data
        test_single_league()
        
        # Test 2: Scrape player profile
        # test_player_profile()
        
        # Test 3: Scrape transfer history
        # test_transfer_history()
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
