"""
FBref Big 5 Leagues Data Collection Guide

The soccerdata library supports FBref's "Big 5" European leagues:
https://fbref.com/en/comps/Big5/2024-2025/2024-2025-Big-5-European-Leagues-Stats

League Codes:
- EPL: English Premier League
- La_Liga: Spanish La Liga
- Bundesliga: German Bundesliga
- Serie_A: Italian Serie A
- Ligue_1: French Ligue 1

Available Stats Categories:
1. Standard Stats (Goals, Assists, xG, xAG, etc.)
2. Shooting Stats (Shots, SoT, Shot accuracy, etc.)
3. Passing Stats (Passes, Key passes, Assists, etc.)
4. Defense Stats (Tackles, Interceptions, Blocks, etc.)
5. Possession Stats (Touches, Dribbles, Carries, etc.)
6. Playing Time Stats (Minutes, Starts, Substitutions, etc.)
7. Miscellaneous Stats (Cards, Fouls, etc.)
8. Goalkeeper Stats (Saves, Clean sheets, etc.)

Current Implementation:
- Fetches: Standard, Shooting, Passing, Defense
- Season: 2024-2025
- Leagues: All Top 5
- Output: CSV files in data/raw/

Data Structure Example:
Player Stats CSV columns include:
- Player, Squad, Pos, Age, MP, Starts, Min
- Gls, Ast, xG, xAG (for standard)
- Sh, SoT, SoT%, G/Sh (for shooting)
- Cmp, Att, Cmp%, TotDist, PrgDist (for passing)
- Tkl, TklW, Int, Blocks, Clr (for defense)

Typical Data Volume:
- ~100 teams across 5 leagues
- ~25-30 players per team
- = ~2,500-3,000 player records
- × 3-4 stat types = ~10,000 total stats rows
- After text conversion: ~2,000-3,000 searchable documents

Benefits of Big 5 Leagues:
✓ Comprehensive coverage of top European football
✓ High-quality data from FBref
✓ Consistent format across leagues
✓ Updated regularly during season
✓ Includes both attacking and defensive metrics
"""

# Example usage in code:
if __name__ == "__main__":
    from script.data_collection.fbref_scraper import FBrefScraper
    
    # Scrape all Big 5 leagues
    scraper = FBrefScraper(
        leagues=["EPL", "La_Liga", "Bundesliga", "Serie_A", "Ligue_1"],
        season="2024-2025"
    )
    
    # Fetch all stats
    stats = scraper.fetch_all_stats()
    
    # Create summary
    scraper.create_summary_json(stats)
    
    print(f"Collected stats from {len(scraper.leagues)} leagues")
