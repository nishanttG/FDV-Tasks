"""
Day 1: Data Collection Pipeline
================================

Reproducible script to collect all data:
  • FBRef player stats (5 seasons, Big 5 leagues)
  • Transfermarkt valuations
  • Tactical blogs (40-80 articles)

Run this to recreate the raw data files.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from script.data_collection.fbref_scraper import scrape_fbref_data
from script.data_collection.transfermarkt_scraper import scrape_transfermarkt_data
from script.data_collection.blog_scraper import scrape_tactical_blogs
from rich.console import Console
from rich.panel import Panel
import time

console = Console()


def main():
    """Run complete Day 1 data collection"""
    
    console.print(Panel.fit(
        "[bold cyan]Day 1: Data Collection Pipeline[/bold cyan]\n\n"
        "This will collect:\n"
        "  • FBRef stats (5 seasons, ~14k players)\n"
        "  • Transfermarkt valuations (~2k players)\n"
        "  • Tactical blogs (40-80 articles)\n\n"
        "Expected time: 30-60 minutes\n"
        "Output: data/raw/ and data/blogs/",
        border_style="cyan"
    ))
    
    input("\nPress ENTER to start data collection (or Ctrl+C to cancel)...")
    
    # Step 1: FBRef player stats
    console.print("\n" + "="*80)
    console.print("[bold green]Step 1/3: FBRef Player Stats[/bold green]")
    console.print("="*80 + "\n")
    
    try:
        console.print("Scraping FBRef (5 seasons, Big 5 leagues)...")
        console.print("⏱️  This may take 20-30 minutes due to rate limiting\n")
        
        fbref_files = scrape_fbref_data(
            seasons=["2021-2022", "2022-2023", "2023-2024", "2024-2025", "2025-2026"],
            leagues=["EPL", "La Liga", "Bundesliga", "Serie A", "Ligue 1"],
            output_dir="data/raw"
        )
        
        console.print(f"✅ FBRef data saved: {len(fbref_files)} files\n", style="bold green")
        
    except Exception as e:
        console.print(f"⚠️  FBRef scraping failed: {e}", style="yellow")
        console.print("Continuing to next step...\n")
    
    time.sleep(2)
    
    # Step 2: Transfermarkt valuations
    console.print("\n" + "="*80)
    console.print("[bold green]Step 2/3: Transfermarkt Valuations[/bold green]")
    console.print("="*80 + "\n")
    
    try:
        console.print("Scraping Transfermarkt valuations...")
        console.print("⏱️  This may take 10-15 minutes\n")
        
        tm_files = scrape_transfermarkt_data(
            leagues=["premier-league", "laliga", "bundesliga", "serie-a", "ligue-1"],
            output_dir="data/transfermarkt"
        )
        
        console.print(f"✅ Transfermarkt data saved: {len(tm_files)} files\n", style="bold green")
        
    except Exception as e:
        console.print(f"⚠️  Transfermarkt scraping failed: {e}", style="yellow")
        console.print("Continuing to next step...\n")
    
    time.sleep(2)
    
    # Step 3: Tactical blogs
    console.print("\n" + "="*80)
    console.print("[bold green]Step 3/3: Tactical Blogs[/bold green]")
    console.print("="*80 + "\n")
    
    try:
        console.print("Scraping tactical blogs...")
        console.print("Target: 40-80 high-quality articles (1200+ words)\n")
        
        blog_file = scrape_tactical_blogs(
            max_articles=80,
            min_word_count=1200,
            output_dir="data/blogs"
        )
        
        console.print(f"✅ Tactical blogs saved: {blog_file}\n", style="bold green")
        
    except Exception as e:
        console.print(f"⚠️  Blog scraping failed: {e}", style="yellow")
        console.print("You can run script/strategic_blog_scraper.py separately\n")
    
    # Summary
    console.print("\n" + "="*80)
    console.print("[bold cyan]DAY 1 COMPLETE![/bold cyan]")
    console.print("="*80 + "\n")
    
    console.print("✅ Raw data collected and saved to:")
    console.print("   • data/raw/ - FBRef player stats")
    console.print("   • data/transfermarkt/ - Transfer market data")
    console.print("   • data/blogs/ - Tactical articles\n")
    
    console.print("[bold yellow]Next Step:[/bold yellow]")
    console.print("   Run: python script/day2_data_processing.py\n")


if __name__ == "__main__":
    main()
