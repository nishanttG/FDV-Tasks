"""
Strategic Blog Scraping Script for TactIQ
Target: 60-80 high-quality tactical blogs across 5 leagues
Balanced coverage: ~12-16 per league
"""

import sys
from pathlib import Path
import json
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from script.data_collection.blog_scraper import BlogScraper
from loguru import logger

# Configure logger
logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")

# LEAGUE-SPECIFIC TACTICAL SOURCES
# Target: Balanced coverage across 5 leagues

STRATEGIC_SOURCES = {
    "premier_league": [
        # EPL-focused tactical sources
        "https://statsbomb.com/articles/soccer/arsenal-tactical-analysis/",
        "https://statsbomb.com/articles/soccer/manchester-city-tactical-analysis/",
        "https://statsbomb.com/articles/soccer/liverpool-tactical-analysis/",
        "https://totalfootballanalysis.com/premier-league",
        "https://thefalse9.com/category/premier-league/",
        "https://spielverlagerung.com/tag/premier-league/",
        "https://between-the-lines.co.uk/category/premier-league/",
        "https://www.theguardian.com/football/premierleague/tactics",
    ],
    "la_liga": [
        # La Liga tactical sources
        "https://statsbomb.com/articles/soccer/real-madrid-tactical-analysis/",
        "https://statsbomb.com/articles/soccer/barcelona-tactical-analysis/",
        "https://totalfootballanalysis.com/la-liga",
        "https://thefalse9.com/category/la-liga/",
        "https://spielverlagerung.com/tag/la-liga/",
        "https://between-the-lines.co.uk/category/la-liga/",
    ],
    "bundesliga": [
        # Bundesliga tactical sources (Spielverlagerung strength)
        "https://spielverlagerung.com/tag/bundesliga/",
        "https://totalfootballanalysis.com/bundesliga",
        "https://thefalse9.com/category/bundesliga/",
        "https://statsbomb.com/articles/soccer/bayern-munich-tactical-analysis/",
        "https://statsbomb.com/articles/soccer/borussia-dortmund-tactical-analysis/",
        "https://between-the-lines.co.uk/category/bundesliga/",
    ],
    "serie_a": [
        # Serie A tactical sources
        "https://statsbomb.com/articles/soccer/inter-milan-tactical-analysis/",
        "https://statsbomb.com/articles/soccer/ac-milan-tactical-analysis/",
        "https://statsbomb.com/articles/soccer/napoli-tactical-analysis/",
        "https://totalfootballanalysis.com/serie-a",
        "https://thefalse9.com/category/serie-a/",
        "https://spielverlagerung.com/tag/serie-a/",
    ],
    "ligue_1": [
        # Ligue 1 tactical sources (often underrepresented)
        "https://statsbomb.com/articles/soccer/psg-tactical-analysis/",
        "https://totalfootballanalysis.com/ligue-1",
        "https://thefalse9.com/category/ligue-1/",
        "https://spielverlagerung.com/tag/ligue-1/",
    ],
}

# HIGH-VALUE MANUAL URLS (Pre-vetted tactical articles)
CURATED_URLS = {
    "pressing_analysis": [
        "https://spielverlagerung.com/2024/05/15/pressing-traps-and-cover-shadows/",
        "https://totalfootballanalysis.com/article/tactical-theory-pressing",
        "https://statsbomb.com/articles/soccer/the-evolution-of-the-high-press/",
    ],
    "build_up_play": [
        "https://spielverlagerung.com/2024/03/20/building-from-the-back/",
        "https://totalfootballanalysis.com/article/tactical-theory-build-up-play",
        "https://statsbomb.com/articles/soccer/progressive-passing-patterns/",
    ],
    "transition_play": [
        "https://spielverlagerung.com/2024/02/10/counter-pressing-principles/",
        "https://totalfootballanalysis.com/article/tactical-theory-transitions",
    ],
    "formations": [
        "https://spielverlagerung.com/2024/01/05/4-3-3-formation-guide/",
        "https://spielverlagerung.com/2023/12/15/3-4-3-tactical-analysis/",
        "https://totalfootballanalysis.com/article/formation-guide-4-2-3-1",
    ],
}


def load_existing_blogs() -> dict:
    """Load existing blog data to avoid duplicates"""
    blog_dir = PROJECT_ROOT / "data" / "blogs"
    existing_files = list(blog_dir.glob("tactical_blogs_*.json"))
    
    if not existing_files:
        return {"articles": [], "urls": set()}
    
    # Load most recent file
    latest_file = max(existing_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"Loading existing blogs from: {latest_file.name}")
    
    with open(latest_file, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    
    urls = {art.get('url') for art in articles if art.get('url')}
    
    logger.info(f"Found {len(articles)} existing articles with {len(urls)} unique URLs")
    return {"articles": articles, "urls": urls}


def analyze_coverage(articles: list) -> dict:
    """Analyze league coverage in current articles"""
    league_keywords = {
        "Premier League": ["premier league", "epl", "manchester", "arsenal", "liverpool", "chelsea", "tottenham"],
        "La Liga": ["la liga", "real madrid", "barcelona", "atletico", "sevilla"],
        "Bundesliga": ["bundesliga", "bayern", "dortmund", "leipzig", "leverkusen"],
        "Serie A": ["serie a", "inter", "milan", "juventus", "napoli", "roma"],
        "Ligue 1": ["ligue 1", "psg", "paris", "marseille", "lyon", "monaco"],
    }
    
    coverage = {league: 0 for league in league_keywords}
    
    for article in articles:
        text_lower = article.get('text', '').lower() + article.get('title', '').lower()
        for league, keywords in league_keywords.items():
            if any(kw in text_lower for kw in keywords):
                coverage[league] += 1
    
    return coverage


def scrape_strategically(target_total: int = 70, target_per_league: int = 14):
    """
    Strategic scraping to reach target with balanced coverage
    
    Args:
        target_total: Total number of articles to aim for (default: 70)
        target_per_league: Target articles per league (default: 14)
    """
    logger.info("="*80)
    logger.info("🎯 STRATEGIC BLOG SCRAPING FOR 5-LEAGUE RAG SYSTEM")
    logger.info("="*80)
    logger.info(f"Target Total: {target_total} articles")
    logger.info(f"Target Per League: {target_per_league} articles")
    
    # Load existing data
    existing_data = load_existing_blogs()
    all_articles = existing_data["articles"]
    scraped_urls = existing_data["urls"]
    
    # Analyze current coverage
    logger.info("\n📊 Current Coverage Analysis:")
    coverage = analyze_coverage(all_articles)
    for league, count in coverage.items():
        status = "✅" if count >= target_per_league else "⚠️" if count >= 8 else "❌"
        logger.info(f"   {status} {league}: {count} articles")
    
    current_total = len(all_articles)
    needed = target_total - current_total
    
    logger.info(f"\n📈 Progress: {current_total}/{target_total} articles ({current_total/target_total*100:.1f}%)")
    
    if needed <= 0:
        logger.info("✅ Target already reached! No additional scraping needed.")
        return all_articles
    
    logger.info(f"🔍 Need to scrape ~{needed} more articles")
    
    # Initialize scraper
    scraper = BlogScraper()
    
    # Scrape strategically
    new_articles = []
    
    # Priority 1: Fill gaps in underrepresented leagues
    logger.info("\n🎯 Phase 1: Filling league gaps...")
    priority_leagues = sorted(coverage.items(), key=lambda x: x[1])  # Weakest first
    
    for league, count in priority_leagues:
        gap = target_per_league - count
        if gap > 0 and len(new_articles) < needed:
            logger.info(f"\n🔍 Scraping for {league} (need {gap} more)...")
            league_key = league.lower().replace(" ", "_")
            
            if league_key in STRATEGIC_SOURCES:
                for source_url in STRATEGIC_SOURCES[league_key]:
                    if len(new_articles) >= needed:
                        break
                    
                    try:
                        logger.info(f"   Checking: {source_url}")
                        articles = scraper.scrape_site_articles(source_url, max_articles=5)
                        
                        for art in articles:
                            if art['url'] not in scraped_urls:
                                new_articles.append(art)
                                scraped_urls.add(art['url'])
                                logger.info(f"   ✓ Added: {art['title'][:60]}...")
                    
                    except Exception as e:
                        logger.warning(f"   ✗ Failed to scrape {source_url}: {e}")
    
    # Priority 2: Scrape curated high-value URLs
    logger.info("\n🎯 Phase 2: Scraping curated tactical articles...")
    curated_flat = []
    for category, urls in CURATED_URLS.items():
        curated_flat.extend([u for u in urls if u not in scraped_urls])
    
    if curated_flat and len(new_articles) < needed:
        logger.info(f"   Found {len(curated_flat)} curated URLs...")
        articles = scraper.scrape_manual_urls(curated_flat[:needed - len(new_articles)])
        
        for art in articles:
            if art['url'] not in scraped_urls:
                new_articles.append(art)
                scraped_urls.add(art['url'])
    
    # Combine old + new
    final_articles = all_articles + new_articles
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = PROJECT_ROOT / "data" / "blogs" / f"tactical_blogs_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_articles, f, indent=2, ensure_ascii=False)
    
    # Final report
    logger.info("\n" + "="*80)
    logger.info("✅ SCRAPING COMPLETE!")
    logger.info("="*80)
    logger.info(f"📊 Results:")
    logger.info(f"   Previous: {current_total} articles")
    logger.info(f"   New: {len(new_articles)} articles")
    logger.info(f"   Total: {len(final_articles)} articles")
    logger.info(f"   Progress: {len(final_articles)}/{target_total} ({len(final_articles)/target_total*100:.1f}%)")
    
    logger.info(f"\n📈 Updated Coverage:")
    final_coverage = analyze_coverage(final_articles)
    for league, count in final_coverage.items():
        status = "✅" if count >= target_per_league else "⚠️" if count >= 8 else "❌"
        logger.info(f"   {status} {league}: {count} articles")
    
    logger.info(f"\n💾 Saved to: {output_file}")
    
    return final_articles


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Strategic blog scraping for 5-league RAG")
    parser.add_argument("--target", type=int, default=70, help="Target total articles (default: 70)")
    parser.add_argument("--per-league", type=int, default=14, help="Target per league (default: 14)")
    
    args = parser.parse_args()
    
    try:
        scrape_strategically(target_total=args.target, target_per_league=args.per_league)
        logger.info("\n✨ Strategic scraping completed successfully!")
    except Exception as e:
        logger.error(f"\n❌ Scraping failed: {e}")
        import traceback
        traceback.print_exc()
