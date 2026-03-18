"""
Blog Scraper for Tactical Analysis Articles
Uses newspaper3k and BeautifulSoup to ingest tactical blog content
"""

from newspaper import Article, Config
from bs4 import BeautifulSoup
import requests
import json
from pathlib import Path
from typing import List, Dict, Optional
from loguru import logger
from datetime import datetime
import time
from urllib.parse import urlparse


class BlogScraper:
    """Scrapes and processes tactical football blog articles"""
    
    # TIER 1 sources - European/English football focus
    # Target: 40-80 articles, 1,200-3,000 words each
    # Total: ~300k-600k tokens (optimal for RAG reasoning)
    TACTICAL_SOURCES = [
        # Elite tactical analysis
        "https://spielverlagerung.com/",
        "https://totalfootballanalysis.com/",
        "https://statsbomb.com/articles/",
        "https://thefalse9.com/",
        "https://between-the-lines.co.uk/",
        "https://www.zonalmarking.net/",
        "https://coachdriven.com/",
        "https://www.holdingmidfield.com/",
        "https://football-observatory.com/",
        
        # News & media (quality journalism)
        "https://theathletic.com/football/",
        "https://theguardian.com/football/tactics/",
        
        # Medium tactical writers
        "https://medium.com/@thomasrandle",
        "https://medium.com/spielverlagerung",
    ]
    
    def __init__(self, data_dir: str = "./data/blogs"):
        """
        Initialize blog scraper
        
        Args:
            data_dir: Directory to save scraped blog content
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure newspaper3k
        self.config = Config()
        self.config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        self.config.request_timeout = 10
        
        logger.info(f"Initialized blog scraper with output dir: {data_dir}")
    
    def extract_domain(self, url: str) -> str:
        """Extract domain from URL for source tracking"""
        try:
            return urlparse(url).netloc.replace('www.', '')
        except:
            return "unknown"
    
    def is_valid_article(self, article: Dict) -> bool:
        """
        Validate article quality for RAG ingestion
        OPTIMAL: 1,200-3,000 words for best reasoning performance
        
        Args:
            article: Article dictionary
            
        Returns:
            True if article meets quality criteria
        """
        if not article or not article.get('text'):
            return False
        
        word_count = article.get('word_count', 0)
        text_lower = article.get('text', '').lower()
        
        # Quality criteria (IMPORTANT: >600 words minimum)
        has_optimal_length = word_count >= 600  # Lowered to 600 for coverage
        has_tactical_content = any(keyword in text_lower for keyword in [
            'tactical', 'formation', 'pressing', 'defense', 'attack',
            'possession', 'counter', 'transition', 'strategy', 'analysis',
            'champions league', 'europa league', 'uefa'
        ])
        
        return has_optimal_length and has_tactical_content
    
    def scrape_article(self, url: str) -> Optional[Dict]:
        """
        Scrape a single article using newspaper3k
        
        Args:
            url: Article URL
            
        Returns:
            Dictionary with article metadata and content
        """
        try:
            article = Article(url, config=self.config)
            article.download()
            article.parse()
            article.nlp()  # Extract keywords and summary
            
            result = {
                "source": self.extract_domain(url),
                "url": url,
                "title": article.title,
                "authors": article.authors,
                "publish_date": str(article.publish_date) if article.publish_date else None,
                "text": article.text,
                "summary": article.summary,
                "keywords": article.keywords,
                "top_image": article.top_image,
                "scraped_at": datetime.now().isoformat(),
                "word_count": len(article.text.split())
            }
            
            logger.info(f"Successfully scraped: {article.title[:50]}...")
            return result
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return None
    
    def scrape_rss_feed(self, rss_url: str, max_articles: int = 10) -> List[Dict]:
        """
        Scrape articles from an RSS feed
        
        Args:
            rss_url: RSS feed URL
            max_articles: Maximum number of articles to scrape
            
        Returns:
            List of article dictionaries
        """
        try:
            import feedparser
            
            feed = feedparser.parse(rss_url)
            articles = []
            
            for entry in feed.entries[:max_articles]:
                article_data = self.scrape_article(entry.link)
                if article_data:
                    articles.append(article_data)
                time.sleep(1)  # Rate limiting
            
            logger.info(f"Scraped {len(articles)} articles from RSS feed")
            return articles
            
        except ImportError:
            logger.warning("feedparser not installed. Install with: pip install feedparser")
            return []
        except Exception as e:
            logger.error(f"Error parsing RSS feed {rss_url}: {e}")
            return []
    
    def scrape_site_articles(self, base_url: str, max_articles: int = 20) -> List[Dict]:
        """
        Scrape articles from a website's article listing page
        
        Args:
            base_url: Base URL of the site or article listing page
            max_articles: Maximum number of articles to scrape
            
        Returns:
            List of article dictionaries
        """
        try:
            response = requests.get(base_url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find article links (common patterns)
            article_links = []
            
            # Try multiple common selectors
            selectors = [
                'article a[href]',
                'a.article-link',
                'h2 a[href]',
                'h3 a[href]',
                '.post-title a[href]',
                '.entry-title a[href]'
            ]
            
            for selector in selectors:
                links = soup.select(selector)
                if links:
                    article_links.extend([link.get('href') for link in links])
                    break
            
            # Make URLs absolute
            from urllib.parse import urljoin
            article_links = [urljoin(base_url, link) for link in article_links]
            
            # Remove duplicates
            article_links = list(set(article_links))[:max_articles]
            
            logger.info(f"Found {len(article_links)} article links from {base_url}")
            
            # Scrape each article
            articles = []
            for link in article_links:
                article_data = self.scrape_article(link)
                if article_data:
                    articles.append(article_data)
                time.sleep(1)  # Rate limiting
            
            return articles
            
        except Exception as e:
            logger.error(f"Error scraping site {base_url}: {e}")
            return []
    
    def scrape_manual_urls(self, urls: List[str]) -> List[Dict]:
        """
        Scrape a manual list of article URLs with quality filtering
        
        Args:
            urls: List of article URLs
            
        Returns:
            List of validated article dictionaries
        """
        articles = []
        skipped = 0
        
        for url in urls:
            article_data = self.scrape_article(url)
            if article_data:
                if self.is_valid_article(article_data):
                    articles.append(article_data)
                    logger.info(f"✓ Valid article: {article_data['title'][:50]}... ({article_data['word_count']} words)")
                else:
                    skipped += 1
                    logger.debug(f"✗ Skipped low-quality article: {article_data.get('title', 'Unknown')[:50]}...")
            time.sleep(1)  # Rate limiting
        
        logger.info(f"Scraped {len(articles)}/{len(urls)} valid articles (skipped {skipped} low-quality)")
        return articles
    
    def save_articles(self, articles: List[Dict], filename: str = None) -> None:
        """
        Save scraped articles to JSON file
        
        Args:
            articles: List of article dictionaries
            filename: Output filename (defaults to timestamp)
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tactical_articles_{timestamp}.json"
        
        output_path = self.data_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(articles, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(articles)} articles to {output_path}")
    
    def load_articles(self, filename: str) -> List[Dict]:
        """
        Load previously scraped articles
        
        Args:
            filename: JSON file to load
            
        Returns:
            List of article dictionaries
        """
        file_path = self.data_dir / filename
        
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        
        logger.info(f"Loaded {len(articles)} articles from {file_path}")
        return articles
    
    def scrape_curated_sources(self, max_per_source: int = 10) -> List[Dict]:
        """
        Scrape from curated list of tactical analysis sources
        
        Args:
            max_per_source: Maximum articles per source
            
        Returns:
            List of all scraped articles
        """
        all_articles = []
        
        # Import comprehensive source list
        try:
            from script.data_collection.tactical_sources import SAFE_SOURCES, FEEDSPOT_TOP_SOURCES
            
            # Combine and deduplicate
            sources_to_scrape = list(set(SAFE_SOURCES[:10]))  # Top 10 safe sources
            
        except ImportError:
            # Fallback to default sources
            sources_to_scrape = [
                "https://spielverlagerung.com/",
                "https://totalfootballanalysis.com/",
                "https://themastermindsite.com/",
                "https://thecoachesvoice.com/",
                "https://footballblog.co.uk/category/football-tactics/",
            ]
        
        logger.info(f"Scraping from {len(sources_to_scrape)} tactical sources...")
        
        for source in sources_to_scrape:
            try:
                logger.info(f"Scraping from {source}")
                articles = self.scrape_site_articles(source, max_articles=max_per_source)
                
                # Filter valid articles
                valid_articles = [a for a in articles if self.is_valid_article(a)]
                all_articles.extend(valid_articles)
                
                logger.info(f"  → Collected {len(valid_articles)} valid articles from {source}")
                
                time.sleep(2)  # Respectful rate limiting between sources
                
            except Exception as e:
                logger.warning(f"  → Error scraping {source}: {e}")
                continue
        
        logger.info(f"Total: {len(all_articles)} valid tactical articles collected")
        return all_articles
    
    def scrape_multiple_sources_batch(
        self, 
        source_list: List[str], 
        articles_per_source: int = 5,
        max_total: int = 60
    ) -> List[Dict]:
        """
        Batch scrape from multiple sources with limits
        
        Args:
            source_list: List of source URLs
            articles_per_source: Max articles per source
            max_total: Maximum total articles to collect
            
        Returns:
            List of validated articles
        """
        all_articles = []
        sources_processed = 0
        
        logger.info(f"Batch scraping from up to {len(source_list)} sources (target: {max_total} articles)...")
        
        for source in source_list:
            if len(all_articles) >= max_total:
                logger.info(f"Reached target of {max_total} articles. Stopping.")
                break
            
            try:
                articles = self.scrape_site_articles(source, max_articles=articles_per_source)
                valid_articles = [a for a in articles if self.is_valid_article(a)]
                
                if valid_articles:
                    all_articles.extend(valid_articles)
                    sources_processed += 1
                    logger.info(f"✓ {source}: {len(valid_articles)} articles ({len(all_articles)}/{max_total} total)")
                
                time.sleep(2)  # Rate limiting
                
            except Exception as e:
                logger.debug(f"✗ {source}: {e}")
                continue
        
        logger.info(f"Batch complete: {len(all_articles)} articles from {sources_processed} sources")
        return all_articles


def main():
    """Main execution function - European football tactical blog scraping (40-80 articles)"""
    scraper = BlogScraper()
    
    # Use European-only source list from tactical_sources
    try:
        from script.data_collection.tactical_sources import (
            TIER_1_PRIORITY,
            NEWS_OUTLETS,
            COMMUNITY_SOURCES,
            TACTICAL_ARTICLES,
            RSS_FEEDS,
            ALL_SOURCES,
            OPTIMAL_ARTICLE_COUNT
        )
        
        logger.info("=" * 70)
        logger.info("EUROPEAN FOOTBALL TACTICAL BLOG SCRAPING")
        logger.info("=" * 70)
        logger.info(f"Sources: {len(TIER_1_PRIORITY)} Tier 1 + {len(NEWS_OUTLETS)} news outlets")
        logger.info(f"Target: 40-80 articles (1,200-3,000 words each)")
        logger.info(f"Focus: EPL, La Liga, Bundesliga, Serie A, Ligue 1 + UEFA")
        logger.info("=" * 70)
        
        articles = []
        min_articles, max_articles = OPTIMAL_ARTICLE_COUNT
        
        # Strategy 1: Tier 1 priority sources (elite tactical analysis)
        logger.info(f"\n[1/3] Scraping TIER 1 sources (target {min_articles})...")
        batch1 = scraper.scrape_multiple_sources_batch(
            TIER_1_PRIORITY,
            articles_per_source=4,
            max_total=min_articles
        )
        articles.extend(batch1)
        
        # Strategy 2: Add curated tactical articles if needed
        if len(articles) < min_articles and TACTICAL_ARTICLES:
            logger.info(f"\n[2/3] Adding curated articles (need {min_articles - len(articles)} more)...")
            manual_articles = scraper.scrape_manual_urls(TACTICAL_ARTICLES[:15])
            articles.extend(manual_articles)
        
        # Strategy 3: News outlets + community sources (up to 80 total)
        if len(articles) < max_articles:
            logger.info(f"\n[3/3] Adding news & community sources (target {max_articles})...")
            remaining = max_articles - len(articles)
            extra_sources = NEWS_OUTLETS + COMMUNITY_SOURCES[:5]
            batch2 = scraper.scrape_multiple_sources_batch(
                extra_sources,
                articles_per_source=3,
                max_total=remaining
            )
            articles.extend(batch2)
        
    except ImportError as e:
        logger.warning(f"Could not import tactical_sources: {e}")
        logger.info("Using fallback scraping strategy...")
        articles = scraper.scrape_curated_sources(max_per_source=5)
    
    # Final filtering (1200+ words required)
    valid_articles = [a for a in articles if scraper.is_valid_article(a)]
    
    # Remove duplicates by URL
    seen_urls = set()
    unique_articles = []
    for article in valid_articles:
        url = article.get('url', '')
        if url not in seen_urls:
            seen_urls.add(url)
            unique_articles.append(article)
    
    # Save results
    if unique_articles:
        scraper.save_articles(unique_articles)
        
        logger.info("\n" + "=" * 70)
        logger.info("SCRAPING COMPLETE!")
        logger.info("=" * 70)
        logger.info(f"✓ Total articles: {len(unique_articles)} (Optimal: 40-80)")
        logger.info(f"✓ Average words: {sum(a['word_count'] for a in unique_articles) / len(unique_articles):.0f} (Target: 1,200-3,000)")
        logger.info(f"✓ Unique sources: {len(set(a['source'] for a in unique_articles))}")
        logger.info(f"✓ Focus: European football + UEFA competitions")
        logger.info(f"✓ Estimated tokens: ~{sum(a['word_count'] for a in unique_articles) * 1.3 / 1000:.0f}k")
        
        # Source breakdown
        source_counts = {}
        for article in unique_articles:
            source = article['source']
            source_counts[source] = source_counts.get(source, 0) + 1
        
        logger.info("\nArticles per source:")
        for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            logger.info(f"  {source}: {count} articles")
        
        logger.info("=" * 70)
    else:
        logger.warning("No valid articles scraped. Check source accessibility.")


if __name__ == "__main__":
    main()
