"""
Unit tests for tactical_sources.py
Tests source list structure, validation, and configuration
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from script.data_collection.tactical_sources import (
    TIER_1_PRIORITY,
    NEWS_OUTLETS,
    COMMUNITY_SOURCES,
    TACTICAL_ARTICLES,
    RSS_FEEDS,
    ALL_SOURCES,
    OPTIMAL_ARTICLE_COUNT,
    OPTIMAL_WORD_COUNT,
    OPTIMAL_TOKEN_COUNT,
    TACTICAL_KEYWORDS,
    TARGET_CLUBS,
    UEFA_COMPETITIONS
)


class TestTacticalSourcesStructure:
    """Test the structure and validity of tactical sources"""
    
    def test_tier1_sources_exist(self):
        """Test that Tier 1 priority sources are defined"""
        assert len(TIER_1_PRIORITY) > 0, "Tier 1 sources should not be empty"
        assert len(TIER_1_PRIORITY) >= 10, "Should have at least 10 Tier 1 sources"
    
    def test_tier1_sources_are_urls(self):
        """Test that all Tier 1 sources are valid URLs"""
        for source in TIER_1_PRIORITY:
            assert source.startswith('http'), f"Source {source} should start with http"
            assert '://' in source, f"Source {source} should be a valid URL"
    
    def test_tier1_contains_key_sources(self):
        """Test that key tactical sources are included"""
        key_sources = [
            'spielverlagerung.com',
            'totalfootballanalysis.com',
            'statsbomb.com',
            'zonalmarking.net'
        ]
        
        tier1_str = ' '.join(TIER_1_PRIORITY).lower()
        for source in key_sources:
            assert source in tier1_str, f"Key source {source} should be in Tier 1"
    
    def test_news_outlets_exist(self):
        """Test that news outlets are defined"""
        assert len(NEWS_OUTLETS) > 0, "News outlets should not be empty"
        assert len(NEWS_OUTLETS) >= 3, "Should have at least 3 news outlets"
    
    def test_news_outlets_quality(self):
        """Test that quality news sources are included"""
        news_str = ' '.join(NEWS_OUTLETS).lower()
        assert any(x in news_str for x in ['athletic', 'guardian', 'bbc', 'sky']), \
            "Should include major news sources"
    
    def test_no_american_football_sources(self):
        """Test that American football sources were removed"""
        all_sources_str = ' '.join(ALL_SOURCES).lower()
        
        american_keywords = [
            'smartfootball',
            'blitzology',
            'flexbone',
            'coachhoover'
        ]
        
        for keyword in american_keywords:
            assert keyword not in all_sources_str, \
                f"American football source {keyword} should be removed"
    
    def test_european_focus(self):
        """Test that sources focus on European football"""
        # Check that European-related keywords are present
        all_str = ' '.join(ALL_SOURCES).lower()
        
        # Should NOT have American football indicators
        assert 'nfl' not in all_str
        assert 'college football' not in all_str


class TestOptimalParameters:
    """Test optimal article and token parameters"""
    
    def test_optimal_article_count_range(self):
        """Test that optimal article count is within reasonable range"""
        min_articles, max_articles = OPTIMAL_ARTICLE_COUNT
        
        assert min_articles == 40, "Minimum articles should be 40"
        assert max_articles == 80, "Maximum articles should be 80"
        assert min_articles < max_articles, "Min should be less than max"
    
    def test_optimal_word_count_range(self):
        """Test that optimal word count is configured correctly"""
        min_words, max_words = OPTIMAL_WORD_COUNT
        
        assert min_words == 1200, "Minimum words should be 1200"
        assert max_words == 3000, "Maximum words should be 3000"
        assert min_words < max_words, "Min should be less than max"
    
    def test_optimal_token_count_range(self):
        """Test that token budget is reasonable for RAG"""
        min_tokens, max_tokens = OPTIMAL_TOKEN_COUNT
        
        assert min_tokens == 300_000, "Min tokens should be 300k"
        assert max_tokens == 600_000, "Max tokens should be 600k"
        assert min_tokens < max_tokens, "Min should be less than max"
    
    def test_token_calculation_matches_articles(self):
        """Test that token budget aligns with article count"""
        min_articles, max_articles = OPTIMAL_ARTICLE_COUNT
        min_words, max_words = OPTIMAL_WORD_COUNT
        min_tokens, max_tokens = OPTIMAL_TOKEN_COUNT
        
        # Rough calculation: articles * avg_words * 1.3 tokens/word
        avg_words = (min_words + max_words) / 2
        expected_min_tokens = min_articles * avg_words * 1.3
        expected_max_tokens = max_articles * avg_words * 1.3
        
        # Allow 20% tolerance
        assert abs(min_tokens - expected_min_tokens) / expected_min_tokens < 0.2, \
            "Min tokens should roughly match article calculation"
        assert abs(max_tokens - expected_max_tokens) / expected_max_tokens < 0.2, \
            "Max tokens should roughly match article calculation"


class TestTacticalKeywords:
    """Test tactical keyword configuration"""
    
    def test_tactical_keywords_exist(self):
        """Test that tactical keywords are defined"""
        assert len(TACTICAL_KEYWORDS) > 0, "Should have tactical keywords"
        assert len(TACTICAL_KEYWORDS) >= 10, "Should have at least 10 keywords"
    
    def test_essential_keywords_present(self):
        """Test that essential tactical keywords are included"""
        essential_keywords = [
            'pressing',
            'formation',
            'tactical',
            'defense',
            'attack'
        ]
        
        keywords_lower = [k.lower() for k in TACTICAL_KEYWORDS]
        for keyword in essential_keywords:
            assert any(keyword in k for k in keywords_lower), \
                f"Essential keyword '{keyword}' should be present"
    
    def test_uefa_keywords_included(self):
        """Test that UEFA competition keywords are included"""
        keywords_str = ' '.join(TACTICAL_KEYWORDS).lower()
        
        assert any(x in keywords_str for x in ['champions league', 'europa', 'uefa']), \
            "Should include UEFA competition keywords"


class TestTargetClubs:
    """Test target clubs configuration"""
    
    def test_target_clubs_exist(self):
        """Test that target clubs are defined"""
        assert len(TARGET_CLUBS) > 0, "Should have target clubs"
        assert len(TARGET_CLUBS) >= 20, "Should have at least 20 clubs"
    
    def test_top5_leagues_represented(self):
        """Test that all top 5 leagues have clubs"""
        clubs_str = ' '.join(TARGET_CLUBS).lower()
        
        # Premier League
        assert any(x in clubs_str for x in ['manchester', 'liverpool', 'arsenal']), \
            "Should include Premier League clubs"
        
        # La Liga
        assert any(x in clubs_str for x in ['barcelona', 'real madrid', 'atletico']), \
            "Should include La Liga clubs"
        
        # Bundesliga
        assert any(x in clubs_str for x in ['bayern', 'dortmund', 'leipzig']), \
            "Should include Bundesliga clubs"
        
        # Serie A
        assert any(x in clubs_str for x in ['inter', 'milan', 'juventus']), \
            "Should include Serie A clubs"
        
        # Ligue 1
        assert any(x in clubs_str for x in ['psg', 'monaco', 'marseille']), \
            "Should include Ligue 1 clubs"
    
    def test_no_american_clubs(self):
        """Test that no American football clubs are included"""
        clubs_str = ' '.join(TARGET_CLUBS).lower()
        
        american_teams = ['patriots', 'cowboys', 'packers', 'chiefs']
        for team in american_teams:
            assert team not in clubs_str, \
                f"American team {team} should not be in target clubs"


class TestUEFACompetitions:
    """Test UEFA competitions configuration"""
    
    def test_uefa_competitions_exist(self):
        """Test that UEFA competitions are defined"""
        assert len(UEFA_COMPETITIONS) > 0, "Should have UEFA competitions"
    
    def test_all_uefa_competitions_present(self):
        """Test that all major UEFA competitions are included"""
        competitions_str = ' '.join(UEFA_COMPETITIONS).lower()
        
        assert 'champions' in competitions_str, "Should include Champions League"
        assert 'europa' in competitions_str, "Should include Europa League"
        assert 'conference' in competitions_str, "Should include Conference League"
    
    def test_uefa_competition_count(self):
        """Test that we have the expected number of UEFA competitions"""
        # Should have UCL, UEL, UECL and possibly abbreviations
        assert len(UEFA_COMPETITIONS) >= 3, "Should have at least 3 UEFA competitions"


class TestAllSources:
    """Test combined sources list"""
    
    def test_all_sources_combines_tiers(self):
        """Test that ALL_SOURCES includes all tiers"""
        assert len(ALL_SOURCES) > 0, "ALL_SOURCES should not be empty"
        
        # Should be combination of Tier 1 + News + Community
        expected_min = len(TIER_1_PRIORITY) + len(NEWS_OUTLETS)
        assert len(ALL_SOURCES) >= expected_min, \
            f"ALL_SOURCES should have at least {expected_min} sources"
    
    def test_all_sources_no_duplicates(self):
        """Test that ALL_SOURCES has no duplicates"""
        assert len(ALL_SOURCES) == len(set(ALL_SOURCES)), \
            "ALL_SOURCES should not contain duplicates"
    
    def test_all_sources_are_urls(self):
        """Test that all sources are valid URLs"""
        for source in ALL_SOURCES:
            assert isinstance(source, str), f"Source {source} should be a string"
            assert source.startswith('http'), f"Source {source} should start with http"


class TestRSSFeeds:
    """Test RSS feed configuration"""
    
    def test_rss_feeds_exist(self):
        """Test that RSS feeds are defined"""
        assert len(RSS_FEEDS) > 0, "Should have RSS feeds"
    
    def test_rss_feeds_are_urls(self):
        """Test that RSS feeds are valid URLs"""
        for feed in RSS_FEEDS:
            assert feed.startswith('http'), f"RSS feed {feed} should be a URL"
            assert 'feed' in feed.lower() or 'rss' in feed.lower(), \
                f"RSS feed {feed} should contain 'feed' or 'rss'"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
