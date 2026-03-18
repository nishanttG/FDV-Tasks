"""
Unit tests for blog_scraper.py
Tests article validation, quality checks, and scraping logic
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from script.data_collection.blog_scraper import BlogScraper


class TestBlogScraperInitialization:
    """Test BlogScraper initialization"""
    
    def test_initialization_default(self):
        """Test default initialization"""
        scraper = BlogScraper()
        assert scraper.data_dir is not None
        assert scraper.config is not None
    
    def test_initialization_custom_dir(self):
        """Test initialization with custom directory"""
        custom_dir = "./test_blogs"
        scraper = BlogScraper(data_dir=custom_dir)
        assert str(scraper.data_dir).endswith('test_blogs')


class TestDomainExtraction:
    """Test domain extraction from URLs"""
    
    def test_extract_domain_simple(self):
        """Test domain extraction from simple URL"""
        scraper = BlogScraper()
        url = "https://spielverlagerung.com/article/tactics"
        domain = scraper.extract_domain(url)
        assert domain == "spielverlagerung.com"
    
    def test_extract_domain_with_www(self):
        """Test domain extraction with www"""
        scraper = BlogScraper()
        url = "https://www.zonalmarking.net/article"
        domain = scraper.extract_domain(url)
        assert domain == "zonalmarking.net", "Should remove www prefix"
    
    def test_extract_domain_with_path(self):
        """Test domain extraction with complex path"""
        scraper = BlogScraper()
        url = "https://statsbomb.com/articles/soccer/tactical-analysis"
        domain = scraper.extract_domain(url)
        assert domain == "statsbomb.com"
    
    def test_extract_domain_invalid_url(self):
        """Test domain extraction with invalid URL"""
        scraper = BlogScraper()
        url = "not-a-valid-url"
        domain = scraper.extract_domain(url)
        assert domain == "unknown", "Should return 'unknown' for invalid URLs"


class TestArticleValidation:
    """Test article validation logic"""
    
    def test_valid_article_1200_words(self):
        """Test that 1200+ word articles pass validation"""
        scraper = BlogScraper()
        
        article = {
            'text': 'tactical ' * 1200,  # 1200 words
            'word_count': 1200,
            'title': 'Tactical Analysis',
            'url': 'https://example.com/article'
        }
        
        assert scraper.is_valid_article(article) is True
    
    def test_invalid_article_below_1200_words(self):
        """Test that articles below 1200 words fail validation"""
        scraper = BlogScraper()
        
        article = {
            'text': 'tactical ' * 1199,  # 1199 words
            'word_count': 1199,
            'title': 'Short Article',
            'url': 'https://example.com/article'
        }
        
        assert scraper.is_valid_article(article) is False
    
    def test_valid_article_with_tactical_keywords(self):
        """Test that articles with tactical keywords pass"""
        scraper = BlogScraper()
        
        article = {
            'text': 'This is a comprehensive tactical analysis of pressing systems and formations. ' * 150,
            'word_count': 1500,
            'title': 'Pressing Analysis',
            'url': 'https://example.com/article'
        }
        
        assert scraper.is_valid_article(article) is True
    
    def test_invalid_article_no_tactical_content(self):
        """Test that articles without tactical keywords fail"""
        scraper = BlogScraper()
        
        article = {
            'text': 'This is about cooking recipes and travel destinations. ' * 200,
            'word_count': 1300,
            'title': 'Non-Football Article',
            'url': 'https://example.com/article'
        }
        
        assert scraper.is_valid_article(article) is False
    
    def test_invalid_article_empty_text(self):
        """Test that articles with empty text fail"""
        scraper = BlogScraper()
        
        article = {
            'text': '',
            'word_count': 0,
            'title': 'Empty',
            'url': 'https://example.com/article'
        }
        
        assert scraper.is_valid_article(article) is False
    
    def test_invalid_article_none(self):
        """Test that None articles fail validation"""
        scraper = BlogScraper()
        assert scraper.is_valid_article(None) is False
    
    def test_valid_article_uefa_keywords(self):
        """Test that articles with UEFA keywords pass"""
        scraper = BlogScraper()
        
        article = {
            'text': 'Analysis of Champions League tactics and Europa League strategies. ' * 180,
            'word_count': 1400,
            'title': 'UEFA Analysis',
            'url': 'https://example.com/article'
        }
        
        assert scraper.is_valid_article(article) is True


class TestWordCountValidation:
    """Test word count requirements"""
    
    def test_minimum_word_count_1200(self):
        """Test that minimum word count is 1200"""
        scraper = BlogScraper()
        
        # Exactly 1200 words
        article_1200 = {
            'text': 'tactical ' * 1200,
            'word_count': 1200,
            'title': 'Analysis',
            'url': 'https://example.com'
        }
        assert scraper.is_valid_article(article_1200) is True
        
        # 1199 words - should fail
        article_1199 = {
            'text': 'tactical ' * 1199,
            'word_count': 1199,
            'title': 'Analysis',
            'url': 'https://example.com'
        }
        assert scraper.is_valid_article(article_1199) is False
    
    def test_optimal_word_range(self):
        """Test that articles in optimal range (1200-3000) pass"""
        scraper = BlogScraper()
        
        # Test various word counts in optimal range
        for word_count in [1200, 1500, 2000, 2500, 3000]:
            article = {
                'text': 'tactical analysis ' * word_count,
                'word_count': word_count,
                'title': f'Article {word_count}',
                'url': 'https://example.com'
            }
            assert scraper.is_valid_article(article) is True, \
                f"Article with {word_count} words should be valid"
    
    def test_above_optimal_still_valid(self):
        """Test that articles above 3000 words still pass (no upper limit in validation)"""
        scraper = BlogScraper()
        
        article = {
            'text': 'tactical analysis ' * 5000,
            'word_count': 5000,
            'title': 'Long Article',
            'url': 'https://example.com'
        }
        # Should still be valid, just above optimal range
        assert scraper.is_valid_article(article) is True


class TestTacticalKeywordDetection:
    """Test tactical keyword detection in articles"""
    
    @pytest.mark.parametrize("keyword", [
        'tactical',
        'formation',
        'pressing',
        'defense',
        'attack',
        'possession',
        'counter',
        'transition',
        'strategy',
        'analysis'
    ])
    def test_individual_tactical_keywords(self, keyword):
        """Test that each tactical keyword is recognized"""
        scraper = BlogScraper()
        
        article = {
            'text': f'This article discusses {keyword} in football. ' * 200,
            'word_count': 1300,
            'title': f'{keyword.title()} Analysis',
            'url': 'https://example.com'
        }
        
        assert scraper.is_valid_article(article) is True, \
            f"Article with keyword '{keyword}' should be valid"
    
    def test_uefa_keywords(self):
        """Test UEFA-specific keywords"""
        scraper = BlogScraper()
        
        for keyword in ['champions league', 'europa league', 'uefa']:
            article = {
                'text': f'Analysis of {keyword} matches and tactics. ' * 180,
                'word_count': 1400,
                'title': f'{keyword} Analysis',
                'url': 'https://example.com'
            }
            assert scraper.is_valid_article(article) is True


class TestQualityMetrics:
    """Test quality metrics and filtering"""
    
    def test_quality_filtering_integration(self):
        """Test that quality filtering works with multiple criteria"""
        scraper = BlogScraper()
        
        # Good article: 1200+ words + tactical content
        good_article = {
            'text': 'Comprehensive tactical analysis of pressing systems and defensive formations. ' * 150,
            'word_count': 1500,
            'title': 'Tactical Analysis',
            'url': 'https://spielverlagerung.com/article',
            'source': 'spielverlagerung.com'
        }
        
        # Bad article: too short
        bad_short = {
            'text': 'Brief tactical note. ' * 50,
            'word_count': 150,
            'title': 'Short Note',
            'url': 'https://example.com'
        }
        
        # Bad article: no tactical content
        bad_content = {
            'text': 'This is about travel and cooking. ' * 200,
            'word_count': 1300,
            'title': 'Travel Guide',
            'url': 'https://example.com'
        }
        
        assert scraper.is_valid_article(good_article) is True
        assert scraper.is_valid_article(bad_short) is False
        assert scraper.is_valid_article(bad_content) is False


class TestSourceMetadata:
    """Test source metadata handling"""
    
    def test_source_extraction_from_url(self):
        """Test that source domain is correctly extracted"""
        scraper = BlogScraper()
        
        test_cases = [
            ('https://spielverlagerung.com/article', 'spielverlagerung.com'),
            ('https://www.zonalmarking.net/post', 'zonalmarking.net'),
            ('https://statsbomb.com/articles/soccer/analysis', 'statsbomb.com'),
            ('https://totalfootballanalysis.com/article/123', 'totalfootballanalysis.com')
        ]
        
        for url, expected_domain in test_cases:
            domain = scraper.extract_domain(url)
            assert domain == expected_domain, \
                f"URL {url} should extract to {expected_domain}"


class TestScrapingConfiguration:
    """Test scraping configuration and parameters"""
    
    def test_tactical_sources_defined(self):
        """Test that TACTICAL_SOURCES is properly defined"""
        scraper = BlogScraper()
        assert hasattr(scraper, 'TACTICAL_SOURCES') or 'TACTICAL_SOURCES' in dir(BlogScraper)
    
    def test_target_article_count(self):
        """Test that scraper targets 40-80 articles"""
        # This is more of a configuration test
        # In actual implementation, we should target 40-80 articles
        # This test documents the requirement
        TARGET_MIN = 40
        TARGET_MAX = 80
        
        assert TARGET_MIN == 40
        assert TARGET_MAX == 80
        assert TARGET_MIN < TARGET_MAX


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_article_with_missing_fields(self):
        """Test articles with missing fields"""
        scraper = BlogScraper()
        
        # Missing word_count
        article_no_count = {
            'text': 'tactical analysis ' * 1300,
            'title': 'Article',
            'url': 'https://example.com'
        }
        # Should handle gracefully (check word_count with get())
        result = scraper.is_valid_article(article_no_count)
        assert isinstance(result, bool)
        
        # Missing text
        article_no_text = {
            'word_count': 1500,
            'title': 'Article',
            'url': 'https://example.com'
        }
        assert scraper.is_valid_article(article_no_text) is False
    
    def test_case_insensitive_keyword_matching(self):
        """Test that keyword matching is case-insensitive"""
        scraper = BlogScraper()
        
        # Test uppercase
        article_upper = {
            'text': 'TACTICAL ANALYSIS OF PRESSING ' * 200,
            'word_count': 1300,
            'title': 'Analysis',
            'url': 'https://example.com'
        }
        assert scraper.is_valid_article(article_upper) is True
        
        # Test mixed case
        article_mixed = {
            'text': 'TaCtiCaL aNaLySiS of FoRmAtIoN ' * 200,
            'word_count': 1300,
            'title': 'Analysis',
            'url': 'https://example.com'
        }
        assert scraper.is_valid_article(article_mixed) is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
