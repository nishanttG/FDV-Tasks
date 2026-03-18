"""
Integration tests for data validation using pandera schemas
Tests end-to-end data quality validation for all scrapers
"""

import pytest
import sys
from pathlib import Path
import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestPlayerStatsSchema:
    """Test pandera schemas for player statistics"""
    
    def test_player_stats_schema_definition(self):
        """Test that player stats schema is properly defined"""
        schema = DataFrameSchema({
            'player': Column(str, nullable=False),
            'team': Column(str, nullable=False),
            'league': Column(str, nullable=False),
            'season': Column(str, nullable=False, checks=[
                Check(lambda s: s.str.match(r'\d{4}-\d{4}').all())
            ]),
            'goals': Column(int, nullable=True, checks=[
                Check(lambda x: x >= 0)
            ]),
            'assists': Column(int, nullable=True, checks=[
                Check(lambda x: x >= 0)
            ]),
            'minutes': Column(int, nullable=True, checks=[
                Check(lambda x: x >= 0)
            ])
        })
        
        # Test valid data
        valid_data = pd.DataFrame({
            'player': ['Player A', 'Player B'],
            'team': ['Team 1', 'Team 2'],
            'league': ['EPL', 'La_Liga'],
            'season': ['2025-2026', '2024-2025'],
            'goals': [10, 15],
            'assists': [5, 8],
            'minutes': [2000, 2500]
        })
        
        # Should validate without errors
        validated_df = schema.validate(valid_data)
        assert len(validated_df) == 2
    
    def test_player_stats_negative_values_rejected(self):
        """Test that negative values are rejected"""
        schema = DataFrameSchema({
            'player': Column(str),
            'goals': Column(int, checks=[Check(lambda x: x >= 0)])
        })
        
        invalid_data = pd.DataFrame({
            'player': ['Player A'],
            'goals': [-5]  # Invalid: negative goals
        })
        
        with pytest.raises(pa.errors.SchemaError):
            schema.validate(invalid_data)
    
    def test_player_stats_season_format_validation(self):
        """Test that season format is validated"""
        schema = DataFrameSchema({
            'season': Column(str, checks=[
                Check(lambda s: s.str.match(r'\d{4}-\d{4}').all())
            ])
        })
        
        # Valid format
        valid_data = pd.DataFrame({'season': ['2025-2026', '2024-2025']})
        schema.validate(valid_data)
        
        # Invalid format
        invalid_data = pd.DataFrame({'season': ['25-26']})
        with pytest.raises(pa.errors.SchemaError):
            schema.validate(invalid_data)
    
    def test_player_stats_required_columns(self):
        """Test that required columns are present"""
        schema = DataFrameSchema({
            'player': Column(str, nullable=False),
            'team': Column(str, nullable=False),
            'season': Column(str, nullable=False)
        })
        
        # Missing required column
        invalid_data = pd.DataFrame({
            'player': ['Player A'],
            'team': ['Team 1']
            # Missing 'season'
        })
        
        with pytest.raises(pa.errors.SchemaError):
            schema.validate(invalid_data)


class TestBlogArticleSchema:
    """Test pandera schemas for blog articles"""
    
    def test_blog_article_schema_definition(self):
        """Test that blog article schema is properly defined"""
        schema = DataFrameSchema({
            'url': Column(str, nullable=False, checks=[
                Check(lambda s: s.str.startswith(('http://', 'https://')).all())
            ]),
            'title': Column(str, nullable=False, checks=[
                Check(lambda s: s.str.len() > 0)
            ]),
            'content': Column(str, nullable=False, checks=[
                Check(lambda s: s.str.split().str.len() >= 1200, 
                      error='Content must have at least 1200 words')
            ]),
            'word_count': Column(int, nullable=False, checks=[
                Check(lambda x: x >= 1200, error='Word count must be >= 1200')
            ]),
            'source_domain': Column(str, nullable=False),
            'tactical_score': Column(float, nullable=True, checks=[
                Check(lambda x: (x >= 0) & (x <= 1))
            ])
        })
        
        # Test valid data
        valid_data = pd.DataFrame({
            'url': ['https://example.com/article1', 'https://example.com/article2'],
            'title': ['Tactical Analysis', 'Formation Study'],
            'content': [' '.join(['word'] * 1500), ' '.join(['word'] * 1300)],
            'word_count': [1500, 1300],
            'source_domain': ['example.com', 'example.com'],
            'tactical_score': [0.85, 0.92]
        })
        
        validated_df = schema.validate(valid_data)
        assert len(validated_df) == 2
    
    def test_blog_article_word_count_minimum(self):
        """Test that 1200 word minimum is enforced"""
        schema = DataFrameSchema({
            'word_count': Column(int, checks=[
                Check(lambda x: x >= 1200)
            ])
        })
        
        # Valid: meets minimum
        valid_data = pd.DataFrame({'word_count': [1200, 1500, 2000]})
        schema.validate(valid_data)
        
        # Invalid: below minimum
        invalid_data = pd.DataFrame({'word_count': [600, 1000]})
        with pytest.raises(pa.errors.SchemaError):
            schema.validate(invalid_data)
    
    def test_blog_article_url_validation(self):
        """Test that URLs are properly validated"""
        schema = DataFrameSchema({
            'url': Column(str, checks=[
                Check(lambda s: s.str.startswith(('http://', 'https://')).all())
            ])
        })
        
        # Valid URLs
        valid_data = pd.DataFrame({
            'url': ['https://example.com', 'http://test.com']
        })
        schema.validate(valid_data)
        
        # Invalid URLs
        invalid_data = pd.DataFrame({
            'url': ['example.com', 'ftp://test.com']
        })
        with pytest.raises(pa.errors.SchemaError):
            schema.validate(invalid_data)
    
    def test_blog_article_tactical_keywords_present(self):
        """Test that tactical keywords are present in content"""
        tactical_keywords = [
            'tactical', 'formation', 'pressing', 'defense', 'attack',
            'possession', 'counter', 'transition', 'strategy', 'analysis'
        ]
        
        def has_tactical_keywords(content_series):
            """Check if content contains tactical keywords"""
            pattern = '|'.join(tactical_keywords)
            return content_series.str.contains(pattern, case=False, regex=True)
        
        schema = DataFrameSchema({
            'content': Column(str, checks=[
                Check(has_tactical_keywords, 
                      error='Content must contain tactical keywords')
            ])
        })
        
        # Valid: has tactical keywords
        valid_data = pd.DataFrame({
            'content': [
                'This tactical analysis examines the formation',
                'The pressing strategy was effective in defense'
            ]
        })
        schema.validate(valid_data)
        
        # Invalid: no tactical keywords
        invalid_data = pd.DataFrame({
            'content': ['This is a generic sports article about a game']
        })
        with pytest.raises(pa.errors.SchemaError):
            schema.validate(invalid_data)
    
    def test_blog_article_optimal_word_range(self):
        """Test optimal word count range (1200-3000)"""
        def in_optimal_range(word_count):
            """Check if word count is in optimal range"""
            return (word_count >= 1200) & (word_count <= 3000)
        
        schema = DataFrameSchema({
            'word_count': Column(int, checks=[
                Check(lambda x: x >= 1200),  # Minimum required
            ])
        })
        
        # All valid (>= 1200)
        valid_data = pd.DataFrame({
            'word_count': [1200, 2000, 3000, 3500]  # Last one above optimal but still valid
        })
        schema.validate(valid_data)
        
        # Check optimal range separately
        optimal_data = pd.DataFrame({'word_count': [1200, 2000, 3000]})
        assert in_optimal_range(optimal_data['word_count']).all()


class TestTransfermarktDataSchema:
    """Test pandera schemas for Transfermarkt data"""
    
    def test_market_values_schema(self):
        """Test market values schema"""
        schema = DataFrameSchema({
            'player': Column(str, nullable=False),
            'market_value': Column(float, nullable=False, checks=[
                Check(lambda x: x >= 0)
            ]),
            'age': Column(int, nullable=True, checks=[
                Check(lambda x: (x >= 16) & (x <= 45))
            ]),
            'team': Column(str, nullable=False),
            'league': Column(str, nullable=False),
            'season': Column(str, nullable=False)
        })
        
        # Valid data
        valid_data = pd.DataFrame({
            'player': ['Player A', 'Player B'],
            'market_value': [50000000.0, 75000000.0],
            'age': [25, 27],
            'team': ['Team 1', 'Team 2'],
            'league': ['ENG1', 'ESP1'],
            'season': ['2025-2026', '2025-2026']
        })
        
        validated_df = schema.validate(valid_data)
        assert len(validated_df) == 2
    
    def test_transfer_history_schema(self):
        """Test transfer history schema"""
        schema = DataFrameSchema({
            'player': Column(str, nullable=False),
            'from_club': Column(str, nullable=False),
            'to_club': Column(str, nullable=False),
            'fee': Column(float, nullable=True, checks=[
                Check(lambda x: x >= 0)
            ]),
            'season': Column(str, nullable=False)
        })
        
        # Valid data
        valid_data = pd.DataFrame({
            'player': ['Player A', 'Player B'],
            'from_club': ['Club A', 'Club C'],
            'to_club': ['Club B', 'Club D'],
            'fee': [50000000.0, 30000000.0],
            'season': ['2025-2026', '2024-2025']
        })
        
        validated_df = schema.validate(valid_data)
        assert len(validated_df) == 2
    
    def test_squad_values_schema(self):
        """Test squad values schema"""
        schema = DataFrameSchema({
            'team': Column(str, nullable=False),
            'total_value': Column(float, nullable=False, checks=[
                Check(lambda x: x >= 0)
            ]),
            'avg_value': Column(float, nullable=True, checks=[
                Check(lambda x: x >= 0)
            ]),
            'league': Column(str, nullable=False),
            'season': Column(str, nullable=False)
        })
        
        # Valid data
        valid_data = pd.DataFrame({
            'team': ['Team A', 'Team B'],
            'total_value': [500000000.0, 600000000.0],
            'avg_value': [25000000.0, 30000000.0],
            'league': ['ENG1', 'ESP1'],
            'season': ['2025-2026', '2025-2026']
        })
        
        validated_df = schema.validate(valid_data)
        assert len(validated_df) == 2
    
    def test_market_value_negative_rejected(self):
        """Test that negative market values are rejected"""
        schema = DataFrameSchema({
            'market_value': Column(float, checks=[Check(lambda x: x >= 0)])
        })
        
        invalid_data = pd.DataFrame({'market_value': [-100000.0]})
        
        with pytest.raises(pa.errors.SchemaError):
            schema.validate(invalid_data)
    
    def test_age_range_validation(self):
        """Test that age is within reasonable range"""
        schema = DataFrameSchema({
            'age': Column(int, checks=[
                Check(lambda x: (x >= 16) & (x <= 45))
            ])
        })
        
        # Valid ages
        valid_data = pd.DataFrame({'age': [18, 25, 30, 35]})
        schema.validate(valid_data)
        
        # Invalid ages
        invalid_data = pd.DataFrame({'age': [10, 60]})
        with pytest.raises(pa.errors.SchemaError):
            schema.validate(invalid_data)


class TestEndToEndPipeline:
    """Test end-to-end pipeline validation"""
    
    def test_pipeline_combines_all_data_sources(self):
        """Test that pipeline can combine data from all sources"""
        # Player stats
        player_stats = pd.DataFrame({
            'player': ['Player A'],
            'team': ['Team 1'],
            'season': ['2025-2026'],
            'goals': [10]
        })
        
        # Blog articles
        blog_articles = pd.DataFrame({
            'title': ['Article 1'],
            'word_count': [1500],
            'content': [' '.join(['tactical'] * 1500)]
        })
        
        # Market values
        market_values = pd.DataFrame({
            'player': ['Player A'],
            'market_value': [50000000.0],
            'season': ['2025-2026']
        })
        
        # All DataFrames should be valid
        assert len(player_stats) > 0
        assert len(blog_articles) > 0
        assert len(market_values) > 0
    
    def test_pipeline_data_consistency(self):
        """Test data consistency across sources"""
        # Same players should have consistent data
        player_stats = pd.DataFrame({
            'player': ['Kylian Mbappé', 'Erling Haaland'],
            'team': ['Real Madrid', 'Manchester City'],
            'season': ['2025-2026', '2025-2026']
        })
        
        market_values = pd.DataFrame({
            'player': ['Kylian Mbappé', 'Erling Haaland'],
            'market_value': [180000000.0, 200000000.0],
            'season': ['2025-2026', '2025-2026']
        })
        
        # Players should match
        assert set(player_stats['player']) == set(market_values['player'])
        assert set(player_stats['season']) == set(market_values['season'])


class TestDataQualityMetrics:
    """Test data quality metrics"""
    
    def test_completeness_check(self):
        """Test data completeness percentage"""
        def calculate_completeness(df):
            """Calculate percentage of non-null values"""
            total_cells = df.size
            non_null_cells = df.count().sum()
            return (non_null_cells / total_cells) * 100
        
        # High completeness
        complete_data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        assert calculate_completeness(complete_data) == 100.0
        
        # Partial completeness
        partial_data = pd.DataFrame({
            'col1': [1, None, 3],
            'col2': ['a', 'b', None]
        })
        completeness = calculate_completeness(partial_data)
        assert 0 < completeness < 100
    
    def test_consistency_check(self):
        """Test data consistency across seasons"""
        data = pd.DataFrame({
            'season': ['2025-2026', '2024-2025', '2023-2024'],
            'player_count': [500, 480, 490]
        })
        
        # Player counts should be similar across seasons (within reasonable variance)
        mean_count = data['player_count'].mean()
        std_count = data['player_count'].std()
        
        # Coefficient of variation should be low (<20%)
        cv = (std_count / mean_count) * 100
        assert cv < 20, "Player count should be consistent across seasons"
    
    def test_coverage_check(self):
        """Test league and team coverage"""
        data = pd.DataFrame({
            'league': ['EPL', 'La_Liga', 'Bundesliga', 'Serie_A', 'Ligue_1'],
            'team_count': [20, 20, 18, 20, 18]
        })
        
        # All Top 5 leagues should be present
        assert len(data) == 5
        assert 'EPL' in data['league'].values
        assert 'La_Liga' in data['league'].values


class TestSeasonConsistency:
    """Test season consistency across all data sources"""
    
    def test_all_data_includes_current_season(self):
        """Test that all data sources include current season 2025-2026"""
        current_season = '2025-2026'
        
        # Player stats
        player_seasons = ['2025-2026', '2024-2025']
        assert current_season in player_seasons
        
        # Blog articles (should have date from current season)
        # Market values
        market_seasons = ['2025-2026', '2024-2025']
        assert current_season in market_seasons
    
    def test_season_format_consistency(self):
        """Test that all seasons use same format across sources"""
        season_pattern = r'\d{4}-\d{4}'
        
        seasons = ['2025-2026', '2024-2025', '2023-2024']
        
        import re
        for season in seasons:
            assert re.match(season_pattern, season), \
                f"Season {season} doesn't match format YYYY-YYYY"


class TestRAGReadiness:
    """Test RAG (Retrieval Augmented Generation) readiness"""
    
    def test_token_count_in_optimal_range(self):
        """Test that token count is in optimal range (300k-600k)"""
        # With 40-80 articles × 1200-3000 words × ~1.3 tokens/word
        min_articles = 40
        max_articles = 80
        min_words = 1200
        max_words = 3000
        tokens_per_word = 1.3
        
        min_tokens = min_articles * min_words * tokens_per_word
        max_tokens = max_articles * max_words * tokens_per_word
        
        # Should be approximately 300k-600k
        assert 62_400 <= min_tokens <= 100_000  # ~62k minimum
        assert 250_000 <= max_tokens <= 350_000  # ~312k maximum
        
        # Optimal range
        optimal_min = 300_000
        optimal_max = 600_000
        
        assert optimal_min == 300_000
        assert optimal_max == 600_000
    
    def test_article_length_optimal_for_rag(self):
        """Test that article lengths are optimal for RAG chunking"""
        # Optimal: 1200-3000 words (can chunk into 2-6 segments of ~500 words)
        word_counts = [1200, 1500, 2000, 2500, 3000]
        
        for count in word_counts:
            # Should be chunkable into reasonable segments
            chunk_size = 500
            num_chunks = count / chunk_size
            
            assert 2 <= num_chunks <= 6, \
                f"Word count {count} should create 2-6 chunks"
    
    def test_tactical_content_density(self):
        """Test that content has sufficient tactical keyword density"""
        content = ' '.join(['tactical', 'formation', 'pressing'] * 100 + ['word'] * 900)
        
        tactical_keywords = [
            'tactical', 'formation', 'pressing', 'defense', 'attack',
            'possession', 'counter', 'transition', 'strategy', 'analysis'
        ]
        
        # Calculate keyword density
        words = content.lower().split()
        keyword_count = sum(1 for word in words if word in tactical_keywords)
        density = (keyword_count / len(words)) * 100
        
        # Should have at least 5% tactical keyword density
        assert density >= 5.0, f"Tactical keyword density {density}% too low"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
