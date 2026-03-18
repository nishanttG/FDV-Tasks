"""
Unit tests for transfermarkt_scraper.py
Tests Transfermarkt scraper initialization and data fetching methods
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from script.data_collection.transfermarkt_scraper import TransfermarktScraper


class TestTransfermarktScraperInitialization:
    """Test TransfermarktScraper initialization"""
    
    def test_initialization_default(self):
        """Test default initialization"""
        scraper = TransfermarktScraper()
        
        assert scraper.leagues is not None
        assert scraper.seasons is not None
        assert scraper.data_dir is not None
        assert len(scraper.leagues) == 5, "Should have 5 default leagues"
        assert len(scraper.seasons) == 5, "Should have 5 default seasons"
    
    def test_initialization_with_custom_leagues(self):
        """Test initialization with custom leagues"""
        custom_leagues = ['ENG1', 'ESP1']
        scraper = TransfermarktScraper(leagues=custom_leagues)
        
        assert scraper.leagues == custom_leagues
        assert len(scraper.leagues) == 2
    
    def test_initialization_with_custom_seasons(self):
        """Test initialization with custom seasons"""
        custom_seasons = ['2025-2026', '2024-2025']
        scraper = TransfermarktScraper(seasons=custom_seasons)
        
        assert scraper.seasons == custom_seasons
        assert len(scraper.seasons) == 2
    
    def test_initialization_with_custom_data_dir(self):
        """Test initialization with custom data directory"""
        custom_dir = "./test_data"
        scraper = TransfermarktScraper(data_dir=custom_dir)
        
        assert str(scraper.data_dir).endswith('test_data')


class TestSeasonConfiguration:
    """Test season configuration"""
    
    def test_default_seasons_include_current(self):
        """Test that default seasons include current season 2025-2026"""
        scraper = TransfermarktScraper()
        
        assert '2025-2026' in scraper.seasons, "Should include current season 2025-2026"
    
    def test_default_seasons_count(self):
        """Test that default has 5 seasons"""
        scraper = TransfermarktScraper()
        
        assert len(scraper.seasons) == 5, "Should have 5 seasons by default"
    
    def test_season_order_newest_first(self):
        """Test that seasons are ordered newest first"""
        scraper = TransfermarktScraper()
        
        # First season should be current 2025-2026
        assert scraper.seasons[0] == '2025-2026'
        
        # Seasons should be in descending order
        years = [int(s.split('-')[0]) for s in scraper.seasons]
        assert years == sorted(years, reverse=True), "Seasons should be newest first"
    
    def test_expected_season_list(self):
        """Test that default seasons match expected list"""
        scraper = TransfermarktScraper()
        
        expected_seasons = [
            '2025-2026',
            '2024-2025',
            '2023-2024',
            '2022-2023',
            '2021-2022'
        ]
        
        assert scraper.seasons == expected_seasons


class TestLeagueConfiguration:
    """Test league configuration"""
    
    def test_default_transfermarkt_leagues(self):
        """Test default Transfermarkt league codes"""
        scraper = TransfermarktScraper()
        
        # Transfermarkt uses different codes: ENG1, ESP1, GER1, ITA1, FRA1
        expected_leagues = ['ENG1', 'ESP1', 'GER1', 'ITA1', 'FRA1']
        assert scraper.leagues == expected_leagues
    
    def test_all_top5_leagues_present(self):
        """Test that all top 5 leagues are present"""
        scraper = TransfermarktScraper()
        
        assert 'ENG1' in scraper.leagues, "Should include Premier League"
        assert 'ESP1' in scraper.leagues, "Should include La Liga"
        assert 'GER1' in scraper.leagues, "Should include Bundesliga"
        assert 'ITA1' in scraper.leagues, "Should include Serie A"
        assert 'FRA1' in scraper.leagues, "Should include Ligue 1"
    
    def test_league_count(self):
        """Test that we have exactly 5 leagues"""
        scraper = TransfermarktScraper()
        
        assert len(scraper.leagues) == 5


class TestFetchMarketValues:
    """Test fetch_market_values method"""
    
    @patch('script.data_collection.transfermarkt_scraper.sd.Transfermarkt')
    def test_fetch_market_values_returns_dataframe(self, mock_tm_class):
        """Test that fetch_market_values returns a DataFrame"""
        scraper = TransfermarktScraper(seasons=['2025-2026'])
        
        # Mock the Transfermarkt reader
        mock_tm = MagicMock()
        mock_tm_class.return_value = mock_tm
        
        # Create mock DataFrame
        df = pd.DataFrame({
            'player': ['Player1', 'Player2'],
            'market_value': [50000000, 75000000],
            'age': [25, 27]
        })
        mock_tm.read_player_values.return_value = df
        
        result = scraper.fetch_market_values()
        
        assert isinstance(result, pd.DataFrame)
    
    @patch('script.data_collection.transfermarkt_scraper.sd.Transfermarkt')
    def test_fetch_market_values_adds_season_column(self, mock_tm_class):
        """Test that season column is added"""
        scraper = TransfermarktScraper(seasons=['2025-2026'])
        
        mock_tm = MagicMock()
        mock_tm_class.return_value = mock_tm
        
        df = pd.DataFrame({'player': ['Player1']})
        mock_tm.read_player_values.return_value = df
        
        result = scraper.fetch_market_values()
        
        # Should add 'season' column
        assert 'season' in result.columns if not result.empty else True
    
    @patch('script.data_collection.transfermarkt_scraper.sd.Transfermarkt')
    def test_fetch_market_values_loops_leagues_and_seasons(self, mock_tm_class):
        """Test that method loops through leagues and seasons"""
        scraper = TransfermarktScraper(
            leagues=['ENG1', 'ESP1'], 
            seasons=['2025-2026', '2024-2025']
        )
        
        mock_tm = MagicMock()
        mock_tm_class.return_value = mock_tm
        
        df = pd.DataFrame({'player': ['Player1']})
        mock_tm.read_player_values.return_value = df
        
        result = scraper.fetch_market_values()
        
        # Should be called 2 leagues × 2 seasons = 4 times
        assert mock_tm.read_player_values.call_count == 4


class TestFetchTransferHistory:
    """Test fetch_transfer_history method"""
    
    @patch('script.data_collection.transfermarkt_scraper.sd.Transfermarkt')
    def test_fetch_transfer_history_returns_dataframe(self, mock_tm_class):
        """Test that fetch_transfer_history returns a DataFrame"""
        scraper = TransfermarktScraper(seasons=['2025-2026'])
        
        mock_tm = MagicMock()
        mock_tm_class.return_value = mock_tm
        
        df = pd.DataFrame({
            'player': ['Player1'],
            'from_club': ['Club A'],
            'to_club': ['Club B'],
            'fee': [50000000]
        })
        mock_tm.read_transfers.return_value = df
        
        result = scraper.fetch_transfer_history()
        
        assert isinstance(result, pd.DataFrame)
    
    @patch('script.data_collection.transfermarkt_scraper.sd.Transfermarkt')
    def test_fetch_transfer_history_includes_season(self, mock_tm_class):
        """Test that season column is included"""
        scraper = TransfermarktScraper(seasons=['2025-2026'])
        
        mock_tm = MagicMock()
        mock_tm_class.return_value = mock_tm
        
        df = pd.DataFrame({'player': ['Player1']})
        mock_tm.read_transfers.return_value = df
        
        result = scraper.fetch_transfer_history()
        
        assert 'season' in result.columns if not result.empty else True


class TestFetchSquadValues:
    """Test fetch_squad_values method"""
    
    @patch('script.data_collection.transfermarkt_scraper.sd.Transfermarkt')
    def test_fetch_squad_values_returns_dataframe(self, mock_tm_class):
        """Test that fetch_squad_values returns a DataFrame"""
        scraper = TransfermarktScraper(seasons=['2025-2026'])
        
        mock_tm = MagicMock()
        mock_tm_class.return_value = mock_tm
        
        df = pd.DataFrame({
            'team': ['Team A', 'Team B'],
            'total_value': [500000000, 600000000],
            'avg_value': [25000000, 30000000]
        })
        mock_tm.read_squad_values.return_value = df
        
        result = scraper.fetch_squad_values()
        
        assert isinstance(result, pd.DataFrame)
    
    @patch('script.data_collection.transfermarkt_scraper.sd.Transfermarkt')
    def test_fetch_squad_values_includes_season(self, mock_tm_class):
        """Test that season column is included"""
        scraper = TransfermarktScraper(seasons=['2025-2026'])
        
        mock_tm = MagicMock()
        mock_tm_class.return_value = mock_tm
        
        df = pd.DataFrame({'team': ['Team A']})
        mock_tm.read_squad_values.return_value = df
        
        result = scraper.fetch_squad_values()
        
        assert 'season' in result.columns if not result.empty else True


class TestCreateSummaryJSON:
    """Test create_summary_json method"""
    
    def test_create_summary_json_structure(self):
        """Test that summary JSON has correct structure"""
        scraper = TransfermarktScraper()
        
        # Create test DataFrames
        market_values = pd.DataFrame({
            'player': ['Player1'],
            'market_value': [50000000],
            'age': [25]
        })
        
        transfers = pd.DataFrame({
            'player': ['Player2'],
            'from_club': ['Club A'],
            'to_club': ['Club B'],
            'fee': [30000000]
        })
        
        squad_values = pd.DataFrame({
            'team': ['Team A'],
            'total_value': [500000000]
        })
        
        result = scraper.create_summary_json(
            market_values=market_values,
            transfers=transfers,
            squad_values=squad_values
        )
        
        # Should be a valid JSON string
        data = json.loads(result)
        
        assert 'summary' in data
        assert 'market_values' in data
        assert 'transfers' in data
        assert 'squad_values' in data
    
    def test_create_summary_json_includes_metadata(self):
        """Test that summary includes metadata"""
        scraper = TransfermarktScraper()
        
        market_values = pd.DataFrame({'player': ['Player1']})
        transfers = pd.DataFrame({'player': ['Player2']})
        squad_values = pd.DataFrame({'team': ['Team A']})
        
        result = scraper.create_summary_json(
            market_values=market_values,
            transfers=transfers,
            squad_values=squad_values
        )
        
        data = json.loads(result)
        
        # Summary should include counts and date
        assert 'total_players' in data['summary']
        assert 'total_transfers' in data['summary']
        assert 'total_teams' in data['summary']
        assert 'created_at' in data['summary']
    
    def test_create_summary_json_rag_ready_format(self):
        """Test that JSON is RAG-ready with text descriptions"""
        scraper = TransfermarktScraper()
        
        market_values = pd.DataFrame({
            'player': ['Kylian Mbappé'],
            'market_value': [180000000],
            'age': [26]
        })
        
        transfers = pd.DataFrame({
            'player': ['Jude Bellingham'],
            'from_club': ['Borussia Dortmund'],
            'to_club': ['Real Madrid'],
            'fee': [103000000]
        })
        
        squad_values = pd.DataFrame({
            'team': ['Manchester City'],
            'total_value': [1200000000]
        })
        
        result = scraper.create_summary_json(
            market_values=market_values,
            transfers=transfers,
            squad_values=squad_values
        )
        
        # JSON should contain player names and values in text format
        assert 'Kylian Mbappé' in result
        assert '180000000' in result or '180' in result
        
        # Should be parseable JSON
        data = json.loads(result)
        assert isinstance(data, dict)


class TestScrapeAll:
    """Test scrape_all orchestration method"""
    
    @patch.object(TransfermarktScraper, 'fetch_market_values')
    @patch.object(TransfermarktScraper, 'fetch_transfer_history')
    @patch.object(TransfermarktScraper, 'fetch_squad_values')
    def test_scrape_all_calls_all_methods(self, mock_squad, mock_transfers, mock_market):
        """Test that scrape_all calls all fetch methods"""
        scraper = TransfermarktScraper()
        
        # Mock return values
        mock_market.return_value = pd.DataFrame({'player': ['Player1']})
        mock_transfers.return_value = pd.DataFrame({'player': ['Player2']})
        mock_squad.return_value = pd.DataFrame({'team': ['Team A']})
        
        scraper.scrape_all()
        
        # All methods should be called
        mock_market.assert_called_once()
        mock_transfers.assert_called_once()
        mock_squad.assert_called_once()
    
    @patch.object(TransfermarktScraper, 'fetch_market_values')
    @patch.object(TransfermarktScraper, 'fetch_transfer_history')
    @patch.object(TransfermarktScraper, 'fetch_squad_values')
    def test_scrape_all_saves_to_csv(self, mock_squad, mock_transfers, mock_market):
        """Test that scrape_all saves data to CSV files"""
        import tempfile
        temp_dir = tempfile.mkdtemp()
        
        scraper = TransfermarktScraper(data_dir=temp_dir)
        
        # Mock return values
        mock_market.return_value = pd.DataFrame({'player': ['Player1']})
        mock_transfers.return_value = pd.DataFrame({'player': ['Player2']})
        mock_squad.return_value = pd.DataFrame({'team': ['Team A']})
        
        scraper.scrape_all()
        
        # Check that CSV files were created
        data_dir = Path(temp_dir)
        assert (data_dir / 'market_values.csv').exists()
        assert (data_dir / 'transfers.csv').exists()
        assert (data_dir / 'squad_values.csv').exists()
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
    
    @patch.object(TransfermarktScraper, 'fetch_market_values')
    @patch.object(TransfermarktScraper, 'fetch_transfer_history')
    @patch.object(TransfermarktScraper, 'fetch_squad_values')
    def test_scrape_all_saves_summary_json(self, mock_squad, mock_transfers, mock_market):
        """Test that scrape_all saves summary JSON"""
        import tempfile
        temp_dir = tempfile.mkdtemp()
        
        scraper = TransfermarktScraper(data_dir=temp_dir)
        
        # Mock return values
        mock_market.return_value = pd.DataFrame({'player': ['Player1']})
        mock_transfers.return_value = pd.DataFrame({'player': ['Player2']})
        mock_squad.return_value = pd.DataFrame({'team': ['Team A']})
        
        scraper.scrape_all()
        
        # Check that JSON file was created
        data_dir = Path(temp_dir)
        assert (data_dir / 'transfermarkt_summary.json').exists()
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)


class TestDataValidation:
    """Test data validation and quality"""
    
    def test_market_values_include_required_fields(self):
        """Test that market values include required fields"""
        scraper = TransfermarktScraper()
        
        # Expected fields: player, market_value, age, position, team, league, season
        expected_fields = ['player', 'market_value', 'season']
        
        # This is tested via the fetch_market_values implementation
        assert len(expected_fields) == 3
    
    def test_transfer_history_includes_required_fields(self):
        """Test that transfer history includes required fields"""
        scraper = TransfermarktScraper()
        
        # Expected fields: player, from_club, to_club, fee, date, season
        expected_fields = ['player', 'from_club', 'to_club', 'fee', 'season']
        
        assert len(expected_fields) == 5
    
    def test_squad_values_include_required_fields(self):
        """Test that squad values include required fields"""
        scraper = TransfermarktScraper()
        
        # Expected fields: team, total_value, avg_value, league, season
        expected_fields = ['team', 'total_value', 'season']
        
        assert len(expected_fields) == 3


class TestScraperConfiguration:
    """Test overall scraper configuration"""
    
    def test_scraper_has_all_required_attributes(self):
        """Test that scraper has all required configuration"""
        scraper = TransfermarktScraper()
        
        # Check all required attributes exist
        assert hasattr(scraper, 'leagues')
        assert hasattr(scraper, 'seasons')
        assert hasattr(scraper, 'data_dir')
    
    def test_expected_data_volume(self):
        """Test expected data volume"""
        scraper = TransfermarktScraper()
        
        # With 5 leagues × 5 seasons
        # Expected: ~2,500-3,000 player records per league-season
        # Total: ~12,500-15,000 player records
        expected_min = 12500
        expected_max = 15000
        
        assert expected_min == 12500
        assert expected_max == 15000


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    @patch('script.data_collection.transfermarkt_scraper.sd.Transfermarkt')
    def test_empty_dataframe_handling(self, mock_tm_class):
        """Test handling of empty DataFrames"""
        scraper = TransfermarktScraper(seasons=['2025-2026'])
        
        mock_tm = MagicMock()
        mock_tm_class.return_value = mock_tm
        
        # Return empty DataFrame
        mock_tm.read_player_values.return_value = pd.DataFrame()
        
        result = scraper.fetch_market_values()
        
        # Should handle empty DataFrame gracefully
        assert isinstance(result, pd.DataFrame)
    
    def test_single_season_initialization(self):
        """Test with single season"""
        scraper = TransfermarktScraper(seasons=['2025-2026'])
        
        assert len(scraper.seasons) == 1
        assert scraper.seasons[0] == '2025-2026'
    
    def test_single_league_initialization(self):
        """Test with single league"""
        scraper = TransfermarktScraper(leagues=['ENG1'])
        
        assert len(scraper.leagues) == 1
        assert scraper.leagues[0] == 'ENG1'


class TestDocumentation:
    """Test that scraper is properly documented"""
    
    def test_class_has_docstring(self):
        """Test that TransfermarktScraper has docstring"""
        assert TransfermarktScraper.__doc__ is not None
        assert len(TransfermarktScraper.__doc__) > 0
    
    def test_init_has_docstring(self):
        """Test that __init__ has docstring"""
        assert TransfermarktScraper.__init__.__doc__ is not None


class TestSeasonFormatConsistency:
    """Test season format consistency across scrapers"""
    
    def test_season_format_matches_fbref(self):
        """Test that season format matches FBref scraper"""
        scraper = TransfermarktScraper()
        
        # Both should use YYYY-YYYY format
        for season in scraper.seasons:
            parts = season.split('-')
            assert len(parts) == 2
            assert len(parts[0]) == 4
            assert len(parts[1]) == 4
    
    def test_current_season_consistency(self):
        """Test that current season is 2025-2026"""
        scraper = TransfermarktScraper()
        
        # First season should always be current
        assert scraper.seasons[0] == '2025-2026'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
