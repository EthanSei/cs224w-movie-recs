import pytest
import torch
from recommender.data.movielens import MovielensDataLoader


@pytest.fixture
def loader():
    """MovielensDataLoader instance for testing."""
    return MovielensDataLoader()


class TestMovieGenreFeatures:
    """Test genre feature building for movies."""
    
    def test_genre_features_basic(self, loader, sample_movies_df):
        """Test basic genre feature construction."""
        genre_features = loader._build_movie_genre_features(sample_movies_df)
        
        # Should have shape (num_movies, num_unique_genres)
        assert genre_features.shape[0] == 3
        assert genre_features.shape[1] == 5  # Action, Adventure, Comedy, Romance, Drama
        
        # Verify multi-hot encoding by checking sums
        assert genre_features[0].sum() == 2.0  # Action + Adventure
        assert genre_features[1].sum() == 2.0  # Comedy + Romance
        assert genre_features[2].sum() == 1.0  # Drama
        
        # Verify all values are binary (0 or 1)
        assert ((genre_features == 0) | (genre_features == 1)).all()
        
        # Verify correct number of genres encoded
        assert (genre_features[0] == 1.0).sum() == 2
        assert (genre_features[1] == 1.0).sum() == 2
        assert (genre_features[2] == 1.0).sum() == 1
    
    def test_genre_features_empty_genres(self, loader, movies_edge_cases):
        """Test handling of empty or missing genres."""
        genre_features = loader._build_movie_genre_features(movies_edge_cases)
        
        assert genre_features.shape[0] == 4
        assert genre_features[0].sum() == 2.0  # Action + Adventure
        assert genre_features[1].sum() == 0.0  # Empty genres
    
    def test_genre_features_single_genre(self, loader, sample_movies_df):
        """Test movies with single genre."""
        movies = sample_movies_df.iloc[[2]]  # Just the Drama movie
        genre_features = loader._build_movie_genre_features(movies)
        
        assert genre_features.shape == (1, 1)
        assert genre_features[0, 0] == 1.0
        assert genre_features[0].sum() == 1.0


class TestMovieYearFeatures:
    """Test year feature building for movies."""
    
    def test_year_features_basic(self, loader, sample_movies_df):
        """Test basic year feature construction."""
        year_features = loader._build_movie_year_features(sample_movies_df)
        
        assert year_features.shape == (3, 25)
        
        # 1995 -> (1995 - 1900) // 5 = 19
        assert year_features[0, 19] == 1.0
        assert year_features[0].sum() == 1.0
        
        # 2000 -> (2000 - 1900) // 5 = 20
        assert year_features[1, 20] == 1.0
        assert year_features[1].sum() == 1.0
        
        # 2010 -> (2010 - 1900) // 5 = 22
        assert year_features[2, 22] == 1.0
        assert year_features[2].sum() == 1.0
    
    def test_year_features_known_years(self, loader, movies_with_known_years):
        """Test movies with known years (no year in title)."""
        year_features = loader._build_movie_year_features(movies_with_known_years)
        
        assert year_features.shape == (2, 25)
        
        # Ready Player One: 2018 -> (2018 - 1900) // 5 = 23
        assert year_features[0, 23] == 1.0
        assert year_features[0].sum() == 1.0
        
        # Moonlight: 2016 -> (2016 - 1900) // 5 = 23
        assert year_features[1, 23] == 1.0
        assert year_features[1].sum() == 1.0
    
    def test_year_features_year_bounds(self, loader, movies_edge_cases):
        """Test year clamping to valid range."""
        year_features = loader._build_movie_year_features(movies_edge_cases)
        
        # 1890 should be clamped to 1900 -> bucket 0
        assert year_features[2, 0] == 1.0
        
        # 2030 should be clamped to 2024 -> bucket 24
        assert year_features[3, 24] == 1.0
    
    def test_year_features_no_year(self, loader, movies_no_year):
        """Test movies without extractable years."""
        year_features = loader._build_movie_year_features(movies_no_year)
        
        assert year_features.shape == (1, 25)
        assert year_features[0].sum() == 0.0


class TestItemFeatures:
    """Test complete item feature construction."""
    
    def test_item_features_construction(self, loader, sample_movies_df):
        """Test that item features combine genre and year correctly."""
        item_features = loader._build_item_features(sample_movies_df)
        
        # Should concatenate genre features and year features
        genre_features = loader._build_movie_genre_features(sample_movies_df)
        year_features = loader._build_movie_year_features(sample_movies_df)
        
        expected_shape = (3, genre_features.shape[1] + year_features.shape[1])
        assert item_features.shape == expected_shape
        
        # Verify concatenation
        expected_features = torch.cat([genre_features, year_features], dim=1)
        assert torch.allclose(item_features, expected_features)
    
    def test_item_features_real_data(self, loader, movies_realistic):
        """Test with realistic MovieLens data structure."""
        item_features = loader._build_item_features(movies_realistic)
        
        assert item_features.shape[0] == 3
        assert (item_features >= 0).all()
        
        # Verify genre and year features are binary
        genre_features = loader._build_movie_genre_features(movies_realistic)
        year_features = loader._build_movie_year_features(movies_realistic)
        assert ((genre_features == 0) | (genre_features == 1)).all()
        assert ((year_features == 0) | (year_features == 1)).all()
        
        # Verify each movie has at least one feature set
        assert (item_features.sum(dim=1) > 0).all()

