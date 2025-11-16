from recommender.data.download_data import download_data

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import RandomLinkSplit, ToUndirected
from typing import Tuple

class MovielensDataLoader:
    def __init__(self, env: str = "dev"):
        self.env = env
        self.data = None

    def get_data(self, force: bool = False) -> HeteroData:
        if self.data is None or force:
            movies, tags, ratings = download_data("movielens", force=force, env=self.env)
            self.data = self._build_movielens_graph(movies, tags, ratings)
        return self.data

    def get_train_val_test_data(self, force: bool = False) -> Tuple[HeteroData, HeteroData, HeteroData]:
        data = self.get_data(force)
        transform = RandomLinkSplit(
            num_val=0.1,
            num_test=0.1,
            edge_types=[("user", "rates", "item")],
            rev_edge_types=[("item", "rev_rates", "user")],
        )
        train_data, val_data, test_data = transform(data)
        return train_data, val_data, test_data

    def _build_user_features(self, users: np.ndarray, feature_dim: int = 100) -> torch.Tensor:
        num_users = len(users)
        # TODO: Replace with features derived from user data
        features = torch.randn(num_users, feature_dim)
        return features

    def _build_item_features(self, items: pd.DataFrame) -> torch.Tensor:
        # Build item features by concatenating genre and year features
        multi_hot_genres: torch.Tensor = self._build_movie_genre_features(items)
        one_hot_year_buckets: torch.Tensor = self._build_movie_year_features(items)

        features = torch.cat([multi_hot_genres, one_hot_year_buckets], dim=1)
        return features

    def _build_movielens_graph(self, movies: pd.DataFrame, tags: pd.DataFrame, ratings: pd.DataFrame):
        movie_index = pd.Index(movies["movieId"].drop_duplicates().to_numpy())
        movie_codes = movie_index.get_indexer(ratings["movieId"])
        valid_mask = movie_codes >= 0
        if not np.all(valid_mask):
            ratings = ratings.loc[valid_mask].reset_index(drop=True)
            movie_codes = movie_codes[valid_mask]

        user_codes, users = pd.factorize(ratings["userId"].to_numpy(), sort=True)

        src = torch.from_numpy(user_codes.astype(np.int64, copy=False))
        dst = torch.from_numpy(movie_codes.astype(np.int64, copy=False))
        edge_index = torch.stack((src, dst), dim=0)

        rating = torch.from_numpy(ratings["rating"].to_numpy(dtype=np.float32, copy=True))
        timestamp = torch.from_numpy(
            ratings["timestamp"].to_numpy(dtype=np.int64, copy=True)
        )

        data = HeteroData()
        
        # Build node features using helper functions
        # Note: movie_index contains unique items that appear in ratings
        # We filter movies DataFrame to match movie_index for consistency
        items_filtered = movies[movies["movieId"].isin(movie_index)]
        data["user"].x = self._build_user_features(users)
        data["item"].x = self._build_item_features(items_filtered)

        edge_store = data[("user", "rates", "item")]
        edge_store.edge_index = edge_index
        edge_store.rating = rating
        edge_store.timestamp = timestamp

        data = ToUndirected(merge=True)(data)

        return data

    def _build_movie_genre_features(self, movies: pd.DataFrame) -> torch.Tensor:
        unique_genres = list(movies["genres"].str.split("|").explode().unique())
        unique_genres = [g for g in unique_genres if g and not pd.isna(g)]  # Filter out empty/NaN
        
        movie_genre_matrix = torch.zeros(len(movies), len(unique_genres), dtype=torch.float32)
        
        for i, movie_genres_str in enumerate(movies["genres"]):
            if pd.isna(movie_genres_str):
                continue
            movie_genres_list = movie_genres_str.split("|")
            for genre in movie_genres_list:
                if genre in unique_genres:
                    genre_idx = unique_genres.index(genre)
                    movie_genre_matrix[i, genre_idx] = 1.0
        
        return movie_genre_matrix

    def _build_movie_year_features(self, movies: pd.DataFrame) -> torch.Tensor:
        # Known years for movies without years in their titles
        known_years = {
            "Babylon 5": 1993,  # TV series, first episode
            "Ready Player One": 2018,
            "Hyena Road": 2015,
            "The Adventures of Sherlock Holmes and Doctor Watson": 1979,  # TV series
            "Nocturnal Animals": 2016,
            "Paterson": 2016,
            "Moonlight": 2016,
            "The OA": 2016,  # TV series
            "Cosmos": 2014,  # TV series
            "Maria Bamford: Old Baby": 2017,
            "Death Note: Desu nôto (2006–2007)": 2006,  # TV series
            "Generation Iron 2": 2017,
            "Black Mirror": 2011  # TV series
        }
        
        # Extract years from titles (format: "Title (YYYY)")
        extracted_years = movies["title"].str.extract(r'\((\d{4})\)', expand=False)
        
        movie_years = torch.zeros(len(movies), 25, dtype=torch.float32)
        
        for idx, row in movies.iterrows():
            year = None
            
            # Try to extract year from title
            extracted_year = extracted_years.iloc[idx]
            if pd.notna(extracted_year):
                try:
                    year = int(extracted_year)
                except (ValueError, TypeError):
                    pass
            
            # If no year found in title, check known_years
            if year is None:
                title = row["title"]
                # Get base title (remove year ranges like "2006–2007")
                base_title = title.split(" (")[0] if " (" in title else title
                year = known_years.get(base_title, None)
            
            # Bucketize year into half-decades (5-year periods)
            if year is not None:
                # Clamp year to valid range [1900, 2024]
                year = max(1900, min(2024, year))
                half_decade = (year - 1900) // 5
                # Ensure index is within bounds [0, 24]
                half_decade = max(0, min(24, half_decade))
                movie_years[idx, half_decade] = 1.0
        
        return movie_years