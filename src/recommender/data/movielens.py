from recommender.data.download_data import download_data

import numpy as np
import pandas as pd
import re
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import RandomLinkSplit, ToUndirected
from typing import Tuple, Optional

class MovielensDataLoader:
    # Constants
    MIN_YEAR = 1900
    MAX_YEAR = 2024
    NUM_YEAR_BUCKETS = 25
    YEARS_PER_BUCKET = 5
    MAX_RATING = 5.0
    
    KNOWN_YEARS = {
        "Babylon 5": 1993,
        "Ready Player One": 2018,
        "Hyena Road": 2015,
        "The Adventures of Sherlock Holmes and Doctor Watson": 1979,
        "Nocturnal Animals": 2016,
        "Paterson": 2016,
        "Moonlight": 2016,
        "The OA": 2016,
        "Cosmos": 2014,
        "Maria Bamford: Old Baby": 2017,
        "Death Note: Desu nôto (2006–2007)": 2006,
        "Generation Iron 2": 2017,
        "Black Mirror": 2011
    }

    def __init__(self, env: str = "dev"):
        self.env = env
        self.data = None

    def get_data(self, force: bool = False) -> HeteroData:
        """Get the full MovieLens graph data."""
        if self.data is None or force:
            movies, tags, ratings = download_data("movielens", force=force, env=self.env)
            self.data = self._build_movielens_graph(movies, tags, ratings)
        return self.data

    def get_train_val_test_data(self, force: bool = False) -> Tuple[HeteroData, HeteroData, HeteroData]:
        """Get train/val/test splits with user features built only from training data."""
        movies, tags, ratings = download_data("movielens", force=force, env=self.env)
        
        data = self._build_movielens_graph(
            movies, tags, ratings, 
            user_features=None, 
            apply_to_undirected=False
        )
        
        transform = RandomLinkSplit(
            num_val=0.1,
            num_test=0.1,
            edge_types=[("user", "rates", "item")],
            rev_edge_types=[("item", "rev_rates", "user")],
        )
        train_data, val_data, test_data = transform(data)
        
        train_ratings = self._extract_training_ratings(train_data, ratings, movies)
        _, users = pd.factorize(ratings["userId"].to_numpy(), sort=True)
        user_features = self._build_user_features(users, train_ratings, movies)
        
        train_data["user"].x = user_features
        val_data["user"].x = user_features
        test_data["user"].x = user_features
        
        train_data = ToUndirected(merge=True)(train_data)
        val_data = ToUndirected(merge=True)(val_data)
        test_data = ToUndirected(merge=True)(test_data)
        
        return train_data, val_data, test_data

    def _build_movielens_graph(
        self, 
        movies: pd.DataFrame, 
        tags: pd.DataFrame, 
        ratings: pd.DataFrame,
        user_features: Optional[torch.Tensor] = None,
        apply_to_undirected: bool = True
    ) -> HeteroData:
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
        items_filtered = movies[movies["movieId"].isin(movie_index)]
        
        if user_features is not None:
            data["user"].x = user_features
        else:
            data["user"].x = self._build_user_features(users, ratings, movies)
        
        data["item"].x = self._build_item_features(items_filtered)

        edge_store = data[("user", "rates", "item")]
        edge_store.edge_index = edge_index
        edge_store.rating = rating
        edge_store.timestamp = timestamp

        if apply_to_undirected:
            data = ToUndirected(merge=True)(data)

        return data

    def _build_user_features(
        self, 
        users: np.ndarray, 
        ratings: pd.DataFrame, 
        movies: pd.DataFrame
    ) -> torch.Tensor:
        user_to_idx = {user_id: idx for idx, user_id in enumerate(users)}
        ratings_with_movies = ratings.merge(
            movies[["movieId", "genres", "title"]], 
            on="movieId", 
            how="left"
        )
        
        unique_genres = self._get_unique_genres(movies)
        
        user_stats = self._compute_user_statistics(users, user_to_idx, ratings_with_movies)
        user_genre_prefs = self._compute_user_genre_preferences(
            users, user_to_idx, ratings_with_movies, unique_genres
        )
        user_year_prefs = self._compute_user_year_preferences(
            users, user_to_idx, ratings_with_movies
        )
        
        user_stats, user_genre_prefs, user_year_prefs = self._normalize_user_features(
            user_stats, user_genre_prefs, user_year_prefs
        )
        
        return torch.cat([user_stats, user_genre_prefs, user_year_prefs], dim=1)

    def _build_item_features(self, items: pd.DataFrame) -> torch.Tensor:
        multi_hot_genres = self._build_movie_genre_features(items)
        one_hot_year_buckets = self._build_movie_year_features(items)
        return torch.cat([multi_hot_genres, one_hot_year_buckets], dim=1)

    def _extract_training_ratings(
        self, 
        train_data: HeteroData, 
        ratings: pd.DataFrame, 
        movies: pd.DataFrame
    ) -> pd.DataFrame:
        train_edge_index = train_data[("user", "rates", "item")].edge_index
        user_codes, _ = pd.factorize(ratings["userId"].to_numpy(), sort=True)
        movie_index = pd.Index(movies["movieId"].drop_duplicates().to_numpy())
        movie_codes = movie_index.get_indexer(ratings["movieId"])
        
        edge_to_rating_idx = {
            (int(u), int(m)): idx for idx, (u, m) in enumerate(zip(user_codes, movie_codes))
        }
        
        train_edges = set(zip(
            train_edge_index[0].numpy(),
            train_edge_index[1].numpy()
        ))
        
        train_rating_indices = [
            idx for (u, m), idx in edge_to_rating_idx.items()
            if (u, m) in train_edges
        ]
        
        return ratings.iloc[train_rating_indices].copy()

    def _build_movie_genre_features(self, movies: pd.DataFrame) -> torch.Tensor:
        unique_genres = self._get_unique_genres(movies)
        genre_to_idx = {g: i for i, g in enumerate(unique_genres)}
        num_movies = len(movies)
        num_genres = len(unique_genres)
        
        # Split genres into lists, handling NaN
        genre_lists = movies["genres"].fillna("").str.split("|")
        
        # Build sparse coordinate lists for efficiency
        row_indices = []
        col_indices = []
        for i, genres in enumerate(genre_lists):
            for genre in genres:
                if genre in genre_to_idx:
                    row_indices.append(i)
                    col_indices.append(genre_to_idx[genre])
        
        # Create tensor and fill using advanced indexing
        movie_genre_matrix = torch.zeros(num_movies, num_genres, dtype=torch.float32)
        if row_indices:
            movie_genre_matrix[row_indices, col_indices] = 1.0
        
        return movie_genre_matrix

    def _build_movie_year_features(self, movies: pd.DataFrame) -> torch.Tensor:
        num_movies = len(movies)
        movie_years = torch.zeros(num_movies, self.NUM_YEAR_BUCKETS, dtype=torch.float32)
        
        # Vectorized year extraction
        extracted_years = pd.to_numeric(
            movies["title"].str.extract(r'\((\d{4})\)', expand=False), 
            errors="coerce"
        )
        
        # Handle known years for titles without parseable years
        missing_mask = extracted_years.isna()
        if missing_mask.any():
            base_titles = movies.loc[missing_mask.values, "title"].str.split(" (", regex=False).str[0]
            known_years = base_titles.map(self.KNOWN_YEARS)
            extracted_years = extracted_years.copy()
            extracted_years.iloc[missing_mask.values] = known_years.values
        
        # Get valid indices and compute buckets vectorized
        valid_mask = extracted_years.notna()
        if not valid_mask.any():
            return movie_years
        
        valid_years = extracted_years[valid_mask].values
        years_clamped = np.clip(valid_years, self.MIN_YEAR, self.MAX_YEAR)
        buckets = ((years_clamped - self.MIN_YEAR) // self.YEARS_PER_BUCKET).astype(int)
        buckets = np.clip(buckets, 0, self.NUM_YEAR_BUCKETS - 1)
        
        # Use positional indices (enumerate-style) not DataFrame indices
        valid_positions = np.where(valid_mask.values)[0]
        movie_years[valid_positions, buckets] = 1.0
        
        return movie_years

    def _get_unique_genres(self, movies: pd.DataFrame) -> list:
        unique_genres = list(movies["genres"].str.split("|").explode().unique())
        return [g for g in unique_genres if g and not pd.isna(g)]

    def _compute_user_statistics(
        self, 
        users: np.ndarray, 
        user_to_idx: dict, 
        ratings_with_movies: pd.DataFrame
    ) -> torch.Tensor:
        num_users = len(users)
        user_stats = torch.zeros(num_users, 3, dtype=torch.float32)
        
        # Vectorized aggregation using groupby
        stats = ratings_with_movies.groupby("userId")["rating"].agg(["mean", "count", "std"])
        stats["std"] = stats["std"].fillna(0.0)
        
        # Map user IDs to indices and fill tensor
        user_ids = stats.index.values
        indices = np.array([user_to_idx[uid] for uid in user_ids])
        user_stats[indices, 0] = torch.from_numpy(stats["mean"].values.astype(np.float32))
        user_stats[indices, 1] = torch.from_numpy(stats["count"].values.astype(np.float32))
        user_stats[indices, 2] = torch.from_numpy(stats["std"].values.astype(np.float32))
        
        return user_stats

    def _compute_user_genre_preferences(
        self,
        users: np.ndarray,
        user_to_idx: dict,
        ratings_with_movies: pd.DataFrame,
        unique_genres: list
    ) -> torch.Tensor:
        num_users = len(users)
        num_genres = len(unique_genres)
        user_genre_prefs = torch.zeros(num_users, num_genres, dtype=torch.float32)
        
        # Filter out rows with missing genres
        valid_ratings = ratings_with_movies[ratings_with_movies["genres"].notna()].copy()
        if len(valid_ratings) == 0:
            return user_genre_prefs
        
        # Explode genres into separate rows
        valid_ratings["genre_list"] = valid_ratings["genres"].str.split("|")
        exploded = valid_ratings.explode("genre_list")
        
        # Compute rating weight and aggregate by user-genre
        exploded["rating_weight"] = exploded["rating"] / self.MAX_RATING
        genre_sums = exploded.groupby(["userId", "genre_list"])["rating_weight"].sum()
        
        # Build genre to index mapping for fast lookup
        genre_to_idx = {g: i for i, g in enumerate(unique_genres)}
        
        # Fill tensor from aggregated results
        for (user_id, genre), weight in genre_sums.items():
            if genre in genre_to_idx and user_id in user_to_idx:
                user_genre_prefs[user_to_idx[user_id], genre_to_idx[genre]] = weight
        
        return user_genre_prefs

    def _compute_user_year_preferences(
        self,
        users: np.ndarray,
        user_to_idx: dict,
        ratings_with_movies: pd.DataFrame
    ) -> torch.Tensor:
        num_users = len(users)
        user_year_prefs = torch.zeros(num_users, self.NUM_YEAR_BUCKETS, dtype=torch.float32)
        
        # Vectorized year extraction
        df = ratings_with_movies.copy()
        extracted_years = df["title"].str.extract(r'\((\d{4})\)', expand=False)
        df["year"] = pd.to_numeric(extracted_years, errors="coerce")
        
        # Handle known years for titles without parseable years
        missing_mask = df["year"].isna()
        if missing_mask.any():
            base_titles = df.loc[missing_mask, "title"].str.split(" (", regex=False).str[0]
            df.loc[missing_mask, "year"] = base_titles.map(self.KNOWN_YEARS)
        
        # Filter to rows with valid years
        valid = df[df["year"].notna()].copy()
        if len(valid) == 0:
            return user_year_prefs
        
        # Compute buckets vectorized
        years_clamped = valid["year"].clip(self.MIN_YEAR, self.MAX_YEAR)
        valid["bucket"] = ((years_clamped - self.MIN_YEAR) // self.YEARS_PER_BUCKET).astype(int)
        valid["bucket"] = valid["bucket"].clip(0, self.NUM_YEAR_BUCKETS - 1)
        
        # Compute rating weight and aggregate
        valid["rating_weight"] = valid["rating"] / self.MAX_RATING
        bucket_sums = valid.groupby(["userId", "bucket"])["rating_weight"].sum()
        
        # Fill tensor from aggregated results
        for (user_id, bucket), weight in bucket_sums.items():
            if user_id in user_to_idx:
                user_year_prefs[user_to_idx[user_id], bucket] = weight
        
        return user_year_prefs

    def _normalize_user_features(
        self,
        user_stats: torch.Tensor,
        user_genre_prefs: torch.Tensor,
        user_year_prefs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_ratings = torch.clamp(user_stats[:, 1:2], min=1.0)
        user_genre_prefs = user_genre_prefs / num_ratings
        user_year_prefs = user_year_prefs / num_ratings
        
        user_stats[:, 0] = user_stats[:, 0] / self.MAX_RATING
        
        max_num_ratings = torch.max(user_stats[:, 1])
        if max_num_ratings > 0:
            user_stats[:, 1] = torch.log1p(user_stats[:, 1]) / torch.log1p(max_num_ratings)
        
        max_std = torch.max(user_stats[:, 2])
        if max_std > 0:
            user_stats[:, 2] = user_stats[:, 2] / max_std
        
        return user_stats, user_genre_prefs, user_year_prefs

    def _extract_movie_year(self, title: str) -> Optional[int]:
        match = re.search(r'\((\d{4})\)', title)
        if match:
            try:
                return int(match.group(1))
            except (ValueError, TypeError):
                pass
        
        base_title = title.split(" (")[0] if " (" in title else title
        return self.KNOWN_YEARS.get(base_title, None)

    def _extract_movie_year_from_row(self, row: pd.Series, extracted_year: pd.Series) -> Optional[int]:
        if pd.notna(extracted_year):
            try:
                return int(extracted_year)
            except (ValueError, TypeError):
                pass
        
        title = row["title"]
        base_title = title.split(" (")[0] if " (" in title else title
        return self.KNOWN_YEARS.get(base_title, None)

    def _year_to_bucket(self, year: int) -> int:
        year = max(self.MIN_YEAR, min(self.MAX_YEAR, year))
        half_decade = (year - self.MIN_YEAR) // self.YEARS_PER_BUCKET
        return max(0, min(self.NUM_YEAR_BUCKETS - 1, half_decade))
