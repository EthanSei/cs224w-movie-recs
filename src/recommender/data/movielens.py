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
            movies, ratings = download_data("movielens", force=force, env=self.env)
            self.data = self._build_movielens_graph(movies, ratings)
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

    def _build_item_features(self, items: pd.DataFrame, feature_dim: int = 100) -> torch.Tensor:
        num_items = len(items)
        # TODO: Replace with features derived from item data
        features = torch.randn(num_items, feature_dim)
        return features

    def _build_movielens_graph(self, movies: pd.DataFrame, ratings: pd.DataFrame):
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
