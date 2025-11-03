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
            edge_types=[("user", "rates", "movie")],
            rev_edge_types=[("movie", "rev_rates", "user")],
        )
        train_data, val_data, test_data = transform(data)
        return train_data, val_data, test_data

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
        data["user"].num_nodes = len(users)
        data["movie"].num_nodes = len(movie_index)

        edge_store = data[("user", "rates", "movie")]
        edge_store.edge_index = edge_index
        edge_store.rating = rating
        edge_store.timestamp = timestamp

        data = ToUndirected(merge=True)(data)

        return data
