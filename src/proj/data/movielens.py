import zipfile
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import requests
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected

class MovielensDataLoader:
    RAW_DATA_DIR = Path("data/raw")
    PROCESSED_DATA_DIR = Path("data/processed")
    DOWNLOAD_CHUNK_SIZE = 1 << 20  # 1 MB

    def __init__(self, env: str = "dev"):
        self.env = env
        if self.env == "dev":
            self.dataset_name = "ml-latest-small"
        elif self.env == "prod":
            self.dataset_name = "ml-latest"
        else:
            raise ValueError(f"Invalid environment: {self.env}")
        self.processed_path = self.PROCESSED_DATA_DIR / f"{self.dataset_name}.pt"
        self.data = None

    def load_data(self, force: bool = False):
        if self.processed_path.exists() and not force:
            self.data = torch.load(self.processed_path, map_location=torch.device("cpu"))
        else:
            (
                self.movies,
                self.ratings,
            ) = self._download_movielens_dataset(self.dataset_name, force)
            self.data = self._build_movielens_graph(self.movies, self.ratings)
            self.processed_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.data, self.processed_path)

    def get_data(self, force: bool = False):
        if self.data is None or force:
            self.load_data(force)
        return self.data

    def _download_movielens_dataset(
        self, dataset_name: str, force: bool
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        url = f"https://files.grouplens.org/datasets/movielens/{dataset_name}.zip"
        zip_path = self.RAW_DATA_DIR / f"{dataset_name}.zip"
        self.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not zip_path.exists() or force:
            print(f"Downloading dataset {dataset_name}...")
            with requests.get(url, stream=True, timeout=60) as response:
                response.raise_for_status()
                with open(zip_path, "wb") as f:
                    for chunk in response.iter_content(self.DOWNLOAD_CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
            print(f"Dataset {dataset_name} downloaded successfully")
        else:
            print(f"Dataset {dataset_name} already downloaded")

        with zipfile.ZipFile(zip_path) as zip_ref:
            with zip_ref.open(f"{dataset_name}/movies.csv") as f:
                movies = pd.read_csv(f)
            with zip_ref.open(f"{dataset_name}/ratings.csv") as f:
                ratings = pd.read_csv(
                    f,
                    usecols=["userId", "movieId", "rating", "timestamp"],
                    dtype={
                        "userId": "int32",
                        "movieId": "int32",
                        "rating": "float32",
                        "timestamp": "int64",
                    },
                )
        return movies, ratings

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

        ts_np = ratings["timestamp"].to_numpy(dtype=np.int64, copy=False)
        t80 = np.quantile(ts_np, 0.8)
        t90 = np.quantile(ts_np, 0.9)
        ts_tensor = torch.from_numpy(ts_np)
        edge_store.train_mask = ts_tensor < t80
        edge_store.val_mask = (ts_tensor >= t80) & (ts_tensor < t90)
        edge_store.test_mask = ts_tensor >= t90

        return ToUndirected(merge=True)(data)
