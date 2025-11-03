import requests
import zipfile
import pandas as pd
from typing import Tuple
from pathlib import Path
import torch

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")
DOWNLOAD_CHUNK_SIZE = 1 << 20  # 1 MB

def download_data(dataset_name: str, force: bool = False, env: str = "dev", chunk_size: int = DOWNLOAD_CHUNK_SIZE):
    processed_path = PROCESSED_DATA_DIR / f"{dataset_name}.pt"
    if processed_path.exists() and not force:
        return torch.load(processed_path, map_location=torch.device("cpu"))
    
    if dataset_name == "movielens" and env == "dev":
        return _download_movielens_dataset("ml-latest-small", chunk_size=chunk_size)
    elif dataset_name == "movielens" and env == "prod":
        return _download_movielens_dataset("ml-latest", chunk_size=chunk_size)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported for environment {env}")

def _download_movielens_dataset(dataset_name: str, chunk_size: int = DOWNLOAD_CHUNK_SIZE) -> Tuple[pd.DataFrame, pd.DataFrame]:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    url = f"https://files.grouplens.org/datasets/movielens/{dataset_name}.zip"
    zip_path = RAW_DATA_DIR / f"{dataset_name}.zip"
    if not zip_path.exists():
        print(f"Downloading dataset {dataset_name}...")
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size):
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