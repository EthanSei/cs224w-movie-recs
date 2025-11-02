from recommender.data.movielens import MovielensDataLoader

import argparse
import torch
import torch_geometric.data as HeteroData

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Force download the dataset")
    parser.add_argument("--env", type=str, choices=["dev", "prod"], default="dev")
    args = parser.parse_args()

    data_loader = MovielensDataLoader(args.env)
    data = data_loader.get_data(args.force)
    print(data)