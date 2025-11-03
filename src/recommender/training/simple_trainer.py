import copy
import tqdm
import torch
import torch_geometric.data as HeteroData

class SimpleTrainer:
    def __init__(self, num_epochs: int):
        self.num_epochs = num_epochs

    def fit(
        self,
        model: torch.nn.Module,
        train_data: HeteroData,
        val_data: HeteroData,
        optimizer: torch.optim.Optimizer = None,
        loss_fn: torch.nn.Module = None,
        device: torch.device = torch.device("cpu"),
        verbose: bool = True,
    ) -> None:
        model.to(device)

        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        if loss_fn is None:
            loss_fn = torch.nn.BCEWithLogitsLoss()

        edge_store = train_data[("user", "rates", "movie")]
        user_store = train_data["user"]
        movie_store = train_data["movie"]

        user_features = self._resolve_features(
            user_store,
            getattr(getattr(model, "user_tower", [None])[0], "in_features", None),
            device,
        )
        movie_features = self._resolve_features(
            movie_store,
            getattr(getattr(model, "item_tower", [None])[0], "in_features", None),
            device,
        )

        edge_label_index = edge_store.edge_label_index.to(device)
        edge_label = edge_store.edge_label.float().to(device)
        best_loss = float("inf")
        best_model = None

        for epoch in tqdm.tqdm(range(self.num_epochs), disable=not verbose):
            model.train()
            optimizer.zero_grad(set_to_none=True)

            user_embeddings, item_embeddings = model(user_features, movie_features)

            logits = (user_embeddings[edge_label_index[0]] * item_embeddings[edge_label_index[1]]).sum(dim=-1)
            loss = loss_fn(logits, edge_label)
            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_model = copy.deepcopy(model)

            model.eval()
            with torch.no_grad():
                # TODO: Evaluate the model
                pass

        if verbose:
            print(f"Best loss: {best_loss}")
        return best_model

    def _resolve_features(
        self, store: HeteroData, in_dim: int, device: torch.device
    ) -> torch.Tensor:
        features = getattr(store, "x", None)
        if features is None:
            if in_dim is None:
                raise ValueError(
                    "Cannot infer feature dimension for nodes without attributes."
                )
            with torch.no_grad():
                features = torch.randn(store.num_nodes, in_dim)
            store.x = features
        return features.to(device)
