import copy
import tqdm
import torch
import torch_geometric.data as HeteroData
from recommender.training.simple_trainer import SimpleTrainer


class GATTrainer(SimpleTrainer):

    def fit(
        self,
        model: torch.nn.Module,
        train_data: HeteroData,
        val_data: HeteroData,
        optimizer: torch.optim.Optimizer = None,
        loss_fn: torch.nn.Module = None,
        device: torch.device = torch.device("cpu"),
        verbose: bool = True,
    ) -> torch.nn.Module:

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
            model.user_proj.in_features,
            device,
        )
        movie_features = self._resolve_features(
            movie_store,
            model.item_proj.in_features,
            device,
        )
        edge_index_dict = {
            ('user', 'rates', 'movie'): train_data[('user', 'rates', 'movie')].edge_index.to(device),
            ('movie', 'rev_rates', 'user'): train_data[('movie', 'rev_rates', 'user')].edge_index.to(device)
        }
        
        edge_label_index = edge_store.edge_label_index.to(device)
        edge_label = edge_store.edge_label.float().to(device)
        
        best_loss = float("inf")
        best_model = None
        #Training loop
        for epoch in tqdm.tqdm(range(self.num_epochs), disable=not verbose):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            
            user_embeddings, item_embeddings = model(
                user_features, 
                movie_features, 
                edge_index_dict
            )
            
            # Compute predictions via dot product for labeled edges
            logits = (
                user_embeddings[edge_label_index[0]] * 
                item_embeddings[edge_label_index[1]]
            ).sum(dim=-1)
            
        
            loss = loss_fn(logits, edge_label)
            
            loss.backward()
            optimizer.step()
            
            # Track best model
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_model = copy.deepcopy(model)
            
            # Validation evaluation (optional)
            model.eval()
            with torch.no_grad():
                # TODO: Implement validation metrics
                # Could compute AUC, recall@K, etc. on val_data
                pass
        
        if verbose:
            print(f"Best training loss: {best_loss:.4f}")
        
        return best_model

