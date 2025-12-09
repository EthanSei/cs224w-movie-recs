import torch
import numpy as np
from collections import defaultdict
import logging
from recommender.utils.device import get_device, clear_memory

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, k=10, device="auto", batch_size=128, item_batch_size=512):
        """
        Args:
            k: Number of top items to consider for Hit@k, NDCG@k, Recall@k
            device: Device to use for computation
            batch_size: Number of users to process at once (reduce if OOM)
            item_batch_size: Number of items to score at once per user batch (reduce if OOM)
        """
        self.k = k
        self.device = get_device(device)
        self.batch_size = batch_size
        self.item_batch_size = item_batch_size

    def evaluate(self, model, data, train_data=None, val_data=None):
        """
        Evaluate model using memory-efficient chunked scoring.
        """
        model.eval()
        model.to(self.device)

        # Use train_data graph structure for encoding if available
        encode_data = train_data if train_data is not None else data

        with torch.no_grad():
            z_dict = model.encode(
                {ntype: encode_data[ntype].x.to(self.device) for ntype in encode_data.node_types},
                {etype: encode_data[etype].edge_index.to(self.device) for etype in encode_data.edge_types}
            )
            user_emb = z_dict["user"]
            item_emb = z_dict["item"]

        num_users, num_items = user_emb.size(0), item_emb.size(0)

        # Get test edges - filter to POSITIVE edges only
        edge_store = data[("user", "rates", "item")]
        edge_label_index = edge_store.edge_label_index
        if hasattr(edge_store, "edge_label"):
            pos_mask = edge_store.edge_label == 1
            edge_label_index = edge_label_index[:, pos_mask]

        user_to_test_items = defaultdict(set)
        for u, i in edge_label_index.T:
            user_to_test_items[u.item()].add(i.item())
        
        num_test_users = len(user_to_test_items)
        avg_test_items_per_user = np.mean([len(items) for items in user_to_test_items.values()]) if user_to_test_items else 0
        logger.info(f"Evaluation stats: {num_test_users} users with test items, avg {avg_test_items_per_user:.2f} test items per user")

        # Build sets of items to mask (train + validation)
        user_to_train_items = defaultdict(set)
        if train_data is not None:
            train_edge_index = train_data[("user", "rates", "item")].edge_index
            for u, i in train_edge_index.T:
                user_to_train_items[u.item()].add(i.item())
        else:
            train_edge_index = data[("user", "rates", "item")].edge_index
            for u, i in train_edge_index.T:
                user_to_train_items[u.item()].add(i.item())
        
        # Also mask validation items if val_data is provided
        user_to_val_items = defaultdict(set)
        if val_data is not None and hasattr(val_data[("user", "rates", "item")], "edge_label_index"):
            val_edge_store = val_data[("user", "rates", "item")]
            val_edge_label_index = val_edge_store.edge_label_index
            if hasattr(val_edge_store, "edge_label"):
                val_pos_mask = val_edge_store.edge_label == 1
                val_edge_label_index = val_edge_label_index[:, val_pos_mask]
            for u, i in val_edge_label_index.T:
                user_to_val_items[u.item()].add(i.item())

        hits, ndcgs, recalls = [], [], []
        
        # Get list of test users to process
        test_users = list(user_to_test_items.keys())
        
        # Process users in batches for memory efficiency
        with torch.no_grad():
            for batch_start in range(0, len(test_users), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(test_users))
                batch_users = test_users[batch_start:batch_end]
                
                # Get user embeddings for this batch
                user_indices = torch.tensor(batch_users, device=self.device, dtype=torch.long)
                batch_user_emb = user_emb[user_indices]  # [batch_size, emb_dim]
                
                # Score items in chunks to avoid OOM
                batch_scores = self._compute_scores_chunked(
                    model, batch_user_emb, item_emb, num_items
                )
                
                # Apply masking and compute metrics for each user in batch
                for idx, u in enumerate(batch_users):
                    user_scores = batch_scores[idx].clone()
                    
                    # Mask out train items
                    train_items = user_to_train_items.get(u, set())
                    if train_items:
                        train_items_tensor = torch.tensor(list(train_items), device=self.device, dtype=torch.long)
                        user_scores[train_items_tensor] = float('-inf')
                    
                    # Mask out validation items
                    val_items = user_to_val_items.get(u, set())
                    if val_items:
                        val_items_tensor = torch.tensor(list(val_items), device=self.device, dtype=torch.long)
                        user_scores[val_items_tensor] = float('-inf')
                    
                    topk_scores, topk_indices = torch.topk(user_scores, k=min(self.k, user_scores.size(0)))
                    topk_items = topk_indices.cpu().numpy()
                    
                    relevant_items = user_to_test_items[u]
                    
                    hit = int(any(i in topk_items for i in relevant_items))
                    hits.append(hit)

                    ndcg_val = 0
                    for i in relevant_items:
                        if i in topk_items:
                            rank = np.where(topk_items == i)[0][0] + 1
                            ndcg_val += 1 / np.log2(rank + 1)
                    if len(relevant_items) > 0:
                        ideal_dcg = sum(1 / np.log2(r + 1) for r in range(1, min(len(relevant_items), self.k) + 1))
                        if ideal_dcg > 0:
                            ndcg_val /= ideal_dcg
                    ndcgs.append(ndcg_val)

                    num_relevant_in_topk = sum(i in topk_items for i in relevant_items)
                    recall = num_relevant_in_topk / len(relevant_items) if len(relevant_items) > 0 else 0
                    recalls.append(recall)
                
                # Clear intermediate tensors
                del batch_scores, batch_user_emb
                
        # Clear memory after evaluation
        clear_memory()

        hr = np.mean(hits) if hits else 0
        ndcg = np.mean(ndcgs) if ndcgs else 0
        recall = np.mean(recalls) if recalls else 0

        logger.info(f"Evaluation results - Hit@{self.k}: {hr:.4f}, NDCG@{self.k}: {ndcg:.4f}, Recall@{self.k}: {recall:.4f}")
        
        return hr, ndcg, recall
    
    def _compute_scores_chunked(
        self, 
        model, 
        user_emb: torch.Tensor, 
        item_emb: torch.Tensor,
        num_items: int
    ) -> torch.Tensor:
        """
        Compute user-item scores in chunks to reduce memory usage.
        """
        batch_size = user_emb.size(0)
        emb_dim = user_emb.size(1)
        scores = torch.zeros(batch_size, num_items, device=self.device)
        
        for item_start in range(0, num_items, self.item_batch_size):
            item_end = min(item_start + self.item_batch_size, num_items)
            item_chunk = item_emb[item_start:item_end]  # [chunk_size, emb_dim]
            chunk_size = item_chunk.size(0)
            
            # Expand for batch scoring: [batch_size, chunk_size, emb_dim]
            user_expanded = user_emb.unsqueeze(1).expand(-1, chunk_size, -1)
            item_expanded = item_chunk.unsqueeze(0).expand(batch_size, -1, -1)
            
            # Flatten for model.score: [batch_size * chunk_size, emb_dim]
            user_flat = user_expanded.reshape(-1, emb_dim)
            item_flat = item_expanded.reshape(-1, emb_dim)
            
            # Score and reshape back
            chunk_scores = model.score(user_flat, item_flat)
            scores[:, item_start:item_end] = chunk_scores.reshape(batch_size, chunk_size)
            
            # Clean up intermediate tensors
            del user_expanded, item_expanded, user_flat, item_flat, chunk_scores
        
        return scores
