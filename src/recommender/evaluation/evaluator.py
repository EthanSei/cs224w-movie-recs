import torch
import numpy as np
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, k=10, device="cpu"):
        self.k = k
        self.device = device

    def evaluate(self, model, data, train_data=None, val_data=None):
        model.eval()
        model.to(self.device)

        # Use train_data graph structure for encoding if available, otherwise use data
        # This is critical: graph-based models should encode using the graph they were trained on
        encode_data = train_data if train_data is not None else data

        with torch.no_grad():
            z_dict = model.encode(
                {ntype: encode_data[ntype].x.to(self.device) for ntype in encode_data.node_types},
                {etype: encode_data[etype].edge_index.to(self.device) for etype in encode_data.edge_types}
            )
            user_emb = z_dict["user"]
            item_emb = z_dict["item"]

        num_users, num_items = user_emb.size(0), item_emb.size(0)
        scores = torch.zeros(num_users, num_items, device=self.device)
        
        batch_size = 1000
        for i in range(0, num_users, batch_size):
            end_i = min(i + batch_size, num_users)
            user_batch = user_emb[i:end_i]
            user_expanded = user_batch.unsqueeze(1).expand(-1, num_items, -1)
            item_expanded = item_emb.unsqueeze(0).expand(end_i - i, -1, -1)
            
            user_flat = user_expanded.reshape(-1, user_emb.size(1))
            item_flat = item_expanded.reshape(-1, item_emb.size(1))
            
            scores_flat = model.score(user_flat, item_flat)
            scores[i:end_i] = scores_flat.reshape(end_i - i, num_items)

        # Get test edges - filter to POSITIVE edges only
        # RandomLinkSplit includes both positives and sampled negatives in edge_label_index
        # edge_label indicates which are positive (1) vs negative (0)
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
        
        # Also mask validation items if val_data is provided - filter to positives only
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

        for u, relevant_items in user_to_test_items.items():
            user_scores = scores[u].clone()
            
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

        hr = np.mean(hits) if hits else 0
        ndcg = np.mean(ndcgs) if ndcgs else 0
        recall = np.mean(recalls) if recalls else 0

        logger.info(f"Evaluation results - Hit@{self.k}: {hr:.4f}, NDCG@{self.k}: {ndcg:.4f}, Recall@{self.k}: {recall:.4f}")
        
        return hr, ndcg, recall
