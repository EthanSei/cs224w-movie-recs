import torch
import numpy as np
from collections import defaultdict
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def evaluate(model, data, k=10, device="cpu"):
    """
    Evaluate a trained recommender model on held-out data.
    Computes Hit@K, NDCG@K, and Recall@K.
    """
    model.eval()
    model.to(device)

    # Build embeddings
    with torch.no_grad():
        z_dict = model.encode(
            {ntype: data[ntype].x.to(device) for ntype in data.node_types},
            {etype: data[etype].edge_index.to(device) for etype in data.edge_types}
        )
        user_emb = z_dict["user"]
        item_emb = z_dict["item"]

    # Compute full score matrix
    scores = torch.matmul(user_emb, item_emb.T)  # [num_users, num_items]

    # Ground truth edges
    edge_label_index = data[("user", "rates", "item")].edge_label_index

    # Build dict: user -> set of relevant items
    user_to_items = defaultdict(set)
    for u, i in edge_label_index.T:
        user_to_items[u.item()].add(i.item())

    hits, ndcgs, recalls = [], [], []

    for u, relevant_items in user_to_items.items():
        user_scores = scores[u]
        topk_items = torch.topk(user_scores, k=k).indices.cpu().numpy()

        # Hit@K: Did we get AT LEAST ONE relevant item?
        hit = int(any(i in topk_items for i in relevant_items))
        hits.append(hit)

        # NDCG@K: Evaluation metric that rewards placing relevant items HIGHER in the ranking
        ndcg_val = 0
        for i in relevant_items:
            if i in topk_items:
                rank = np.where(topk_items == i)[0][0] + 1
                ndcg_val += 1 / np.log2(rank + 1)
        if len(relevant_items) > 0:
            ideal_dcg = sum(1 / np.log2(r + 1) for r in range(1, min(len(relevant_items), k) + 1))
            ndcg_val /= ideal_dcg
        ndcgs.append(ndcg_val)

        # Recall@K: Proportion of relevant items found in top K
        num_relevant_in_topk = sum(i in topk_items for i in relevant_items)
        recall = num_relevant_in_topk / len(relevant_items)
        recalls.append(recall)

    hr = np.mean(hits)
    ndcg = np.mean(ndcgs)
    recall = np.mean(recalls)

    logger.info(f"Evaluation results - Hit@{k}: {hr:.4f}, NDCG@{k}: {ndcg:.4f}, Recall@{k}: {recall:.4f}")
    logger.info("Evaluation completed.")
    
    return hr, ndcg, recall
