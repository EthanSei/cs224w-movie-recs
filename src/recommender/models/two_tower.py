import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoTower(nn.Module):
    def __init__(self, user_dim, item_dim, hidden_dim=128, dropout=0.1):
        super(TwoTower, self).__init__()
        self.user_tower = nn.Sequential(
            nn.Linear(user_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
        )
        self.item_tower = nn.Sequential(
            nn.Linear(item_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
        )
    
    def forward(self, users, items):
        user_embeddings = self.user_tower(users)
        item_embeddings = self.item_tower(items)
        return user_embeddings, item_embeddings