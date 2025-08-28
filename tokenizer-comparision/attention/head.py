import torch
from torch import nn
import torch.nn.functional as F

class Head(nn.Module):
    """One head of self-attention"""

    def __init__(self, n_embds, head_size):
        super.__init__()
        self.key = nn.Linear(n_embds, head_size, bias=False)
        self.query = nn.Linear(n_embds, head_size, bias=False)
        self.value = nn.Linear(n_embds, head_size, bias=False)

    def forward(self, X):
        B, T, C = X.shape
        k = self.key(X) # (B, T, C)
        q = self.query(X) # (B, T, C)
        weights = q @ k.transpose(-2, -1) * C ** -0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1) # (B, T, T)
        v = self.value(X)
        out = weights @ v
        return out
    


class MultiHeadAttention(nn.Module):

        def __init__(self, num_heads, head_size, n_embds) -> None:
            super().__init__()
            self.heads = nn.ModuleList([Head(n_embds, head_size) for _ in range(num_heads)])
        
        def forward(self, X):
            return torch.cat([h(X) for h in self.heads], dim=-1)
