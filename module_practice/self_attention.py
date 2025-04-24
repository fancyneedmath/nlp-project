import math
import torch
import torch.nn as nn

import warnings
warnings.filterwarnings(action="ignore")

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim) -> None:
        super().__init__()
        self.dim = hidden_dim

        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.attention_drop = nn.Dropout(0.1)
    
    def forward(self, X, attention_mask=None):
        Q = self.query_proj(X)
        K = self.key_proj(X)
        V = self.value_proj(X)

        attention_weight = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.dim)
        if attention_mask is not None:
            attention_weight = attention_weight.mask_fill(attention_mask == 0, float("-1e20"))
        
        attention_weight = torch.softmax(attention_weight, dim=-1)

        attention_weight = self.attention_drop(attention_weight)

        output = torch.matmul(attention_weight, V)# attention_weight @ V

        return output



if __name__ == '__main__':
    pass