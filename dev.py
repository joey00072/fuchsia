import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Separate projections for query, key, value
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None, is_causal=False):
        B, T, C = x.shape  # Batch size, Sequence length, Embedding dim

        # Linear projections
        q = self.q_proj(x)  # (B, T, C)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, T, head_dim)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # scaled_dot_product_attention expects (B, num_heads, T, head_dim)
        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=is_causal
        )  # (B, num_heads, T, head_dim)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, embed_dim)

        return self.out_proj(attn_output)
    
    
    
class MergedAttention(nn.Module):
    def __init__(self, embed_dim, num_heads , wqk, wqo):
        super().__init__()
        
        
        self.wqk = None
        self.wqo = None
    
    def forward(self, x):
        pass
    
    
    @classmethod
    def from_mha(cls, mha: MultiHeadAttention):
        return cls(mha.embed_dim, mha.num_heads, mha.wqk, mha.wqo)
    
    
    
    
# Example usage:
embed_dim = 512
num_heads = 8
mha = MultiHeadAttention(embed_dim, num_heads)
x = torch.rand(32, 10, embed_dim)  # (batch_size, seq_length, embed_dim)
output = mha(x)
print(output.shape)  # (32, 10, 512)
