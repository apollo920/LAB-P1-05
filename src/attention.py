import torch
import torch.nn as nn
import math


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)

    scores = torch.matmul(Q, K.transpose(-2, -1))
    scores = scores / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)

    return output, attn_weights


def make_causal_mask(seq_len, device='cpu'):
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0).unsqueeze(0)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def split_heads(self, x):
        batch, seq_len, _ = x.size()
        x = x.view(batch, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def concat_heads(self, x):
        batch, _, seq_len, _ = x.size()
        x = x.transpose(1, 2).contiguous()
        return x.view(batch, seq_len, self.d_model)

    def forward(self, Q_in, K_in, V_in, mask=None):
        Q = self.split_heads(self.W_q(Q_in))
        K = self.split_heads(self.W_k(K_in))
        V = self.split_heads(self.W_v(V_in))

        attn_out, _ = scaled_dot_product_attention(Q, K, V, mask)

        concat = self.concat_heads(attn_out)
        return self.W_o(concat)