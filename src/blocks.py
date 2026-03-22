import torch.nn as nn
from src.attention import MultiHeadAttention
from src.sublayers import FeedForwardNetwork, AddAndNorm


class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.add_norm1 = AddAndNorm(d_model, dropout)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)
        self.add_norm2 = AddAndNorm(d_model, dropout)

    def forward(self, x, src_mask=None):
        attn_out = self.self_attn(x, x, x, mask=src_mask)
        x = self.add_norm1(x, attn_out)
        ffn_out = self.ffn(x)
        x = self.add_norm2(x, ffn_out)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.masked_self_attn = MultiHeadAttention(d_model, num_heads)
        self.add_norm1 = AddAndNorm(d_model, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.add_norm2 = AddAndNorm(d_model, dropout)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)
        self.add_norm3 = AddAndNorm(d_model, dropout)

    def forward(self, y, Z, tgt_mask=None, src_mask=None):
        masked_attn_out = self.masked_self_attn(y, y, y, mask=tgt_mask)
        y = self.add_norm1(y, masked_attn_out)
        cross_attn_out = self.cross_attn(y, Z, Z, mask=src_mask)
        y = self.add_norm2(y, cross_attn_out)
        ffn_out = self.ffn(y)
        y = self.add_norm3(y, ffn_out)
        return y