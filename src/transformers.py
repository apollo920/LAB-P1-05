import math
import torch
import torch.nn as nn
from src.blocks import EncoderBlock, DecoderBlock


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8,
                 num_layers=6, d_ff=2048, max_len=512, dropout=0.1):
        super().__init__()

        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.embed_scale = math.sqrt(d_model)

        self.src_pe = PositionalEncoding(d_model, max_len, dropout)
        self.tgt_pe = PositionalEncoding(d_model, max_len, dropout)

        self.encoder_layers = nn.ModuleList(
            [EncoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask=None):
        x = self.src_embedding(src) * self.embed_scale
        x = self.src_pe(x)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, Z, tgt_mask=None, src_mask=None):
        y = self.tgt_embedding(tgt) * self.embed_scale
        y = self.tgt_pe(y)
        for layer in self.decoder_layers:
            y = layer(y, Z, tgt_mask, src_mask)
        return self.output_projection(y)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        Z = self.encode(src, src_mask)
        return self.decode(tgt, Z, tgt_mask, src_mask)