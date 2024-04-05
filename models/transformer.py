import math

import torch
import torch.nn as nn

from .normalization import LayerNorm


class PositionEncoding(nn.Module):
    """位置编码
    用法：直接和 embedding vector 相加，(max_len, dimension)
    input = torch.rand((2, 5, 10), dtype=torch.float32)
    encoder = PositionEncoding(10)
    output = encoder(input)
    print(f"output: {output.shape}")
    """

    def __init__(self, d_model: int, max_len=1000, drop_out=0.1):
        super(PositionEncoding, self).__init__()
        self.dropout = nn.Dropout(drop_out)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        exp = torch.exp(torch.arange(0, d_model, 2) / d_model * math.log(10000.0) * -1)
        pe[:, 0::2] = torch.sin(pos * exp)
        pe[:, 1::2] = torch.cos(pos * exp)
        # buffer不会被 optimizer 更新
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_head, d_ff=2048, drop_out=0.1):
        super(TransformerEncoder, self).__init__()

        self.attention = MultiHeadAttention(d_model, n_head, drop_out)
        self.dropout = nn.Dropout(drop_out)
        self.ffw = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.layer_norm1 = LayerNorm(d_model)
        self.layer_norm2 = LayerNorm(d_model)

    def forward(self, x, mask):
        x = self.layer_norm1(x + self.dropout(self.attention(x, x, x, mask)))
        x = self.layer_norm2(x + self.dropout(self.ffw(x)))
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, d_model, n_head, d_ff=2048, drop_out=0.1):
        super(TransformerDecoder, self).__init__()
        self.masked_attention = MultiHeadAttention(d_model, n_head, drop_out)
        self.encoder_attention = MultiHeadAttention(d_model, n_head, drop_out)
        self.ffw = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(drop_out)
        self.layer_norm1 = LayerNorm(d_model)
        self.layer_norm2 = LayerNorm(d_model)
        self.layer_norm3 = LayerNorm(d_model)

    def forward(self, x, encoder_out, input_mask, ans_mask):
        x = self.layer_norm1(x + self.dropout(self.masked_attention(x, x, x, input_mask)))
        x = self.layer_norm2(x + self.dropout(self.encoder_attention(x, encoder_out, encoder_out, ans_mask)))
        x = self.layer_norm3(x + self.dropout(self.ffw(x)))
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        b = query.size(0)
        # (batch, seq_len, d_model) -> (batch, seq_len, n_heads, head_dim) -> (batch, n_heads, seq_len, head_dim)
        q = self.q(query).view(b, -1, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(key).view(b, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(value).view(b, -1, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)
        score = self.dropout(torch.softmax(score, dim=-1))
        out = torch.matmul(score, v)
        out = out.transpose(1, 2).contiguous().view(b, -1, self.d_model)
        out = self.fc(out)
        return out
