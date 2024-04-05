import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_len, dropout):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = self.get_positional_encoding(max_len, d_model)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, src, trg):
        src_mask = self.generate_square_subsequent_mask(src.size(1))
        trg_mask = self.generate_square_subsequent_mask(trg.size(1))
        src = self.embedding(src) + self.positional_encoding[:src.size(1), :]
        trg = self.embedding(trg) + self.positional_encoding[:trg.size(1), :]
        for layer in self.encoder_layers:
            src = layer(src, src_mask)
        for layer in self.decoder_layers:
            trg = layer(trg, src, trg_mask, src_mask)
        output = self.output_layer(trg)
        return output

    def generate_square_subsequent_mask(self, size):
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def get_positional_encoding(self, max_len, d_model):
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        position_encoding = torch.zeros(max_len, d_model)
        position_encoding[:, 0::2] = torch.sin(position * div_term)
        position_encoding[:, 1::2] = torch.cos(position * div_term)
        return position_encoding


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        residual = x
        x = self.layer_norm1(x + self.dropout(self.self_attention(x, x, x, mask)))
        x = self.layer_norm2(x + self.dropout(self.feed_forward(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.encoder_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_output, trg_mask, src_mask):
        residual = x
        x = self.layer_norm1(x + self.dropout(self.self_attention(x, x, x, trg_mask)))
        x = self.layer_norm2(x + self.dropout(self.encoder_attention(x, enc_output, enc_output, src_mask)))
        x = self.layer_norm3(x + self.dropout(self.feed_forward(x)))
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        Q = self.wq(query)
        K = self.wk(key)
        V = self.wv(value)
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-inf'))
        attention = F.softmax(energy, dim=-1)
        x = torch.matmul(self.dropout(attention), V)
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)
        x = self.fc(x)
        return x
