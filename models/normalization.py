import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, dimension, eps=1e-5):
        # input: (batch_size, seq_len, dimension)
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(dimension))  # 初始化缩放参数
        self.beta = nn.Parameter(torch.zeros(dimension))  # 初始化偏移参数
        self.eps = eps

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        x_normalized = (x - mean) / (std + self.eps)
        y = self.gamma * x_normalized + self.beta
        return y