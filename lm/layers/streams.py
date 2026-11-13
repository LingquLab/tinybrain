"""多条残差流的入口和出口，都用注意力来做。

入口：一个位置上有几份「来源」（词向量、二元记忆、三元记忆）。每条流有一个自己的查询向量，
      对这几份来源打分、加权求和，得到这条流的初值。不同的流可以偏好不同的来源。
出口：一个可学的查询向量对 n 条流打分、加权求和，合成一份交给输出层。
"""
import math

import torch
import torch.nn as nn

from .norm import RMSNorm


class StreamEntry(nn.Module):
    def __init__(self, dim: int, n_streams: int, key_dim: int):
        super().__init__()
        self.key = nn.Linear(dim, key_dim, bias=False)
        self.key_norm = RMSNorm(key_dim)
        self.query = nn.Parameter(torch.randn(n_streams, key_dim))  # 随机起步，各条流一开始就不一样

    def forward(self, sources: torch.Tensor) -> torch.Tensor:
        """sources: (B, T, 来源数, d) → (B, T, n, d)"""
        keys = self.key_norm(self.key(sources))
        weights = torch.softmax(torch.einsum("btsk,mk->btms", keys, self.query.to(keys.dtype)) / math.sqrt(keys.size(-1)), dim=-1)
        return torch.einsum("btms,btsd->btmd", weights, sources)


class StreamExit(nn.Module):
    def __init__(self, dim: int, key_dim: int):
        super().__init__()
        self.key = nn.Linear(dim, key_dim, bias=False)
        self.in_norm, self.key_norm = RMSNorm(dim), RMSNorm(key_dim)
        self.query = nn.Parameter(torch.zeros(key_dim))  # 从零起步：一开始是各条流的平均

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, n, d) → (B, T, d)"""
        keys = self.key_norm(self.key(self.in_norm(x)))
        weights = torch.softmax(keys @ self.query.to(keys.dtype) / math.sqrt(keys.size(-1)), dim=-1)
        return torch.einsum("btn,btnd->btd", weights, x)
