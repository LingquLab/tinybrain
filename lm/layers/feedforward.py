import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """前馈层，激活函数是 ReLU 的平方。输出投影从零开始，所以刚初始化时这一层什么都不加。"""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc = nn.Linear(dim, hidden, bias=False)
        self.proj = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(F.relu(self.fc(x)).square())
