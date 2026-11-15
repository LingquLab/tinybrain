import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from liger_kernel.ops.relu_squared import LigerReLUSquaredFunction
except ImportError:
    LigerReLUSquaredFunction = None


class FeedForward(nn.Module):
    """前馈层，激活函数是 ReLU 的平方。输出投影从零开始，所以刚初始化时这一层什么都不加。"""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc = nn.Linear(dim, hidden, bias=False)
        self.proj = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.fc(x)
        if torch.compiler.is_compiling():
            # 让 Inductor 联合搜索 GEMM 与激活的融合实现，包括反向 epilogue。
            hidden = F.relu(hidden).square()
        elif hidden.is_cuda and LigerReLUSquaredFunction is not None:
            hidden = LigerReLUSquaredFunction.apply(hidden)
        else:
            hidden = F.relu(hidden).square()
        return self.proj(hidden)
