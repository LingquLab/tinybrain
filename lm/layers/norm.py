import torch.nn as nn


class RMSNorm(nn.RMSNorm):
    """RMSNorm：把最后一维缩放到均方根为 1，再逐维乘一个可学的系数（初始为 1）。

    系数只有 dim 个，推理时可以并进后面的线性层，不花额外的算力。
    """

    def __init__(self, dim: int):
        super().__init__(dim)
