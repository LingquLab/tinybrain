"""模型配置：所有尺寸和开关都在这一个数据类里，三档预设也在这里。"""
import dataclasses
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    dim: int = 768
    n_layers: int = 10
    n_heads: int = 12
    n_kv_heads: int = 3            # 分组查询注意力：几个查询头共用一组键值头
    max_seq_len: int = 8192
    ffn_mult: float = 4.0          # 前馈层中间宽度是 dim 的几倍
    rope_base: float = 10000.0

    # 注意力的范围：每 long_every 层里最后一层看全文，其余只看最近 window 个词（含自己）。最后一层总是看全文。
    window: int = 1024
    long_every: int = 4

    # 多流残差（mHC）：残差流的条数，以及把混合矩阵化成双随机矩阵的迭代次数
    n_streams: int = 4
    sinkhorn_iters: int = 20
    stream_key_dim: int = 64       # 流的入口、出口做注意力时键的宽度

    # Engram：按前面 n-1 个词加当前词查表的记忆。engram_orders 为空就整个关掉。
    engram_orders: tuple = (2, 3)
    engram_heads: int = 8          # 每个阶数用几个不同的哈希
    engram_head_dim: int = 32      # 每个哈希取回的向量宽度
    engram_rows: int = 16384       # 每个哈希的表大约多少行（实际取不超过它的质数）
    engram_layers: tuple = (1,)    # 在第几层（从 0 数）前面再做一次带门控的记忆注入

    # 混合专家（MoE）：moe_experts 为 0 表示普通前馈层
    moe_experts: int = 0
    moe_top_k: int = 3
    moe_hidden_mult: float = 1.0   # 每个专家的中间宽度是 dim 的几倍
    moe_shared_mult: float = 1.0   # 共享专家的中间宽度是 dim 的几倍
    moe_latent_dim: int = 0        # 大于 0 时专家在压缩后的空间里工作（LatentMoE）
    moe_balance_rate: float = 1e-3  # 负载均衡偏置每步调多少
    moe_balance_loss: float = 1e-4  # 负载均衡损失的权重

    def __post_init__(self):
        object.__setattr__(self, "engram_orders", tuple(self.engram_orders))
        object.__setattr__(self, "engram_layers", tuple(self.engram_layers) if self.engram_orders else ())
        problems = []
        if self.dim % self.n_heads:
            problems.append(f"dim={self.dim} 不能被 n_heads={self.n_heads} 整除")
        elif (self.dim // self.n_heads) % 2:
            problems.append(f"每个头的宽度 {self.dim // self.n_heads} 必须是偶数（旋转位置编码要成对）")
        if self.n_heads % self.n_kv_heads:
            problems.append(f"n_heads={self.n_heads} 不能被 n_kv_heads={self.n_kv_heads} 整除")
        if not 1 <= self.window <= self.max_seq_len:
            problems.append(f"window={self.window} 要在 1 和 max_seq_len={self.max_seq_len} 之间")
        if self.n_streams < 1 or self.long_every < 1 or self.n_layers < 1:
            problems.append("n_streams、long_every、n_layers 都至少是 1")
        if any(n < 2 for n in self.engram_orders):
            problems.append(f"engram_orders={self.engram_orders} 里的阶数至少是 2")
        if any(not 0 <= i < self.n_layers for i in self.engram_layers):
            problems.append(f"engram_layers={self.engram_layers} 超出了层数 {self.n_layers}")
        if self.moe_experts and not 1 <= self.moe_top_k <= self.moe_experts:
            problems.append(f"moe_top_k={self.moe_top_k} 要在 1 和 moe_experts={self.moe_experts} 之间")
        if problems:
            raise ValueError("模型配置有问题：" + "；".join(problems))

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    @property
    def engram_dim(self) -> int:
        """一个位置上取回的全部记忆向量拼起来的宽度。"""
        return len(self.engram_orders) * self.engram_heads * self.engram_head_dim

    def window_of(self, layer: int) -> int | None:
        """第 layer 层的注意力范围；None 表示看全文。"""
        long = layer % self.long_every == self.long_every - 1 or layer == self.n_layers - 1
        return None if long or self.window >= self.max_seq_len else self.window

    def trunk_params(self) -> int:
        """主干的参数量：除了词向量、输出层、记忆表以外的全部。不用真的建模型就能算。"""
        d, n, mem = self.dim, self.n_streams, self.engram_dim
        layer = 2 * d * d + 2 * d * self.n_kv_heads * self.head_dim  # 注意力的 q、输出投影，加上 k、v
        if self.moe_experts:
            width = self.moe_latent_dim or d
            layer += 2 * d * int(self.moe_shared_mult * d) + width * self.moe_experts + 2 * self.moe_experts * int(self.moe_hidden_mult * d) * width
            layer += 2 * d * width if self.moe_latent_dim else 0
        else:
            layer += 2 * d * int(self.ffn_mult * d)
        total = self.n_layers * layer + 2 * self.n_layers * (2 * n + n * n)       # 各层，加上多流残差的系数
        total += 2 * d * self.stream_key_dim + (n + 1) * self.stream_key_dim      # 流的入口和出口
        total += len(self.engram_orders) * self.engram_heads * self.engram_head_dim * d  # 每个阶数的记忆投影成一份来源
        total += len(self.engram_layers) * (mem * d + mem * n * d)                # 记忆注入的值投影和各条流的键投影
        # 归一化的系数：每层两个子层各一组、查询和键各一组；词向量、每份记忆来源、出口前、输出层前各一组；入口出口的键各一组；
        # 记忆表后一组；每处注入两组
        total += self.n_layers * (2 * d + 2 * self.head_dim) + (3 + len(self.engram_orders)) * d + 2 * self.stream_key_dim
        total += (self.engram_heads * self.engram_head_dim if self.engram_orders else 0) + len(self.engram_layers) * 2 * d
        return total

    def with_engram_share(self, share: float) -> "ModelConfig":
        """调整记忆表的行数，让记忆表占「主干 + 记忆表」的比例等于 share（词向量和输出层不算在内）。"""
        if not self.engram_orders:
            return self
        table = share / (1 - share) * self.trunk_params()
        rows = round(table / (len(self.engram_orders) * self.engram_heads * self.engram_head_dim))
        return dataclasses.replace(self, engram_rows=rows)

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), ensure_ascii=False, indent=1)

    @classmethod
    def from_json(cls, text: str) -> "ModelConfig":
        return cls(**json.loads(text))


ENGRAM_SHARE = 0.45  # 记忆表占「主干 + 记忆表」的比例

# 三档预设，各有各的主干和记忆表。small 的主干约 64M；medium、large 的数字是我提的，还没定。
# 记忆表：每个哈希取回的向量宽度写在这里，行数由 preset() 按 ENGRAM_SHARE 算出来。
# large 比 medium 更宽更深，前馈层是 24 个四分之一宽的专家、每个词选 8 个，外加一个整宽的共享专家。
PRESETS = {
    "small": dict(dim=768, n_layers=10, n_heads=12, n_kv_heads=3, engram_head_dim=32, engram_layers=(1,)),
    "medium": dict(dim=1152, n_layers=16, n_heads=16, n_kv_heads=4, engram_head_dim=48, engram_layers=(1, 8)),
    "large": dict(dim=1280, n_layers=20, n_heads=20, n_kv_heads=5, engram_head_dim=64, engram_layers=(1, 10),
                  moe_experts=24, moe_top_k=8, moe_hidden_mult=0.25),
}


def preset(name: str, vocab_size: int, engram_share: float = ENGRAM_SHARE, **overrides) -> ModelConfig:
    """按预设建配置。没有明说 engram_rows 时，记忆表的行数按 engram_share 定。"""
    if name not in PRESETS:
        raise ValueError(f"没有 {name!r} 这个预设，可选：{', '.join(PRESETS)}")
    cfg = ModelConfig(vocab_size=vocab_size, **{**PRESETS[name], **overrides})
    return cfg if "engram_rows" in overrides else cfg.with_engram_share(engram_share)
