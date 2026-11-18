"""推理：分词器、对话格式、生成，以及对话和服务的命令行。训练相关的在 training，模型的定义在 lm。"""
from .template import ChatFormat
from .tokenizer import DEFAULT_MODEL, Tokenizer

__all__ = ["ChatFormat", "DEFAULT_MODEL", "Tokenizer", "generate", "sample_next"]


def __getattr__(name):
    if name in ('generate', 'sample_next'):
        from .generate import generate, sample_next
        globals().update(generate=generate, sample_next=sample_next)
        return globals()[name]
    raise AttributeError(name)
