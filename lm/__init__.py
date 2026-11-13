"""一个小型中文语言模型：多流残差（mHC）、Engram 记忆、滑动窗口注意力，大号带混合专家。"""
from .config import PRESETS, ModelConfig, preset
from .model import LanguageModel

__all__ = ["PRESETS", "ModelConfig", "LanguageModel", "preset"]
