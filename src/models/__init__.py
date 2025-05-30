"""
模型模块
"""

# tecGPT模型
from .tecGPT import tecGPT

# 模型注册表，用于动态选择模型
MODEL_REGISTRY = {
    "tecGPT": tecGPT,
}


def get_model_class(model_name: str):
    """根据模型名称获取模型类"""
    model_cls = MODEL_REGISTRY.get(model_name)
    if model_cls is None:
        raise ValueError(f"Model {model_name} not found. Available models: {list(MODEL_REGISTRY.keys())}")
    return model_cls


__all__ = [
    "tecGPT",
    "MODEL_REGISTRY",
    "get_model_class",
]
