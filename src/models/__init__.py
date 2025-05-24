"""
模型模块
"""

# tecGPT模型
from .tecGPT import ST_LLM, PFA_GPT2, TecHistoryEmbedding, TimeFeatureEmbedding, SpatialEmbedding, SpaceWeatherEmbedding, FusionLayer

# 模型注册表，用于动态选择模型
MODEL_REGISTRY = {"tecGPT": ST_LLM, "ST_LLM": ST_LLM}  # 别名


def get_model(model_name):
    """根据模型名称获取模型类"""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not found. Available models: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_name]


__all__ = [
    "ST_LLM",
    "PFA_GPT2",
    "TecHistoryEmbedding",
    "TimeFeatureEmbedding",
    "SpatialEmbedding",
    "SpaceWeatherEmbedding",
    "FusionLayer",
    "MODEL_REGISTRY",
    "get_model",
]
