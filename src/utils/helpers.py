import torch
import numpy as np
import random
import os
import logging

logger = logging.getLogger(__name__)

def seed_everything(seed: int) -> None:
    """
    设置随机种子以确保实验的可复现性。
    Args:
        seed (int): 要设置的种子值。
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    # Potentially set CUDNN options for full determinism, but this can impact performance.
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False # If deterministic is True, benchmark should be False
    logger.info(f"Global seed set to {seed}")

# 你可以在这里添加其他通用的辅助函数，例如：
# - 函数用于获取设备 (cuda or cpu)
# - 函数用于配置日志记录 (如果 Hydra 的默认配置不够用)
# - 函数用于保存和加载检查点 (如果不用 PyTorch Lightning 或类似的框架)

# Example helper for device
def get_device(device_config: str = "auto") -> torch.device:
    """
    根据配置获取torch设备。
    Args:
        device_config (str): "auto", "cuda", "cpu", "cuda:0" etc.
    Returns:
        torch.device: The selected torch device.
    """
    if device_config == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif "cuda" in device_config and not torch.cuda.is_available():
        logger.warning(f"CUDA specified ({device_config}) but not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(device_config)
    logger.info(f"Using device: {device}")
    return device

if __name__ == '__main__':
    print("Testing seed_everything...")
    seed_everything(42)
    # Verify by generating some random numbers
    print(f"Python random: {random.random()}")
    print(f"Numpy random: {np.random.rand(2)}")
    print(f"Torch random CPU: {torch.rand(2)}")
    if torch.cuda.is_available():
        print(f"Torch random CUDA: {torch.cuda.FloatTensor(2).normal_()}")
    
    print("\nTesting get_device...")
    device_auto = get_device("auto")
    print(f"Device (auto): {device_auto}")
    device_cuda = get_device("cuda") # Will fallback if no CUDA
    print(f"Device (cuda): {device_cuda}")
    device_cpu = get_device("cpu")
    print(f"Device (cpu): {device_cpu}")
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        device_cuda1 = get_device("cuda:1")
        print(f"Device (cuda:1): {device_cuda1}") 