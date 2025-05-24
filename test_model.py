#!/usr/bin/env python3
"""
简单的模型测试脚本
"""

import torch
import sys

sys.path.append("src")


def test_model():
    print("Testing tecGPT model creation and forward pass...")

    try:
        from models import get_model

        # 测试模型获取
        model_class = get_model("tecGPT")
        print(f"✓ Model class retrieved: {model_class.__name__}")

        # 创建小型测试模型，使用与GPT-2兼容的参数
        test_args = {
            "input_len": 12,
            "output_len": 12,
            "num_nodes": 100,  # 使用较小的节点数进行测试
            "n_lat": 10,
            "n_lon": 10,
            "d_embed": 32,
            "d_llm": 768,  # 必须与GPT-2的n_embd一致
            "llm_model_local_path": "/home/panxiong/tecGPT-forecasting/src/models/tecGPT/gpt2",
            "llm_layers_to_use": 2,
            "U_unfrozen_mha": 1,
            "device": "cpu",
        }

        model = model_class(**test_args)
        print(f"✓ Model created successfully with {model.param_num(trainable_only=True):,} trainable parameters")

        # 测试前向传播
        batch_size = 2
        test_input = torch.randn(batch_size, 12, 100, 12)  # [B, P, N, C]
        with torch.no_grad():
            output = model(test_input)
        print(f"✓ Forward pass successful: input {test_input.shape} -> output {output.shape}")

        print("All tests passed! 🎉")
        return True

    except Exception as e:
        print(f"✗ Model test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_model()
    sys.exit(0 if success else 1)
