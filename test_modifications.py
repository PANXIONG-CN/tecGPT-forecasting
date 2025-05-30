#!/usr/bin/env python3
"""
测试脚本：验证所有修改是否正确
"""

import sys
import os
import numpy as np
import torch

# 添加src目录到路径
sys.path.insert(0, "src")


def test_dataloader_modifications():
    """测试DataLoader的修改"""
    print("=== 测试DataLoader修改 ===")

    from utils import util

    # 创建虚拟数据
    x_dummy = np.random.randn(100, 12, 2911, 12)
    y_dummy = np.random.randn(100, 12, 2911, 1)

    # 测试训练集DataLoader (应该shuffle)
    train_loader = util.DataLoader(x_dummy, y_dummy, batch_size=16, shuffle=True)
    print(f"✓ 训练集DataLoader创建成功，批次数: {train_loader.num_batch}")
    print(f"✓ 训练集should_shuffle: {train_loader.should_shuffle}")

    # 测试验证集DataLoader (不应该shuffle)
    val_loader = util.DataLoader(x_dummy, y_dummy, batch_size=16, shuffle=False)
    print(f"✓ 验证集DataLoader创建成功，批次数: {val_loader.num_batch}")
    print(f"✓ 验证集should_shuffle: {val_loader.should_shuffle}")

    # 测试get_iterator方法
    train_iter = train_loader.get_iterator()
    val_iter = val_loader.get_iterator()
    print("✓ get_iterator方法工作正常")

    print("DataLoader修改测试通过！\n")


def test_loss_functions():
    """测试损失函数"""
    print("=== 测试损失函数 ===")

    from utils import util

    # 创建虚拟数据
    pred = torch.randn(10, 2911, 12)
    true = torch.randn(10, 2911, 12)

    # 测试各种损失函数
    mae_loss = util.MAE_torch(pred, true)
    rmse_loss = util.RMSE_torch(pred, true)
    mape_loss = util.MAPE_torch(pred, true)
    wmape_loss = util.WMAPE_torch(pred, true)

    print(f"✓ MAE损失: {mae_loss.item():.4f}")
    print(f"✓ RMSE损失: {rmse_loss.item():.4f}")
    print(f"✓ MAPE损失: {mape_loss.item():.4f}")
    print(f"✓ WMAPE损失: {wmape_loss.item():.4f}")

    # 测试metric函数
    metrics = util.metric(pred, true)
    print(f"✓ metric函数返回: MAE={metrics[0]:.4f}, MAPE={metrics[1]:.4f}, RMSE={metrics[2]:.4f}, WMAPE={metrics[3]:.4f}")

    print("损失函数测试通过！\n")


def test_pfa_gpt2_import():
    """测试PFA_GPT2导入"""
    print("=== 测试PFA_GPT2导入 ===")

    try:
        from models.tecGPT.pfa_llm import PFA_GPT2, PatchedGPT2Model

        print("✓ PFA_GPT2和PatchedGPT2Model导入成功")

        # 检查PFA_GPT2类是否有正确的方法
        assert hasattr(PFA_GPT2, "__init__"), "PFA_GPT2缺少__init__方法"
        assert hasattr(PFA_GPT2, "forward"), "PFA_GPT2缺少forward方法"
        print("✓ PFA_GPT2类结构正确")

    except Exception as e:
        print(f"✗ PFA_GPT2导入失败: {e}")
        return False

    print("PFA_GPT2导入测试通过！\n")
    return True


def test_model_imports():
    """测试模型导入"""
    print("=== 测试模型导入 ===")

    try:
        from models.tecGPT.tec_gpt import ST_LLM
        from models.tecGPT.embeddings import TecHistoryEmbedding, TimeFeatureEmbedding, SpatialEmbedding, SpaceWeatherEmbedding, FusionLayer
        from models import get_model, MODEL_REGISTRY

        print("✓ 所有模型类导入成功")
        print(f"✓ 模型注册表: {list(MODEL_REGISTRY.keys())}")

        # 测试get_model函数
        model_class = get_model("tecGPT")
        assert model_class == ST_LLM, "get_model返回的类不正确"
        print("✓ get_model函数工作正常")

    except Exception as e:
        print(f"✗ 模型导入失败: {e}")
        return False

    print("模型导入测试通过！\n")
    return True


def test_training_script_syntax():
    """测试训练脚本语法"""
    print("=== 测试训练脚本语法 ===")

    try:
        import py_compile

        py_compile.compile("src/tec_train.py", doraise=True)
        print("✓ 训练脚本语法正确")

        # 测试参数解析
        from src.tec_train import parse_args

        print("✓ 参数解析函数导入成功")

    except Exception as e:
        print(f"✗ 训练脚本语法错误: {e}")
        return False

    print("训练脚本语法测试通过！\n")
    return True


def main():
    """主测试函数"""
    print("开始测试所有修改...\n")

    tests = [
        test_dataloader_modifications,
        test_loss_functions,
        test_pfa_gpt2_import,
        test_model_imports,
        test_training_script_syntax,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            result = test()
            if result is not False:
                passed += 1
        except Exception as e:
            print(f"✗ 测试失败: {e}\n")

    print("=" * 50)
    print(f"测试结果: {passed}/{total} 通过")

    if passed == total:
        print("🎉 所有测试通过！修改成功！")
        return True
    else:
        print("❌ 部分测试失败，请检查修改")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
