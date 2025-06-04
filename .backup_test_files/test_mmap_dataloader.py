#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试内存映射DataLoader的功能
"""

import numpy as np
import os
import sys
import torch

# 添加src路径以导入模块
sys.path.append("src")
from utils.util import DataLoader, load_dataset


def create_test_data():
    """创建测试数据文件"""
    print("创建测试数据...")

    # 创建测试目录
    test_dir = "test_data"
    os.makedirs(test_dir, exist_ok=True)

    # 创建一些小的测试数据
    np.random.seed(42)

    # 训练数据: 100个样本
    train_x = np.random.randn(100, 10, 24).astype(np.float32)  # [samples, nodes, time_steps]
    train_y = np.random.randn(100, 10, 12).astype(np.float32)  # [samples, nodes, forecast_steps]

    # 验证数据: 30个样本
    val_x = np.random.randn(30, 10, 24).astype(np.float32)
    val_y = np.random.randn(30, 10, 12).astype(np.float32)

    # 测试数据: 20个样本
    test_x = np.random.randn(20, 10, 24).astype(np.float32)
    test_y = np.random.randn(20, 10, 12).astype(np.float32)

    # 保存为npz文件
    np.savez(os.path.join(test_dir, "train.npz"), x=train_x, y=train_y)
    np.savez(os.path.join(test_dir, "val.npz"), x=val_x, y=val_y)
    np.savez(os.path.join(test_dir, "test.npz"), x=test_x, y=test_y)

    print(f"测试数据已保存到 {test_dir}/")
    print(f"训练数据: x shape {train_x.shape}, y shape {train_y.shape}")
    print(f"验证数据: x shape {val_x.shape}, y shape {val_y.shape}")
    print(f"测试数据: x shape {test_x.shape}, y shape {test_y.shape}")

    return test_dir


class DummyNodeScaler:
    """虚拟的节点标准化器类"""

    def __init__(self):
        self.mean_ = np.zeros(10).astype(np.float32)
        self.std_ = np.ones(10).astype(np.float32)


def create_dummy_scaler():
    """创建一个虚拟的scaler文件用于测试"""
    import pickle

    scaler_params = {"tec_scaler": DummyNodeScaler(), "sw_scaler": None}

    scaler_path = "test_scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler_params, f)

    print(f"虚拟scaler已保存到 {scaler_path}")
    return scaler_path


def test_dataloader_basic():
    """测试DataLoader的基本功能"""
    print("\n=== 测试DataLoader基本功能 ===")

    # 创建小测试数据
    x_data = np.random.randn(17, 5, 10).astype(np.float32)  # 17个样本，批次大小8会有填充
    y_data = np.random.randn(17, 5, 6).astype(np.float32)

    data_dict = {"x": x_data, "y": y_data}

    # 测试不打乱
    loader = DataLoader(data_dict, batch_size=8, shuffle=False)
    print(f"原始大小: {loader.original_size_unpadded}")
    print(f"填充数量: {loader.num_padding}")
    print(f"逻辑大小: {loader.size}")
    print(f"批次数量: {loader.num_batch}")

    # 测试迭代
    iterator = loader.get_iterator()
    batch_count = 0
    total_samples = 0

    for i, (x_batch, y_batch) in enumerate(iterator):
        batch_count += 1
        batch_size = x_batch.shape[0]
        total_samples += batch_size
        print(f"批次 {i+1}: x shape {x_batch.shape}, y shape {y_batch.shape}")

        # 验证数据类型
        assert isinstance(x_batch, np.ndarray), "x_batch应该是numpy数组"
        assert isinstance(y_batch, np.ndarray), "y_batch应该是numpy数组"

    print(f"总批次数: {batch_count}")
    print(f"总样本数: {total_samples}")
    assert batch_count == loader.num_batch, f"批次数不匹配: {batch_count} vs {loader.num_batch}"
    assert total_samples == loader.size, f"样本数不匹配: {total_samples} vs {loader.size}"


def test_dataloader_shuffle():
    """测试DataLoader的打乱功能"""
    print("\n=== 测试DataLoader打乱功能 ===")

    # 创建可识别的测试数据
    x_data = np.arange(15 * 3 * 4).reshape(15, 3, 4).astype(np.float32)
    y_data = np.arange(15 * 3 * 2).reshape(15, 3, 2).astype(np.float32)

    data_dict = {"x": x_data, "y": y_data}

    # 测试打乱
    loader = DataLoader(data_dict, batch_size=5, shuffle=True)

    # 运行两次迭代，检查打乱是否生效
    first_run = []
    second_run = []

    # 第一次运行
    iterator1 = loader.get_iterator()
    for x_batch, y_batch in iterator1:
        first_run.append(x_batch[0, 0, 0])  # 记录第一个样本的第一个元素

    # 第二次运行
    iterator2 = loader.get_iterator()
    for x_batch, y_batch in iterator2:
        second_run.append(x_batch[0, 0, 0])

    print(f"第一次运行: {first_run}")
    print(f"第二次运行: {second_run}")

    # 打乱后应该不完全相同（除非非常巧合）
    if first_run != second_run:
        print("✓ 打乱功能正常工作")
    else:
        print("⚠ 打乱功能可能没有生效（或者巧合相同）")


def test_memory_mapping():
    """测试内存映射功能"""
    print("\n=== 测试内存映射功能 ===")

    # 创建测试数据并保存
    test_dir = create_test_data()
    scaler_path = create_dummy_scaler()

    try:
        # 使用内存映射加载数据
        dataset = load_dataset(dataset_dir=test_dir, scaler_path=scaler_path, batch_size=16, load_test=True, load_train_val=True)

        train_loader = dataset["train_loader"]
        val_loader = dataset["val_loader"]
        test_loader = dataset["test_loader"]

        print(f"训练集批次数: {train_loader.num_batch}")
        print(f"验证集批次数: {val_loader.num_batch}")
        print(f"测试集批次数: {test_loader.num_batch}")

        # 测试训练集迭代
        print("\n测试训练集迭代:")
        train_iter = train_loader.get_iterator()
        for i, (x_batch, y_batch) in enumerate(train_iter):
            print(f"训练批次 {i+1}: x shape {x_batch.shape}, y shape {y_batch.shape}")
            if i >= 2:  # 只显示前3个批次
                break

        # 测试验证集迭代
        print("\n测试验证集迭代:")
        val_iter = val_loader.get_iterator()
        for i, (x_batch, y_batch) in enumerate(val_iter):
            print(f"验证批次 {i+1}: x shape {x_batch.shape}, y shape {y_batch.shape}")
            if i >= 1:  # 只显示前2个批次
                break

        # 检查内存映射对象
        print(f"\n内存映射验证:")
        print(f"训练数据 x 类型: {type(train_loader.xs_orig)}")
        print(f"训练数据 y 类型: {type(train_loader.ys_orig)}")

        # 验证是否为内存映射数组
        if hasattr(train_loader.xs_orig, "filename"):
            print(f"✓ x数据确实是内存映射: {train_loader.xs_orig.filename}")
        if hasattr(train_loader.ys_orig, "filename"):
            print(f"✓ y数据确实是内存映射: {train_loader.ys_orig.filename}")

        print("✓ 内存映射功能测试完成")

    finally:
        # 清理测试文件
        import shutil

        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
            print(f"已清理测试目录: {test_dir}")
        if os.path.exists(scaler_path):
            os.remove(scaler_path)
            print(f"已清理测试scaler: {scaler_path}")


def main():
    """主函数"""
    print("开始测试内存映射DataLoader...")

    try:
        test_dataloader_basic()
        test_dataloader_shuffle()
        test_memory_mapping()
        print("\n🎉 所有测试通过！")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
