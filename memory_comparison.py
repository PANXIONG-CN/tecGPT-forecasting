#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
内存使用对比：传统DataLoader vs 内存映射DataLoader
"""

import numpy as np
import os
import sys
import psutil
import time
import gc

# 添加src路径
sys.path.append("src")


def get_memory_usage():
    """获取当前进程的内存使用量（MB）"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def create_large_test_data(samples=10000, nodes=512, time_steps=24, forecast_steps=12):
    """创建一个较大的测试数据集"""
    print(f"创建大型测试数据: {samples} 样本, {nodes} 节点, {time_steps} 时间步")

    # 创建测试目录
    test_dir = "large_test_data"
    os.makedirs(test_dir, exist_ok=True)

    # 创建训练数据
    train_x = np.random.randn(samples, nodes, time_steps).astype(np.float32)
    train_y = np.random.randn(samples, nodes, forecast_steps).astype(np.float32)

    # 保存数据
    train_path = os.path.join(test_dir, "train.npz")
    np.savez(train_path, x=train_x, y=train_y)

    # 计算文件大小
    file_size = os.path.getsize(train_path) / 1024 / 1024  # MB
    print(f"数据文件大小: {file_size:.2f} MB")
    print(f"训练数据形状: x={train_x.shape}, y={train_y.shape}")

    return test_dir, train_path


def test_traditional_loading(data_path):
    """测试传统的全量加载方式"""
    print("\n=== 测试传统全量加载 ===")

    # 记录初始内存
    initial_memory = get_memory_usage()
    print(f"初始内存使用: {initial_memory:.2f} MB")

    start_time = time.time()

    # 传统方式：完全加载到内存
    data = np.load(data_path)
    train_x = np.array(data["x"])  # 完全复制到内存
    train_y = np.array(data["y"])  # 完全复制到内存

    load_time = time.time() - start_time
    after_load_memory = get_memory_usage()

    print(f"数据加载时间: {load_time:.3f} 秒")
    print(f"加载后内存使用: {after_load_memory:.2f} MB")
    print(f"内存增长: {after_load_memory - initial_memory:.2f} MB")

    # 模拟使用数据（创建DataLoader等操作）
    data_dict = {"x": train_x, "y": train_y}

    # 手动清理
    del train_x, train_y, data, data_dict
    gc.collect()

    return {"load_time": load_time, "memory_increase": after_load_memory - initial_memory, "peak_memory": after_load_memory}


def test_mmap_loading(data_path):
    """测试内存映射加载方式"""
    print("\n=== 测试内存映射加载 ===")

    # 记录初始内存
    initial_memory = get_memory_usage()
    print(f"初始内存使用: {initial_memory:.2f} MB")

    start_time = time.time()

    # 内存映射方式：不完全加载到内存
    data = np.load(data_path, mmap_mode="r")
    train_x = data["x"]  # 内存映射对象，不复制
    train_y = data["y"]  # 内存映射对象，不复制

    load_time = time.time() - start_time
    after_load_memory = get_memory_usage()

    print(f"数据加载时间: {load_time:.3f} 秒")
    print(f"加载后内存使用: {after_load_memory:.2f} MB")
    print(f"内存增长: {after_load_memory - initial_memory:.2f} MB")

    # 测试访问少量数据（模拟批次访问）
    print("\n测试批次访问...")
    batch_start_time = time.time()

    # 访问前几个批次
    batch_size = 32
    for i in range(5):  # 访问5个批次
        start_idx = i * batch_size
        end_idx = start_idx + batch_size

        # 将内存映射数据转换为实际的numpy数组（模拟DataLoader行为）
        batch_x = np.array(train_x[start_idx:end_idx])
        batch_y = np.array(train_y[start_idx:end_idx])

        # 模拟使用数据
        _ = batch_x.shape, batch_y.shape

        del batch_x, batch_y

    batch_time = time.time() - batch_start_time
    after_batch_memory = get_memory_usage()

    print(f"批次访问时间: {batch_time:.3f} 秒")
    print(f"批次访问后内存: {after_batch_memory:.2f} MB")

    # 手动清理
    del train_x, train_y, data
    gc.collect()

    return {
        "load_time": load_time,
        "memory_increase": after_load_memory - initial_memory,
        "peak_memory": max(after_load_memory, after_batch_memory),
        "batch_time": batch_time,
    }


def test_dataloader_comparison(data_path):
    """对比两种DataLoader的性能"""
    print("\n=== 对比DataLoader性能 ===")

    from utils.util import DataLoader

    # 1. 测试传统DataLoader（需要先创建旧版本）
    print("\n传统DataLoader测试:")
    initial_memory = get_memory_usage()

    # 完全加载数据
    data = np.load(data_path)
    train_x_full = np.array(data["x"])
    train_y_full = np.array(data["y"])

    traditional_loader = DataLoader({"x": train_x_full, "y": train_y_full}, batch_size=32, shuffle=False)
    traditional_memory = get_memory_usage()

    print(f"传统DataLoader内存使用: {traditional_memory - initial_memory:.2f} MB")

    # 清理
    del train_x_full, train_y_full, traditional_loader, data
    gc.collect()

    # 2. 测试内存映射DataLoader
    print("\n内存映射DataLoader测试:")
    initial_memory = get_memory_usage()

    # 内存映射加载
    data_mmap = np.load(data_path, mmap_mode="r")
    train_x_mmap = data_mmap["x"]
    train_y_mmap = data_mmap["y"]

    mmap_loader = DataLoader({"x": train_x_mmap, "y": train_y_mmap}, batch_size=32, shuffle=False)
    mmap_memory = get_memory_usage()

    print(f"内存映射DataLoader内存使用: {mmap_memory - initial_memory:.2f} MB")

    # 测试迭代性能
    print("\n测试迭代性能...")

    # 内存映射版本的迭代
    start_time = time.time()
    iterator = mmap_loader.get_iterator()
    batch_count = 0
    for x_batch, y_batch in iterator:
        batch_count += 1
        if batch_count >= 10:  # 只测试前10个批次
            break

    mmap_iteration_time = time.time() - start_time
    print(f"内存映射版本迭代时间 (10批次): {mmap_iteration_time:.3f} 秒")

    # 清理
    del train_x_mmap, train_y_mmap, mmap_loader, data_mmap
    gc.collect()


def main():
    """主函数"""
    print("内存使用对比：传统DataLoader vs 内存映射DataLoader")
    print("=" * 60)

    # 创建测试数据
    test_dir, data_path = create_large_test_data(samples=5000, nodes=256, time_steps=24)

    try:
        # 对比测试
        traditional_results = test_traditional_loading(data_path)

        # 等待内存回收
        time.sleep(2)
        gc.collect()

        mmap_results = test_mmap_loading(data_path)

        # 等待内存回收
        time.sleep(2)
        gc.collect()

        test_dataloader_comparison(data_path)

        # 打印对比结果
        print("\n" + "=" * 60)
        print("📊 性能对比总结:")
        print(f"传统加载 - 内存增长: {traditional_results['memory_increase']:.2f} MB")
        print(f"传统加载 - 加载时间: {traditional_results['load_time']:.3f} 秒")

        print(f"内存映射 - 内存增长: {mmap_results['memory_increase']:.2f} MB")
        print(f"内存映射 - 加载时间: {mmap_results['load_time']:.3f} 秒")

        memory_savings = traditional_results["memory_increase"] - mmap_results["memory_increase"]
        memory_savings_percent = (memory_savings / traditional_results["memory_increase"]) * 100

        print(f"\n💾 内存节省: {memory_savings:.2f} MB ({memory_savings_percent:.1f}%)")

        if mmap_results["load_time"] < traditional_results["load_time"]:
            time_improvement = traditional_results["load_time"] - mmap_results["load_time"]
            print(f"⚡ 加载速度提升: {time_improvement:.3f} 秒")

        print("\n✅ 内存映射DataLoader优化完成！")

    finally:
        # 清理测试文件
        import shutil

        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
            print(f"\n🧹 已清理测试目录: {test_dir}")


if __name__ == "__main__":
    main()
