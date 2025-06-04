#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
内存映射DataLoader使用示例
展示如何在实际项目中使用优化后的DataLoader
"""

import sys

sys.path.append("src")

from utils.util import load_dataset


def example_usage():
    """内存映射DataLoader使用示例"""

    print("🚀 内存映射DataLoader使用示例")
    print("=" * 50)

    # 配置参数
    dataset_dir = "processed_tec_data"  # 数据目录
    scaler_path = "processed_tec_data/scaler.pkl"  # 标准化器路径
    batch_size = 32

    print(f"📁 数据目录: {dataset_dir}")
    print(f"📊 批次大小: {batch_size}")

    try:
        # 使用内存映射加载数据集
        print("\n🔄 加载数据集（使用内存映射）...")
        dataset = load_dataset(dataset_dir=dataset_dir, scaler_path=scaler_path, batch_size=batch_size, load_test=True, load_train_val=True)

        train_loader = dataset["train_loader"]
        val_loader = dataset["val_loader"]
        test_loader = dataset["test_loader"]
        scaler = dataset["scaler"]

        print("✅ 数据集加载完成！")
        print(f"   训练集批次数: {train_loader.num_batch}")
        print(f"   验证集批次数: {val_loader.num_batch}")
        print(f"   测试集批次数: {test_loader.num_batch}")

        # 示例：训练循环
        print("\n🔁 训练循环示例...")
        epoch = 1

        # 训练迭代
        train_iterator = train_loader.get_iterator()
        for batch_idx, (x_batch, y_batch) in enumerate(train_iterator):
            print(f"Epoch {epoch}, Batch {batch_idx+1}: " f"x shape {x_batch.shape}, y shape {y_batch.shape}")

            # 在这里进行模型训练
            # model_output = model(x_batch)
            # loss = criterion(model_output, y_batch)
            # ...

            if batch_idx >= 2:  # 只演示前3个批次
                print("   ...")
                break

        print(f"   训练集总共 {train_loader.num_batch} 个批次")

        # 示例：验证循环
        print("\n📊 验证循环示例...")
        val_iterator = val_loader.get_iterator()
        for batch_idx, (x_batch, y_batch) in enumerate(val_iterator):
            print(f"Validation Batch {batch_idx+1}: " f"x shape {x_batch.shape}, y shape {y_batch.shape}")

            # 在这里进行模型验证
            # with torch.no_grad():
            #     model_output = model(x_batch)
            #     val_loss = criterion(model_output, y_batch)
            # ...

            if batch_idx >= 1:  # 只演示前2个批次
                print("   ...")
                break

        print(f"   验证集总共 {val_loader.num_batch} 个批次")

        # 内存映射优势说明
        print("\n💡 内存映射优势:")
        print("   ✓ 避免将整个数据集加载到内存")
        print("   ✓ 支持任意大小的数据集")
        print("   ✓ 快速启动，按需加载批次")
        print("   ✓ 保持所有原有功能（打乱、填充等）")

    except FileNotFoundError as e:
        print(f"❌ 数据文件未找到: {e}")
        print("💡 请确保已运行数据预处理脚本生成数据文件")

    except Exception as e:
        print(f"❌ 发生错误: {e}")
        print("💡 请检查数据文件路径和格式")


if __name__ == "__main__":
    example_usage()
