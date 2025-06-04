#!/usr/bin/env python3
"""
创建用于测试Optuna集成的小数据集
"""

import numpy as np
import os
import pickle
import sys

# 添加项目路径以导入实际的Scaler类
sys.path.append(".")
from data_preparation.preprocess_data import NodeScaler, FeatureScaler


def create_small_dataset():
    """创建小数据集"""
    print("创建小测试数据集...")

    # 创建输出目录
    output_dir = "./small_test_data"
    os.makedirs(output_dir, exist_ok=True)

    # 设置随机种子
    np.random.seed(42)

    # 小数据集参数
    # 对应 [样本数, 历史长度, 节点数, 特征数]
    history_len = 12  # P
    forecast_len = 12  # S
    num_nodes = 2911  # N (保持真实节点数以兼容模型)
    num_features = 12  # C_in (1 TEC + 5 SW + 6 Time)

    # 非常小的样本数以快速训练
    train_samples = 50
    val_samples = 15
    test_samples = 10

    print(f"创建数据集:")
    print(f"  训练样本: {train_samples}")
    print(f"  验证样本: {val_samples}")
    print(f"  测试样本: {test_samples}")
    print(f"  历史长度: {history_len}")
    print(f"  预测长度: {forecast_len}")
    print(f"  节点数: {num_nodes}")
    print(f"  特征数: {num_features}")

    # 生成训练数据
    train_x = np.random.randn(train_samples, history_len, num_nodes, num_features).astype(np.float32)
    train_y = np.random.randn(train_samples, forecast_len, num_nodes, 1).astype(np.float32)

    # 生成验证数据
    val_x = np.random.randn(val_samples, history_len, num_nodes, num_features).astype(np.float32)
    val_y = np.random.randn(val_samples, forecast_len, num_nodes, 1).astype(np.float32)

    # 生成测试数据
    test_x = np.random.randn(test_samples, history_len, num_nodes, num_features).astype(np.float32)
    test_y = np.random.randn(test_samples, forecast_len, num_nodes, 1).astype(np.float32)

    # 保存数据
    np.savez_compressed(os.path.join(output_dir, "train.npz"), x=train_x, y=train_y)
    np.savez_compressed(os.path.join(output_dir, "val.npz"), x=val_x, y=val_y)
    np.savez_compressed(os.path.join(output_dir, "test.npz"), x=test_x, y=test_y)

    print("数据文件已保存:")
    for filename in ["train.npz", "val.npz", "test.npz"]:
        filepath = os.path.join(output_dir, filename)
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"  {filename}: {size_mb:.1f} MB")


def create_dummy_scaler():
    """创建虚拟的标准化器"""
    print("创建虚拟标准化器...")

    # 使用实际的标准化器类
    tec_scaler = NodeScaler(2911)  # 节点数
    sw_scaler = FeatureScaler(5)  # 空间天气特征数

    # 初始化标准化器参数
    tec_scaler.mean_ = np.zeros(2911, dtype=np.float32)
    tec_scaler.std_ = np.ones(2911, dtype=np.float32)

    sw_scaler.mean_ = np.zeros(5, dtype=np.float32)
    sw_scaler.std_ = np.ones(5, dtype=np.float32)

    scaler_data = {"tec_scaler": tec_scaler, "sw_scaler": sw_scaler}

    scaler_path = "./small_test_data/scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler_data, f)

    print(f"标准化器已保存到: {scaler_path}")


def create_test_dataset_config():
    """创建测试数据集配置文件"""
    print("创建测试数据集配置...")

    config_content = """# conf/dataset/small_test_data.yaml
# 小测试数据集配置
data_dir: ./small_test_data/
scaler_path: ./small_test_data/scaler.pkl
num_nodes: 2911
n_lat: 41
n_lon: 71
history_len: 12
forecast_len: 12
use_time_features: true
tec_feat_dim: 1
sw_feat_dim: 5
time_feat_dim: 6
"""

    config_path = "conf/dataset/small_test_data.yaml"
    with open(config_path, "w") as f:
        f.write(config_content)

    print(f"数据集配置已保存到: {config_path}")


if __name__ == "__main__":
    print("=== 创建Optuna测试数据集 ===")
    create_small_dataset()
    create_dummy_scaler()
    create_test_dataset_config()
    print("\n✅ 测试数据集创建完成!")
    print("\n使用方法:")
    print("python src/tec_train.py dataset=small_test_data trainer.epochs=1")
