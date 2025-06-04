#!/usr/bin/env python3
"""
验证 Optuna 配置修复
"""
import os
import sys
import yaml

def test_config_files():
    """测试配置文件是否正确"""
    print("=== 配置文件验证 ===")
    
    # 1. 检查主配置文件
    config_path = "conf/config_optuna_test.yaml"
    print(f"\n1. 检查 {config_path}")
    
    if not os.path.exists(config_path):
        print(f"✗ 配置文件不存在: {config_path}")
        return False
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 检查关键配置
        if config.get('use_optuna') != True:
            print("✗ use_optuna 未设置为 true")
            return False
        
        if not config.get('optuna_storage'):
            print("✗ optuna_storage 未设置")
            return False
        
        if not config.get('optuna_study_name'):
            print("✗ optuna_study_name 未设置")
            return False
        
        print(f"✓ 主配置文件正确")
        print(f"  - use_optuna: {config['use_optuna']}")
        print(f"  - optuna_storage: {config['optuna_storage']}")
        print(f"  - optuna_trials: {config['optuna_trials']}")
        
    except Exception as e:
        print(f"✗ 读取配置文件失败: {e}")
        return False
    
    # 2. 检查 Hydra sweeper 配置
    sweeper_path = "conf/hydra/sweeper/test_optuna.yaml"
    print(f"\n2. 检查 {sweeper_path}")
    
    if not os.path.exists(sweeper_path):
        print(f"✗ Sweeper 配置文件不存在: {sweeper_path}")
        return False
    
    try:
        with open(sweeper_path, 'r', encoding='utf-8') as f:
            sweeper_config = yaml.safe_load(f)
        
        # 检查 sweeper 配置
        sweeper = sweeper_config.get('sweeper', {})
        if not sweeper.get('_target_'):
            print("✗ sweeper._target_ 未设置")
            return False
        
        if 'storage_options' not in sweeper:
            print("✗ storage_options 未设置")
            return False
        
        print(f"✓ Sweeper 配置文件正确")
        print(f"  - _target_: {sweeper['_target_']}")
        print(f"  - storage_options: {sweeper['storage_options']}")
        
    except Exception as e:
        print(f"✗ 读取 sweeper 配置文件失败: {e}")
        return False
    
    # 3. 检查数据集配置
    dataset_path = "conf/dataset/small_test_data.yaml"
    print(f"\n3. 检查 {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f"✗ 数据集配置文件不存在: {dataset_path}")
        return False
    
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset_config = yaml.safe_load(f)
        
        data_dir = dataset_config.get('data_dir')
        scaler_path = dataset_config.get('scaler_path')
        
        if not data_dir or not os.path.exists(data_dir):
            print(f"✗ 数据目录不存在: {data_dir}")
            return False
        
        if not scaler_path or not os.path.exists(scaler_path):
            print(f"✗ Scaler 文件不存在: {scaler_path}")
            return False
        
        print(f"✓ 数据集配置正确")
        print(f"  - data_dir: {data_dir}")
        print(f"  - scaler_path: {scaler_path}")
        
    except Exception as e:
        print(f"✗ 读取数据集配置文件失败: {e}")
        return False
    
    # 4. 检查 logs 目录
    print(f"\n4. 检查 logs 目录")
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir, exist_ok=True)
        print(f"✓ 创建 logs 目录: {logs_dir}")
    else:
        print(f"✓ logs 目录已存在: {logs_dir}")
    
    return True

def main():
    print("开始验证 Optuna 配置修复...")
    
    if test_config_files():
        print("\n🎉 所有配置文件验证通过！")
        print("\n下一步可以运行:")
        print("python src/tec_train.py --config-name config_optuna_test hydra/sweeper=test_optuna --multirun")
        return True
    else:
        print("\n❌ 配置文件验证失败！")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
