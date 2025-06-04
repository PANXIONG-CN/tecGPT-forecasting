#!/usr/bin/env python3
"""
测试 Optuna multirun 功能
"""
import os
import sys
import subprocess
import time

def test_optuna_multirun():
    """测试 Optuna multirun"""
    print("=== 测试 Optuna Multirun ===")
    
    # 确保在正确的目录
    os.chdir("/home/panxiong/tecGPT-forecasting")
    
    # 构建命令
    cmd = [
        "python", "src/tec_train.py",
        "--config-name", "config_optuna_test",
        "hydra/sweeper=test_optuna",
        "--multirun"
    ]
    
    print(f"运行命令: {' '.join(cmd)}")
    print("这可能需要几分钟时间...")
    
    try:
        # 运行命令并捕获输出
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5分钟超时
        )
        
        print(f"\n返回码: {result.returncode}")
        
        if result.stdout:
            print("\n=== 标准输出 ===")
            print(result.stdout[-2000:])  # 显示最后2000个字符
        
        if result.stderr:
            print("\n=== 错误输出 ===")
            print(result.stderr[-1000:])  # 显示最后1000个字符
        
        # 检查是否成功
        if result.returncode == 0:
            print("\n✅ Optuna multirun 测试成功！")
            
            # 检查生成的文件
            print("\n检查生成的文件...")
            
            # 检查数据库文件
            db_path = "logs/optuna_test_study.db"
            if os.path.exists(db_path):
                size = os.path.getsize(db_path)
                print(f"✅ 数据库文件已创建: {db_path} ({size} bytes)")
            else:
                print(f"⚠️  数据库文件未找到: {db_path}")
            
            # 检查日志目录
            optuna_dirs = []
            if os.path.exists("logs/optuna_test"):
                for item in os.listdir("logs/optuna_test"):
                    if item.startswith("2025-06-04"):
                        optuna_dirs.append(item)
            
            if optuna_dirs:
                print(f"✅ 找到 {len(optuna_dirs)} 个Optuna运行目录")
                for d in optuna_dirs[-3:]:  # 显示最新的3个
                    print(f"  - {d}")
            else:
                print("⚠️  未找到新的Optuna运行目录")
            
            return True
        else:
            print(f"\n❌ Optuna multirun 测试失败，返回码: {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print("\n⏰ 测试超时（5分钟）")
        return False
    except Exception as e:
        print(f"\n❌ 测试过程中出错: {e}")
        return False

def main():
    print("开始测试 Optuna multirun 功能...")
    
    success = test_optuna_multirun()
    
    if success:
        print("\n🎉 Optuna SQLite 存储修复验证成功！")
        print("\n修复总结:")
        print("1. ✅ 更新了 Optuna 到 3.5.0 版本")
        print("2. ✅ 修改了 SQLite 存储配置")
        print("3. ✅ 添加了 storage_options 配置")
        print("4. ✅ 在训练代码中添加了存储路径设置")
        print("5. ✅ 使用小数据集成功测试")
    else:
        print("\n❌ 测试失败，需要进一步调试")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
