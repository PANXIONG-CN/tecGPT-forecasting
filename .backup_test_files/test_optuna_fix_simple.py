#!/usr/bin/env python3
"""
简单测试 Optuna SQLite 修复
"""
import os
import sys

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_optuna_import():
    """测试 Optuna 导入"""
    try:
        import optuna
        print(f"✓ Optuna 导入成功，版本: {optuna.__version__}")
        return True
    except ImportError as e:
        print(f"✗ Optuna 导入失败: {e}")
        return False

def test_sqlite_storage():
    """测试 SQLite 存储"""
    try:
        import optuna
        import sqlite3
        
        # 确保 logs 目录存在
        os.makedirs("./logs", exist_ok=True)
        
        # 数据库路径
        db_path = "./logs/test_fix.db"
        
        # 删除旧文件
        if os.path.exists(db_path):
            os.remove(db_path)
        
        # 创建存储
        storage_url = f"sqlite:///{db_path}"
        print(f"测试存储URL: {storage_url}")
        
        storage = optuna.storages.RDBStorage(
            url=storage_url,
            engine_kwargs={"connect_args": {"timeout": 60, "check_same_thread": False}}
        )
        
        # 创建study
        study = optuna.create_study(
            study_name="test_fix",
            storage=storage,
            direction="minimize"
        )
        
        # 简单目标函数
        def objective(trial):
            x = trial.suggest_float("x", -10, 10)
            return x**2
        
        # 运行优化
        study.optimize(objective, n_trials=2)
        
        print(f"✓ SQLite 存储测试成功")
        print(f"✓ 最佳值: {study.best_value:.4f}")
        print(f"✓ 数据库文件: {db_path}")
        
        # 验证数据库
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM trials;")
            count = cursor.fetchone()[0]
            print(f"✓ 数据库中有 {count} 个trials")
            conn.close()
        
        return True
        
    except Exception as e:
        print(f"✗ SQLite 存储测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=== Optuna SQLite 修复测试 ===")
    
    # 测试导入
    if not test_optuna_import():
        return False
    
    # 测试SQLite存储
    if not test_sqlite_storage():
        return False
    
    print("\n🎉 所有测试通过！")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
