#!/usr/bin/env python3
"""
测试 Optuna 与 SQLite 存储集成修复
"""

import os
import sys
import optuna
import sqlite3
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_sqlite_storage():
    """测试 SQLite 存储功能"""
    logger.info("=== 测试 SQLite 存储 ===")

    # 确保 logs 目录存在
    os.makedirs("./logs", exist_ok=True)

    # 数据库路径
    db_path = "./logs/test_optuna_fix.db"
    
    # 删除旧的数据库文件（如果存在）
    if os.path.exists(db_path):
        os.remove(db_path)
        logger.info(f"删除旧数据库文件: {db_path}")

    try:
        # 创建 SQLite 数据库
        storage_url = f"sqlite:///{db_path}"
        logger.info(f"使用存储 URL: {storage_url}")
        logger.info(f"Optuna 版本: {optuna.__version__}")

        # 简单的目标函数
        def simple_objective(trial):
            x = trial.suggest_float("x", -10, 10)
            return x**2

        # 先创建存储对象
        logger.info("创建 RDBStorage 对象...")
        storage = optuna.storages.RDBStorage(
            url=storage_url,
            engine_kwargs={"connect_args": {"timeout": 60}}
        )

        # 创建 study
        logger.info("创建 Optuna study...")
        study = optuna.create_study(
            study_name="test_sqlite_fix", 
            storage=storage, 
            direction="minimize", 
            load_if_exists=True
        )

        logger.info(f"✓ Optuna study 创建成功")
        logger.info(f"数据库路径: {db_path}")

        # 运行几个简单的 trials
        logger.info("运行优化...")
        study.optimize(simple_objective, n_trials=3)

        # 验证结果
        logger.info(f"✓ 完成 {len(study.trials)} 个 trials")
        logger.info(f"最佳值: {study.best_value:.6f}")
        logger.info(f"最佳参数: {study.best_params}")

        # 验证数据库文件
        if os.path.exists(db_path):
            file_size = os.path.getsize(db_path)
            logger.info(f"✓ 数据库文件已创建: {db_path} ({file_size} bytes)")

            # 验证数据库内容
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # 检查表
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            logger.info(f"✓ 数据库表: {[table[0] for table in tables]}")

            # 检查 trials 数量
            cursor.execute("SELECT COUNT(*) FROM trials;")
            trial_count = cursor.fetchone()[0]
            logger.info(f"✓ 数据库中的 trials 数量: {trial_count}")

            conn.close()

            return True
        else:
            logger.error(f"✗ 数据库文件未创建")
            return False

    except Exception as e:
        logger.error(f"✗ SQLite 存储测试失败: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = test_sqlite_storage()
    if success:
        logger.info("\n🎉 SQLite 存储测试成功！")
        sys.exit(0)
    else:
        logger.error("\n❌ SQLite 存储测试失败！")
        sys.exit(1)
