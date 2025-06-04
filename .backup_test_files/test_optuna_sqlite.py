#!/usr/bin/env python3
"""
测试 Optuna 与 SQLite 数据库集成
"""

import os
import sys
import optuna
import sqlite3
import subprocess

# 添加项目路径
sys.path.append(".")


def test_sqlite_storage():
    """测试 SQLite 存储功能"""
    print("=== 测试 SQLite 存储 ===")

    # 确保 logs 目录存在
    os.makedirs("./logs", exist_ok=True)

    # 数据库路径
    db_path = "./logs/test_optuna_final.db"

    # 删除旧的数据库文件（如果存在）
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"删除旧数据库文件: {db_path}")

    try:
        # 创建 SQLite 数据库
        storage_url = f"sqlite:///{db_path}"

        # 简单的目标函数
        def simple_objective(trial):
            x = trial.suggest_float("x", -10, 10)
            return x**2

        # 先创建数据库表结构
        import optuna.storages

        storage = optuna.storages.RDBStorage(storage_url)

        # 创建 study
        study = optuna.create_study(study_name="test_sqlite_study", storage=storage, direction="minimize", load_if_exists=True)

        print(f"✓ Optuna study 创建成功")
        print(f"数据库路径: {db_path}")

        # 运行几个简单的 trials
        study.optimize(simple_objective, n_trials=3)

        # 验证结果
        print(f"✓ 完成 {len(study.trials)} 个 trials")
        print(f"最佳值: {study.best_value:.6f}")
        print(f"最佳参数: {study.best_params}")

        # 验证数据库文件
        if os.path.exists(db_path):
            file_size = os.path.getsize(db_path)
            print(f"✓ 数据库文件已创建: {db_path} ({file_size} bytes)")

            # 验证数据库内容
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # 检查表
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            print(f"✓ 数据库表: {[table[0] for table in tables]}")

            # 检查 trials 数量
            cursor.execute("SELECT COUNT(*) FROM trials;")
            trial_count = cursor.fetchone()[0]
            print(f"✓ 数据库中的 trials 数量: {trial_count}")

            conn.close()

            return True
        else:
            print(f"✗ 数据库文件未创建")
            return False

    except Exception as e:
        print(f"✗ SQLite 存储测试失败: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_training_with_sqlite():
    """测试训练脚本与 SQLite 数据库的集成"""
    print("\n=== 测试训练脚本与 SQLite 集成 ===")

    # 数据库路径
    db_path = "./logs/test_training_optuna.db"

    # 删除旧的数据库文件（如果存在）
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"删除旧数据库文件: {db_path}")

    try:
        # 目标函数：运行训练脚本
        def training_objective(trial):
            learning_rate = trial.suggest_float("learning_rate", 1e-4, 2e-4, log=True)
            batch_size = trial.suggest_categorical("batch_size", [1, 2])

            cmd = [
                sys.executable,
                "src/tec_train.py",
                "--config-name",
                "config_optuna_test",
                f"trainer.learning_rate={learning_rate}",
                f"trainer.batch_size={batch_size}",
                "trainer.epochs=1",
                "use_optuna=false",
            ]

            print(f"Trial {trial.number}: lr={learning_rate:.2e}, batch={batch_size}")

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode != 0:
                print(f"Trial {trial.number} 失败")
                raise optuna.TrialPruned()

            # 提取验证 RMSE
            output_lines = result.stdout.split("\n")
            for line in output_lines:
                if "Returning validation RMSE:" in line:
                    val_rmse = float(line.split(":")[-1].strip())
                    print(f"Trial {trial.number} 完成: RMSE={val_rmse:.6f}")
                    return val_rmse

            raise optuna.TrialPruned()

        # 创建 study 使用 SQLite 存储
        storage_url = f"sqlite:///{db_path}"
        study = optuna.create_study(study_name="training_test_study", storage=storage_url, direction="minimize", load_if_exists=True)

        # 运行 2 个 trials
        study.optimize(training_objective, n_trials=2)

        # 验证结果
        print(f"\n✓ 训练测试完成")
        print(f"最佳 trial: {study.best_trial.number}")
        print(f"最佳 RMSE: {study.best_value:.6f}")
        print(f"最佳参数: {study.best_params}")

        # 验证数据库
        if os.path.exists(db_path):
            file_size = os.path.getsize(db_path)
            print(f"✓ 训练数据库已创建: {db_path} ({file_size} bytes)")
            return True
        else:
            print(f"✗ 训练数据库未创建")
            return False

    except Exception as e:
        print(f"✗ 训练与 SQLite 集成测试失败: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("=== Optuna SQLite 集成测试 ===")

    # 测试 1: 基本 SQLite 存储
    sqlite_ok = test_sqlite_storage()
    if not sqlite_ok:
        print("✗ SQLite 存储测试失败")
        return False

    # 测试 2: 训练脚本与 SQLite 集成
    training_ok = test_training_with_sqlite()
    if not training_ok:
        print("✗ 训练与 SQLite 集成测试失败")
        return False

    print("\n=== 所有 SQLite 测试通过 ===")
    print("✓ SQLite 数据库存储正常")
    print("✓ 训练脚本与 Optuna 集成正常")
    print("✓ 数据持久化功能正常")

    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 SQLite 集成测试完成！所有功能正常工作。")
        sys.exit(0)
    else:
        print("\n❌ SQLite 集成测试失败！请检查错误信息。")
        sys.exit(1)
