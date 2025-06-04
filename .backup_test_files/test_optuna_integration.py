#!/usr/bin/env python3
"""
测试 Optuna 集成的简化脚本
直接使用 Optuna 而不依赖 Hydra sweeper
"""

import os
import sys
import optuna
import pickle
import tempfile
import subprocess
from pathlib import Path

# 添加项目路径
sys.path.append(".")


def test_scaler_loading():
    """测试 scaler.pkl 文件是否能正确加载"""
    print("=== 测试 Scaler 加载 ===")

    try:
        # 测试直接加载 pickle 文件
        with open("./small_test_data/scaler.pkl", "rb") as f:
            scaler_data = pickle.load(f)

        print(f"✓ scaler.pkl 文件加载成功")
        print(f"scaler_data 键: {list(scaler_data.keys())}")
        print(f"tec_scaler 类型: {type(scaler_data['tec_scaler'])}")
        print(f"sw_scaler 类型: {type(scaler_data['sw_scaler'])}")

        # 测试 StandardScaler 加载
        from src.utils.util import StandardScaler

        scaler = StandardScaler(scaler_path="./small_test_data/scaler.pkl", device="cpu")
        print("✓ StandardScaler 加载成功!")
        print(f"TEC mean shape: {scaler.np_mean_tec.shape}")
        print(f"TEC std shape: {scaler.np_std_tec.shape}")

        return True

    except Exception as e:
        print(f"✗ Scaler 加载失败: {e}")
        import traceback

        traceback.print_exc()
        return False


def objective(trial):
    """Optuna 目标函数"""
    # 建议超参数
    learning_rate = trial.suggest_float("learning_rate", 5e-5, 2e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [1, 2])
    d_embed = trial.suggest_categorical("d_embed", [64, 128])
    llm_layers_to_use = trial.suggest_int("llm_layers_to_use", 2, 3)
    U_unfrozen_mha = trial.suggest_categorical("U_unfrozen_mha", [1, 2])

    # 构建命令
    cmd = [
        sys.executable,
        "src/tec_train.py",
        "--config-name",
        "config_optuna_test",
        f"trainer.learning_rate={learning_rate}",
        f"trainer.batch_size={batch_size}",
        f"model.d_embed={d_embed}",
        f"model.llm_layers_to_use={llm_layers_to_use}",
        f"model.U_unfrozen_mha={U_unfrozen_mha}",
        "trainer.epochs=1",
        "use_optuna=false",  # 禁用内置 Optuna 以避免冲突
    ]

    print(f"\n--- Trial {trial.number} ---")
    print(f"参数: lr={learning_rate:.2e}, batch={batch_size}, embed={d_embed}, layers={llm_layers_to_use}, mha={U_unfrozen_mha}")

    try:
        # 运行训练
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            print(f"✗ Trial {trial.number} 失败:")
            print(f"stderr: {result.stderr}")
            raise optuna.TrialPruned()

        # 从输出中提取验证 RMSE
        output_lines = result.stdout.split("\n")
        val_rmse = None

        for line in output_lines:
            if "Returning validation RMSE:" in line:
                val_rmse = float(line.split(":")[-1].strip())
                break
            elif "Best validation RMSE achieved during training:" in line:
                val_rmse = float(line.split(":")[-1].strip())
                break

        if val_rmse is None:
            print(f"✗ Trial {trial.number}: 无法提取验证 RMSE")
            raise optuna.TrialPruned()

        print(f"✓ Trial {trial.number} 完成, 验证 RMSE: {val_rmse:.6f}")
        return val_rmse

    except subprocess.TimeoutExpired:
        print(f"✗ Trial {trial.number} 超时")
        raise optuna.TrialPruned()
    except Exception as e:
        print(f"✗ Trial {trial.number} 异常: {e}")
        raise optuna.TrialPruned()


def test_optuna_optimization():
    """测试 Optuna 优化"""
    print("\n=== 测试 Optuna 优化 ===")

    # 创建 Optuna study (使用内存存储)
    study_name = "test_optuna_integration"

    try:
        study = optuna.create_study(study_name=study_name, direction="minimize")

        print(f"✓ Optuna study 创建成功 (内存存储)")

        # 运行优化
        n_trials = 2
        print(f"开始运行 {n_trials} 个 trials...")

        study.optimize(objective, n_trials=n_trials)

        # 显示结果
        print(f"\n=== 优化结果 ===")
        print(f"最佳 trial: {study.best_trial.number}")
        print(f"最佳值: {study.best_value:.6f}")
        print(f"最佳参数: {study.best_params}")

        # 验证 trials 数量
        completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        print(f"✓ 完成的 trials: {completed_trials}/{n_trials}")

        return True

    except Exception as e:
        print(f"✗ Optuna 优化失败: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("=== Optuna 集成测试 ===")

    # 测试 1: Scaler 加载
    scaler_ok = test_scaler_loading()
    if not scaler_ok:
        print("✗ Scaler 测试失败，停止测试")
        return False

    # 测试 2: Optuna 优化
    optuna_ok = test_optuna_optimization()
    if not optuna_ok:
        print("✗ Optuna 优化测试失败")
        return False

    print("\n=== 所有测试通过 ===")
    print("✓ Pickle 序列化问题已解决")
    print("✓ Optuna 集成正常工作")
    print("✓ 小数据集测试成功")

    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 测试完成！所有功能正常工作。")
        sys.exit(0)
    else:
        print("\n❌ 测试失败！请检查错误信息。")
        sys.exit(1)
