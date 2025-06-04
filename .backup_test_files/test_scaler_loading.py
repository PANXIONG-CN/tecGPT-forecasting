#!/usr/bin/env python3
"""
测试新生成的scaler.pkl文件能否正确加载
"""
import pickle
import sys
import os

# 添加src路径
sys.path.append("src")


def test_scaler_loading():
    print("=== 测试 Scaler 加载 ===")

    try:
        # 尝试加载scaler.pkl文件
        with open("./small_test_data/scaler.pkl", "rb") as f:
            scaler_data = pickle.load(f)

        print("✓ scaler.pkl 文件加载成功")

        # 检查内容
        print(f"scaler_data 包含的键: {list(scaler_data.keys())}")

        # 检查tec_scaler
        tec_scaler = scaler_data.get("tec_scaler")
        if tec_scaler:
            print(f"✓ tec_scaler 类型: {type(tec_scaler)}")
            print(f"✓ tec_scaler.scaler_type: {getattr(tec_scaler, 'scaler_type', 'N/A')}")
            print(f"✓ tec_scaler.mean_ shape: {tec_scaler.mean_.shape}")
            print(f"✓ tec_scaler.std_ shape: {tec_scaler.std_.shape}")

        # 检查sw_scaler
        sw_scaler = scaler_data.get("sw_scaler")
        if sw_scaler:
            print(f"✓ sw_scaler 类型: {type(sw_scaler)}")
            print(f"✓ sw_scaler.scaler_type: {getattr(sw_scaler, 'scaler_type', 'N/A')}")
            print(f"✓ sw_scaler.mean_ shape: {sw_scaler.mean_.shape}")
            print(f"✓ sw_scaler.std_ shape: {sw_scaler.std_.shape}")

        print("✓ 所有scaler对象加载并验证成功")
        return True

    except Exception as e:
        print(f"✗ 加载scaler.pkl时出错: {e}")
        return False


def test_standardscaler_with_new_pickle():
    print("\n=== 测试 StandardScaler 使用新的 scaler.pkl ===")

    try:
        from utils.util import StandardScaler

        # 创建StandardScaler实例
        scaler = StandardScaler(scaler_path="./small_test_data/scaler.pkl", device="cpu")
        print("✓ StandardScaler 使用新的 scaler.pkl 创建成功")

        # 检查一些属性
        print(f"✓ TEC mean shape: {scaler.np_mean_tec.shape}")
        print(f"✓ TEC std shape: {scaler.np_std_tec.shape}")

        if scaler.sw_mean_np is not None:
            print(f"✓ SW mean shape: {scaler.sw_mean_np.shape}")
            print(f"✓ SW std shape: {scaler.sw_std_np.shape}")

        return True

    except Exception as e:
        print(f"✗ StandardScaler 创建失败: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("开始测试新生成的 scaler.pkl 文件...")

    success1 = test_scaler_loading()
    success2 = test_standardscaler_with_new_pickle()

    if success1 and success2:
        print("\n🎉 所有测试通过！新的 scaler.pkl 文件工作正常。")
    else:
        print("\n❌ 部分测试失败，需要进一步检查。")
