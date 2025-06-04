"""
测试 tec_train.py 能否正确加载 scaler.pkl
"""

import sys
import pickle

# 添加 src 目录到路径
sys.path.append(".")

# 导入必要的模块
from src.utils.util import StandardScaler, DummyScaler


def main():
    """主函数"""
    print(f"=== 测试 tec_train.py 加载 small_test_data/scaler.pkl ===")

    try:
        # 直接加载scaler.pkl
        print("1. 使用pickle直接加载scaler.pkl...")
        with open("./small_test_data/scaler.pkl", "rb") as f:
            scaler_data = pickle.load(f)
        print(f"✓ 直接加载成功! scaler_data 类型: {type(scaler_data)}")
        print(f"✓ scaler_data 键: {list(scaler_data.keys())}")

        # 使用StandardScaler加载
        print("\n2. 使用StandardScaler加载scaler.pkl...")
        scaler = StandardScaler(scaler_path="./small_test_data/scaler.pkl", device="cpu")
        print(f"✓ StandardScaler加载成功!")

        print("\n✅ 测试成功: scaler.pkl 可以被正确加载!")
        return True

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
