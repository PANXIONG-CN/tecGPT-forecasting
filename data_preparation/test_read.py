import h5py
import numpy as np
import pandas as pd
import os
import sys

print("开始测试脚本")

# 测试文件路径
hdf5_dir = "data_preparation"
year = 2013
filepath = os.path.join(hdf5_dir, f"CRIM_SW2hr_AI_v1.2_{year}_DataDrivenRange_CN.hdf5")

print(f"尝试打开文件: {filepath}")
print(f"文件是否存在: {os.path.exists(filepath)}")

# 尝试打开HDF5文件
with h5py.File(filepath, "r") as f:
    print("成功打开HDF5文件")

    # 查看文件结构
    print("文件结构:")
    print(list(f.keys()))

    # 读取TEC数据
    print("读取TEC数据...")
    tec_data = f["/ionosphere/TEC"][:]
    print(f"TEC数据形状: {tec_data.shape}")

    # 读取空间天气指数
    print("\n读取空间天气指数...")
    sw_indices = ["Kp_Index", "Dst_Index", "ap_Index", "F107_Index", "AE_Index"]
    for idx_name in sw_indices:
        dset = f[f"/space_weather_indices/{idx_name}"]
        data = dset[:]
        print(f"{idx_name} 形状: {data.shape}, 样例值: {data[:5]}")

    # 读取时间信息
    print("\n读取时间信息...")
    time_components = ["year", "month", "day", "hour", "day_of_year"]
    for comp in time_components:
        data = f[f"/coordinates/{comp}"][:]
        print(f"{comp} 形状: {data.shape}, 样例值: {data[:5]}")

print("\n测试完成")
