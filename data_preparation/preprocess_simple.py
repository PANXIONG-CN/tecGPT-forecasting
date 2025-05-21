import h5py
import numpy as np
import pandas as pd
from datetime import datetime
import math
import os
import pickle
from tqdm import tqdm

# --- 配置参数 ---
HDF5_DIR = "data_preparation"  # 存放年度 HDF5 文件的目录
OUTPUT_DIR = "./processed_tec_data"  # 处理后 .npz 文件的输出目录
SCALER_PATH = os.path.join(OUTPUT_DIR, "scaler.pkl")  # Scaler 参数保存路径

# 为简化测试，只处理几年数据
YEARS_TRAIN = [2013, 2014]  # 训练集年份
YEARS_VAL = [2020]  # 验证集年份
YEARS_TEST = [2022]  # 测试集年份

HISTORY_LEN = 12  # P: 使用过去 12 个时间步
FORECAST_LEN = 12  # S: 预测未来 12 个时间步

N_LAT = 41
N_LON = 71
N_NODES = N_LAT * N_LON

# --- 特征索引和名称 ---
SW_INDICES = ["Kp_Index", "Dst_Index", "ap_Index", "F107_Index", "AE_Index"]
TIME_COMPONENTS = ["year", "month", "day", "hour", "day_of_year"]

# 填充值
FILL_VALUE = -9999.0
# ------------------


def is_leap(year):
    """判断是否是闰年"""
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def generate_cyclical_features(df_time):
    """为 Pandas DataFrame 生成周期性特征"""
    # 小时 (0, 2, ..., 22)
    hour = df_time["hour"]
    df_time["hour_sin"] = np.sin(2 * math.pi * hour / 24.0)
    df_time["hour_cos"] = np.cos(2 * math.pi * hour / 24.0)

    # 星期几
    dt_series = pd.to_datetime(df_time[["year", "month", "day", "hour"]])
    dayofweek = dt_series.dt.dayofweek
    df_time["dayofweek_sin"] = np.sin(2 * math.pi * dayofweek / 7.0)
    df_time["dayofweek_cos"] = np.cos(2 * math.pi * dayofweek / 7.0)

    # 年积日 (1-365/366)
    dayofyear = df_time["day_of_year"]
    days_in_year = df_time["year"].apply(lambda y: 366 if is_leap(y) else 365)
    df_time["dayofyear_sin"] = np.sin(2 * math.pi * dayofyear / days_in_year)
    df_time["dayofyear_cos"] = np.cos(2 * math.pi * dayofyear / days_in_year)

    return df_time[["hour_sin", "hour_cos", "dayofweek_sin", "dayofweek_cos", "dayofyear_sin", "dayofyear_cos"]].values


def process_single_year(year, output_dir, scaler=None, save_batch_size=500, split_name=None):
    """处理单年份数据，计算统计量或创建序列"""
    # 根据实际文件名格式进行调整
    filepath = os.path.join(HDF5_DIR, f"CRIM_SW2hr_AI_v1.2_{year}_DataDrivenRange_CN.hdf5")
    if not os.path.exists(filepath):
        print(f"文件不存在: {filepath}")
        return None, None, None, 0

    print(f"处理年份 {year}...")

    # 加载数据
    with h5py.File(filepath, "r") as f:
        # 1. 加载 TEC 数据
        print("加载TEC数据...")
        tec_data = f["/ionosphere/TEC"][:]
        fill_value = f["/ionosphere/TEC"].attrs.get("_FillValue", FILL_VALUE)
        tec_mask = tec_data == fill_value
        tec_data_original = tec_data.copy()  # 保存原始数据用于生成Y
        tec_data[tec_mask] = np.nan  # 替换为NaN
        tec_data_flat = tec_data.reshape(tec_data.shape[0], -1)  # (N_times_year, N_nodes)

        # 2. 加载时间组件
        print("加载时间组件...")
        time_components = {comp: f[f"/coordinates/{comp}"][:] for comp in TIME_COMPONENTS}
        df_time = pd.DataFrame(time_components)

        # 3. 加载空间天气指数
        print("加载空间天气指数...")
        sw_data = {}
        for idx_name in SW_INDICES:
            dset = f[f"/space_weather_indices/{idx_name}"]
            data = dset[:]
            # 特殊处理
            if idx_name == "Kp_Index":
                scale_factor = dset.attrs.get("scale_factor", 0.1)
                data = data.astype(np.float32) * scale_factor
            elif idx_name == "F107_Index":
                fill_val_f107 = dset.attrs.get("_FillValue", 999.9)
                data[data == fill_val_f107] = np.nan
                data_series = pd.Series(data)
                data_series.fillna(method="ffill", inplace=True)
                data_series.fillna(method="bfill", inplace=True)
                data = data_series.values
                data = np.log10(np.maximum(data, 1e-6))
            sw_data[idx_name] = data.astype(np.float32)

        df_sw = pd.DataFrame(sw_data)
        sw_array = df_sw[SW_INDICES].values

        # 4. 生成周期性时间特征
        print("生成周期性时间特征...")
        time_features_array = generate_cyclical_features(df_time)

    # 如果只是计算统计量，返回原始数据
    if scaler is None:
        print(f"返回年份 {year} 的原始数据用于计算统计量")
        return tec_data_flat, sw_array, year, 0  # 添加0作为第四个返回值

    # 否则，应用标准化并创建序列
    print(f"开始为年份 {year} 创建序列...")
    tec_mean, tec_std = scaler["tec_mean"], scaler["tec_std"]
    sw_mean, sw_std = scaler["sw_mean"], scaler["sw_std"]

    # 标准化
    tec_scaled = (tec_data_flat - tec_mean[np.newaxis, :]) / tec_std[np.newaxis, :]
    tec_scaled[np.isnan(tec_scaled)] = 0

    sw_scaled = (sw_array - sw_mean[np.newaxis, :]) / sw_std[np.newaxis, :]
    sw_scaled[np.isnan(sw_scaled)] = 0

    # 确保目录存在
    if split_name:
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

    # 创建和保存少量序列作为测试
    n_samples = min(tec_scaled.shape[0] - HISTORY_LEN - FORECAST_LEN + 1, 1000)  # 限制最大样本数

    if n_samples <= 0:
        print(f"年份 {year} 无法创建序列，数据量不足")
        return None, None, None, 0

    # 简化：只创建一小批数据
    print(f"为年份 {year} 创建 {n_samples} 个序列样本")

    # 预处理特征
    N = tec_scaled.shape[1]
    C_sw = sw_scaled.shape[1]
    C_time = time_features_array.shape[1]
    C_in = 1 + C_sw + C_time

    sample_count = 0
    batch_idx = 0

    for start_idx in range(0, n_samples, save_batch_size):
        end_idx = min(start_idx + save_batch_size, n_samples)
        current_batch_size = end_idx - start_idx

        # 为当前批次创建序列
        X_batch = np.zeros((current_batch_size, HISTORY_LEN, N, C_in), dtype=np.float32)
        Y_batch = np.zeros((current_batch_size, FORECAST_LEN, N, 1), dtype=np.float32)

        for i in range(current_batch_size):
            idx = start_idx + i

            # 提取输入特征
            x_tec = tec_scaled[idx : idx + HISTORY_LEN, :]
            x_sw = sw_scaled[idx : idx + HISTORY_LEN, :]
            x_time = time_features_array[idx : idx + HISTORY_LEN, :]

            # 扩展特征
            x_sw_expanded = np.expand_dims(x_sw, axis=1).repeat(N, axis=1)
            x_time_expanded = np.expand_dims(x_time, axis=1).repeat(N, axis=1)
            x_tec_expanded = np.expand_dims(x_tec, axis=2)

            # 合并
            x_combined = np.concatenate([x_tec_expanded, x_sw_expanded, x_time_expanded], axis=2)
            X_batch[i] = x_combined

            # 输出标签
            y_tec = tec_data_original.reshape(tec_data_original.shape[0], -1)[idx + HISTORY_LEN : idx + HISTORY_LEN + FORECAST_LEN, :]
            Y_batch[i] = np.expand_dims(y_tec, axis=2)

        # 保存批次
        if split_name:
            batch_filename = f"{split_name}_{year}_{batch_idx+1:03d}.npz"
            output_filepath = os.path.join(output_dir, split_name, batch_filename)
            np.savez_compressed(output_filepath, x=X_batch, y=Y_batch)
            print(f"已保存批次 {batch_idx+1} 到 {output_filepath}")

        sample_count += current_batch_size
        batch_idx += 1

    print(f"年份 {year} 处理完成，共创建 {sample_count} 个序列")
    return None, None, None, sample_count


def main():
    """主处理函数"""
    print("开始预处理...")

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)

    # 1. 计算训练集统计量
    print("\n--- 计算训练集统计量 ---")
    tec_mean_sums = []
    tec_std_sums = []
    sw_mean_sums = []
    sw_std_sums = []
    total_samples = 0

    for year in YEARS_TRAIN:
        tec_data, sw_data, year_processed, _ = process_single_year(year, OUTPUT_DIR)
        if tec_data is not None:
            # 计算统计量
            print(f"计算年份 {year} 的统计量...")
            year_tec_mean = np.nanmean(tec_data, axis=0)
            year_tec_std = np.nanstd(tec_data, axis=0)
            year_sw_mean = np.nanmean(sw_data, axis=0)
            year_sw_std = np.nanstd(sw_data, axis=0)
            year_samples = tec_data.shape[0]

            # 添加到加权和
            tec_mean_sums.append((year_tec_mean, year_samples))
            tec_std_sums.append((year_tec_std, year_samples))
            sw_mean_sums.append((year_sw_mean, year_samples))
            sw_std_sums.append((year_sw_std, year_samples))

            total_samples += year_samples
            print(f"年份 {year} 统计量计算完成，样本数: {year_samples}")

    # 计算加权平均
    if total_samples == 0:
        raise ValueError("没有找到训练数据，无法计算统计量")

    print(f"计算所有年份的加权平均值 (总样本数: {total_samples})...")

    # 计算加权平均
    tec_means = np.zeros_like(tec_mean_sums[0][0])
    tec_stds = np.zeros_like(tec_std_sums[0][0])
    sw_means = np.zeros_like(sw_mean_sums[0][0])
    sw_stds = np.zeros_like(sw_std_sums[0][0])

    for mean, weight in tec_mean_sums:
        tec_means += mean * (weight / total_samples)
    for std, weight in tec_std_sums:
        tec_stds += std * (weight / total_samples)
    for mean, weight in sw_mean_sums:
        sw_means += mean * (weight / total_samples)
    for std, weight in sw_std_sums:
        sw_stds += std * (weight / total_samples)

    # 防止标准差为零
    tec_stds[tec_stds < 1e-10] = 1.0
    sw_stds[sw_stds < 1e-10] = 1.0

    # 保存Scaler参数
    scaler_params = {"tec_mean": tec_means, "tec_std": tec_stds, "sw_mean": sw_means, "sw_std": sw_stds}
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler_params, f)
    print(f"Scaler参数已保存到 {SCALER_PATH}")

    # 2. 生成序列数据
    print("\n--- 创建训练集序列 ---")
    for year in YEARS_TRAIN:
        _, _, _, count = process_single_year(year, OUTPUT_DIR, scaler_params, split_name="train")
        print(f"训练集年份 {year} 创建了 {count} 个序列")

    print("\n--- 创建验证集序列 ---")
    for year in YEARS_VAL:
        _, _, _, count = process_single_year(year, OUTPUT_DIR, scaler_params, split_name="val")
        print(f"验证集年份 {year} 创建了 {count} 个序列")

    print("\n--- 创建测试集序列 ---")
    for year in YEARS_TEST:
        _, _, _, count = process_single_year(year, OUTPUT_DIR, scaler_params, split_name="test")
        print(f"测试集年份 {year} 创建了 {count} 个序列")

    print("\n预处理完成!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback

        traceback.print_exc()
