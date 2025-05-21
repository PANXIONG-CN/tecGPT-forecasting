import h5py
import numpy as np
import pandas as pd
from datetime import datetime
import math
import os
import pickle
from tqdm import tqdm  # 用于显示进度条

# --- 配置参数 ---
# 根据实际文件位置修改目录
HDF5_DIR = "data_preparation"  # 存放年度 HDF5 文件的目录
OUTPUT_DIR = "./processed_tec_data"  # 处理后 .npz 文件的输出目录
SCALER_PATH = os.path.join(OUTPUT_DIR, "scaler.pkl")  # Scaler 参数保存路径

YEARS_ALL = list(range(2013, 2026))  # 数据覆盖的所有年份 (2013-2025)
# --- 数据集划分年份 ---
YEARS_TRAIN = list(range(2013, 2020))  # 2013-2019
YEARS_VAL = list(range(2020, 2022))  # 2020-2021
YEARS_TEST = list(range(2022, 2026))  # 2022-2025

TARGET_END_DATE = "2025-04-30 22:00:00"  # 测试集最终日期和时间

HISTORY_LEN = 12  # P: 使用过去 12 个时间步 (24 小时)
FORECAST_LEN = 12  # S: 预测未来 12 个时间步 (24 小时)

N_LAT = 41
N_LON = 71
N_NODES = N_LAT * N_LON

# --- 特征索引和名称 (根据 HDF5 文件) ---
SW_INDICES = ["Kp_Index", "Dst_Index", "ap_Index", "F107_Index", "AE_Index"]
N_SW_INDICES = len(SW_INDICES)
TIME_COMPONENTS = ["year", "month", "day", "hour", "day_of_year"]  # 用于生成周期特征

# 填充值 (从HDF5文件属性中确认)
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

    # 星期几 (0=Monday, 6=Sunday)
    # 需要先构造 datetime 对象
    # 确保 year, month, day 列存在
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


def create_and_save_sequences_by_year(year, hdf5_dir, output_dir, p, s, tec_scaler, sw_scaler, split_name=None, save_batch_size=1000):
    """为单个年份创建和保存序列，避免一次加载所有年份的数据"""
    # 根据实际文件名格式进行调整
    filepath = os.path.join(hdf5_dir, f"CRIM_SW2hr_AI_v1.2_{year}_DataDrivenRange_CN.hdf5")
    if not os.path.exists(filepath):
        print(f"Warning: File not found for year {year}, skipping: {filepath}")
        return 0  # 返回处理的样本数为0

    # 如果年份属于特定的分割集，设置输出目录
    if split_name:
        year_split_dir = os.path.join(output_dir, split_name)
        os.makedirs(year_split_dir, exist_ok=True)
    else:
        return 0  # 如果没有分割标识，跳过处理

    print(f"Processing year {year} for {split_name} split...")

    try:
        # 加载并处理年度数据
        with h5py.File(filepath, "r") as f:
            print(f"Successfully opened {filepath}")
            # 1. 加载 TEC 数据
            print(f"Reading TEC data from {filepath}")
            tec_data = f["/ionosphere/TEC"][:]  # (N_times_year, 41, 71)
            print(f"TEC data shape: {tec_data.shape}")
            fill_value = f["/ionosphere/TEC"].attrs.get("_FillValue", FILL_VALUE)
            print(f"Fill value: {fill_value}")
            tec_mask = tec_data == fill_value
            tec_data_original = tec_data.copy()  # 保存原始数据用于Y
            tec_data[tec_mask] = np.nan  # 将填充值替换为NaN
            tec_data_flat = tec_data.reshape(tec_data.shape[0], -1)  # (N_times_year, N_nodes)
            print(f"TEC data flattened shape: {tec_data_flat.shape}")

            # 2. 加载时间组件
            print("Loading time components...")
            time_components = {comp: f[f"/coordinates/{comp}"][:] for comp in TIME_COMPONENTS}
            df_time = pd.DataFrame(time_components)
            print(f"Time components loaded, shape: {df_time.shape}")

            # 3. 加载空间天气指数
            print("Loading space weather indices...")
            sw_data = {}
            for idx_name in SW_INDICES:
                dset = f[f"/space_weather_indices/{idx_name}"]
                data = dset[:]
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
            sw_array = df_sw[SW_INDICES].values  # (N_times_year, N_SW_Indices=5)
            print(f"Space weather indices loaded, shape: {sw_array.shape}")

            # 4. 生成周期性时间特征
            print("Generating cyclical time features...")
            time_features_array = generate_cyclical_features(df_time)  # (N_times_year, N_Time_Features=6)
            print(f"Cyclical time features generated, shape: {time_features_array.shape}")

        # 应用标准化
        print("Applying standardization...")
        tec_mean, tec_std = tec_scaler
        sw_mean, sw_std = sw_scaler

        # 标准化 TEC 数据
        tec_scaled = (tec_data_flat - tec_mean[np.newaxis, :]) / tec_std[np.newaxis, :]
        tec_scaled[np.isnan(tec_scaled)] = 0  # 替换NaN

        # 标准化空间天气指数
        sw_scaled = (sw_array - sw_mean[np.newaxis, :]) / sw_std[np.newaxis, :]
        sw_scaled[np.isnan(sw_scaled)] = 0

        # 确保时间特征不含NaN
        time_features_array[np.isnan(time_features_array)] = 0

        # 预处理特征以加速序列创建
        N = tec_scaled.shape[1]  # 节点数
        C_sw = sw_scaled.shape[1]  # 空间天气指数数量
        C_time = time_features_array.shape[1]  # 时间特征数量
        C_in = 1 + C_sw + C_time  # 总输入特征数

        # 预先扩展 SW 和 Time 特征
        print(f"Preparing features for year {year}...")
        sw_expanded = np.expand_dims(sw_scaled, axis=1).repeat(N, axis=1)
        time_expanded = np.expand_dims(time_features_array, axis=1).repeat(N, axis=1)
        tec_expanded = np.expand_dims(tec_scaled, axis=2)

        # 拼接特征
        all_x_features = np.concatenate([tec_expanded, sw_expanded, time_expanded], axis=2)

        # 处理原始TEC数据用于Y
        tec_data_original_flat = tec_data_original.reshape(tec_data_original.shape[0], -1)
        raw_tec_y = np.expand_dims(tec_data_original_flat, axis=2)  # (Time, N, 1)

        # 计算可以创建的序列数量
        n_samples = tec_scaled.shape[0] - HISTORY_LEN - FORECAST_LEN + 1
        if n_samples <= 0:
            print(f"No sequences can be created for year {year}")
            return 0

        print(f"Will create {n_samples} sequences for year {year}")

        # 分批创建和保存序列
        num_batches = (n_samples + save_batch_size - 1) // save_batch_size
        total_saved = 0

        for batch_idx in tqdm(range(num_batches), desc=f"Creating and saving sequences for year {year}"):
            start_idx = batch_idx * save_batch_size
            end_idx = min(start_idx + save_batch_size, n_samples)
            batch_size_actual = end_idx - start_idx

            # 为当前批次创建序列
            X_batch = np.zeros((batch_size_actual, p, N, C_in), dtype=np.float32)
            Y_batch = np.zeros((batch_size_actual, s, N, 1), dtype=np.float32)

            for i in range(batch_size_actual):
                idx = start_idx + i
                X_batch[i] = all_x_features[idx : idx + p]
                Y_batch[i] = raw_tec_y[idx + p : idx + p + s]

            # 保存批次
            batch_filename = f"{split_name}_{year}_{batch_idx+1:03d}.npz"
            output_filepath = os.path.join(output_dir, split_name, batch_filename)
            np.savez_compressed(output_filepath, x=X_batch, y=Y_batch)
            print(f"Saved {batch_size_actual} sequences to {output_filepath}")

            total_saved += batch_size_actual

        print(f"Total {total_saved} sequences created and saved for year {year}")
        return total_saved
    except Exception as e:
        print(f"Error processing year {year}: {str(e)}")
        import traceback

        traceback.print_exc()
        return 0


def preprocess_tec_data_stream(hdf5_dir, output_dir, scaler_path, years_all, years_train, years_val, years_test, p, s):
    """流式处理版本的主预处理函数，避免一次加载所有数据"""
    os.makedirs(output_dir, exist_ok=True)

    # 为每个分割创建目录
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)

    # --- 1. 首先仅计算训练集的Scaler参数 ---
    print("Calculating scaler parameters from training data...")

    # 收集训练集统计量
    tec_means_sum = None
    tec_stds_sum = None
    sw_means_sum = None
    sw_stds_sum = None
    n_train_samples = 0

    # 首先计算每年的均值
    for year in years_train:
        filepath = os.path.join(hdf5_dir, f"CRIM_SW2hr_AI_v1.2_{year}_DataDrivenRange_CN.hdf5")
        if not os.path.exists(filepath):
            print(f"Warning: File not found for year {year}, skipping.")
            continue

        print(f"Computing statistics for year {year}...")
        try:
            with h5py.File(filepath, "r") as f:
                # TEC数据
                print(f"Reading TEC data for statistics from {filepath}")
                tec_data = f["/ionosphere/TEC"][:]
                print(f"TEC data shape: {tec_data.shape}")
                fill_value = f["/ionosphere/TEC"].attrs.get("_FillValue", FILL_VALUE)
                print(f"Fill value: {fill_value}")
                tec_mask = tec_data == fill_value
                tec_data[tec_mask] = np.nan
                tec_data_flat = tec_data.reshape(tec_data.shape[0], -1)

                # 空间天气数据
                print("Reading space weather data for statistics")
                sw_data = {}
                for idx_name in SW_INDICES:
                    dset = f[f"/space_weather_indices/{idx_name}"]
                    data = dset[:]
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

                # 更新累计统计量
                print("Calculating statistics for this year...")
                year_tec_mean = np.nanmean(tec_data_flat, axis=0)
                year_tec_std = np.nanstd(tec_data_flat, axis=0)
                year_sw_mean = np.nanmean(sw_array, axis=0)
                year_sw_std = np.nanstd(sw_array, axis=0)
                year_samples = tec_data.shape[0]

                if tec_means_sum is None:
                    print(f"First year - initializing sums with {year_samples} samples")
                    tec_means_sum = year_tec_mean * year_samples
                    tec_stds_sum = year_tec_std * year_samples
                    sw_means_sum = year_sw_mean * year_samples
                    sw_stds_sum = year_sw_std * year_samples
                else:
                    print(f"Adding statistics from {year_samples} samples")
                    tec_means_sum += year_tec_mean * year_samples
                    tec_stds_sum += year_tec_std * year_samples
                    sw_means_sum += year_sw_mean * year_samples
                    sw_stds_sum += year_sw_std * year_samples

                n_train_samples += year_samples
                print(f"Processed statistics for year {year}, total samples so far: {n_train_samples}")
        except Exception as e:
            print(f"Error calculating statistics for year {year}: {str(e)}")
            import traceback

            traceback.print_exc()

    if n_train_samples == 0:
        raise ValueError("No training data found for computing statistics")

    # 计算加权平均
    print(f"Calculating final statistics from {n_train_samples} total samples")
    tec_means = tec_means_sum / n_train_samples
    tec_stds = tec_stds_sum / n_train_samples
    sw_means = sw_means_sum / n_train_samples
    sw_stds = sw_stds_sum / n_train_samples

    # 确保标准差不为零
    tec_stds[tec_stds < 1e-10] = 1.0
    sw_stds[sw_stds < 1e-10] = 1.0

    # 保存Scaler参数
    scaler_params = {"tec_mean": tec_means, "tec_std": tec_stds, "sw_mean": sw_means, "sw_std": sw_stds}
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler_params, f)
    print(f"Scaler parameters saved to {scaler_path}")

    # --- 2. 分别处理训练、验证和测试集的每一年 ---
    tec_scaler = (tec_means, tec_stds)
    sw_scaler = (sw_means, sw_stds)

    total_train = 0
    total_val = 0
    total_test = 0

    # 处理训练集年份
    print("\n--- Processing training set years ---")
    for year in years_train:
        n_samples = create_and_save_sequences_by_year(year, hdf5_dir, output_dir, p, s, tec_scaler, sw_scaler, "train")
        total_train += n_samples

    # 处理验证集年份
    print("\n--- Processing validation set years ---")
    for year in years_val:
        n_samples = create_and_save_sequences_by_year(year, hdf5_dir, output_dir, p, s, tec_scaler, sw_scaler, "val")
        total_val += n_samples

    # 处理测试集年份
    print("\n--- Processing test set years ---")
    for year in years_test:
        n_samples = create_and_save_sequences_by_year(year, hdf5_dir, output_dir, p, s, tec_scaler, sw_scaler, "test")
        total_test += n_samples

    print(f"Preprocessing finished. Total sequences: train={total_train}, val={total_val}, test={total_test}")


if __name__ == "__main__":
    try:
        # 使用流式处理版本
        print("Starting preprocessing with stream processing...")
        preprocess_tec_data_stream(
            hdf5_dir=HDF5_DIR,
            output_dir=OUTPUT_DIR,
            scaler_path=SCALER_PATH,
            years_all=YEARS_ALL,
            years_train=YEARS_TRAIN,
            years_val=YEARS_VAL,
            years_test=YEARS_TEST,
            p=HISTORY_LEN,
            s=FORECAST_LEN,
        )
    except Exception as e:
        print(f"Error in main routine: {str(e)}")
        import traceback

        traceback.print_exc()
