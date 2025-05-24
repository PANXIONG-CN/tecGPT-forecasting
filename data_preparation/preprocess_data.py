import h5py
import numpy as np
import pandas as pd
from datetime import datetime
import math
import os
import pickle
from tqdm import tqdm
import argparse

# --- 配置参数 ---
YEARS_ALL = list(range(2013, 2026))  # 数据覆盖的所有年份 (2013-2025)
YEARS_TRAIN = list(range(2013, 2020))  # 2013-2019
YEARS_VAL = list(range(2020, 2022))  # 2020-2021
YEARS_TEST = list(range(2022, 2026))  # 2022-2025

HISTORY_LEN = 12  # P: 使用过去 12 个时间步 (24 小时)
FORECAST_LEN = 12  # S: 预测未来 12 个时间步 (24 小时)

N_LAT = 41
N_LON = 71
N_NODES = N_LAT * N_LON

# --- 特征索引和名称 ---
SW_INDICES = ["Kp_Index", "Dst_Index", "ap_Index", "F107_Index", "AE_Index"]
N_SW_INDICES = len(SW_INDICES)
TIME_COMPONENTS = ["year", "month", "day", "hour", "day_of_year"]

# 填充值
FILL_VALUE = -9999.0


class NodeScaler:
    """节点级标准化器，每个节点独立标准化"""

    def __init__(self, n_nodes, fill_value=None):
        self.n_nodes = n_nodes
        self.fill_value = fill_value
        self.mean_ = None
        self.std_ = None

    def fit(self, data):
        """
        拟合标准化参数
        data: shape [..., n_nodes]
        """
        # Reshape to (samples, n_nodes)
        original_shape = data.shape
        data_reshaped = data.reshape(-1, self.n_nodes)

        self.mean_ = np.zeros(self.n_nodes)
        self.std_ = np.ones(self.n_nodes)

        for i in range(self.n_nodes):
            node_data = data_reshaped[:, i]
            if self.fill_value is not None:
                valid_mask = node_data != self.fill_value
                if valid_mask.sum() > 0:
                    self.mean_[i] = np.mean(node_data[valid_mask])
                    self.std_[i] = np.std(node_data[valid_mask])
                    if self.std_[i] < 1e-8:
                        self.std_[i] = 1.0
            else:
                self.mean_[i] = np.mean(node_data)
                self.std_[i] = np.std(node_data)
                if self.std_[i] < 1e-8:
                    self.std_[i] = 1.0

    def transform(self, data):
        """标准化数据"""
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler has not been fitted yet.")

        original_shape = data.shape
        data_reshaped = data.reshape(-1, self.n_nodes)
        transformed = np.zeros_like(data_reshaped)

        for i in range(self.n_nodes):
            node_data = data_reshaped[:, i]
            if self.fill_value is not None:
                valid_mask = node_data != self.fill_value
                transformed[valid_mask, i] = (node_data[valid_mask] - self.mean_[i]) / self.std_[i]
                transformed[~valid_mask, i] = self.fill_value
            else:
                transformed[:, i] = (node_data - self.mean_[i]) / self.std_[i]

        return transformed.reshape(original_shape)

    def inverse_transform(self, data):
        """逆标准化数据"""
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler has not been fitted yet.")

        original_shape = data.shape
        data_reshaped = data.reshape(-1, self.n_nodes)
        inverted = np.zeros_like(data_reshaped)

        for i in range(self.n_nodes):
            node_data = data_reshaped[:, i]
            if self.fill_value is not None:
                valid_mask = node_data != self.fill_value
                inverted[valid_mask, i] = (node_data[valid_mask] * self.std_[i]) + self.mean_[i]
                inverted[~valid_mask, i] = self.fill_value
            else:
                inverted[:, i] = (node_data * self.std_[i]) + self.mean_[i]

        return inverted.reshape(original_shape)


class FeatureScaler:
    """特征级标准化器，使用sklearn兼容接口"""

    def __init__(self, n_features):
        self.n_features = n_features
        self.mean_ = None
        self.std_ = None

    def fit(self, data):
        """拟合标准化参数"""
        if data.shape[-1] != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {data.shape[-1]}")

        data_reshaped = data.reshape(-1, self.n_features)
        self.mean_ = np.mean(data_reshaped, axis=0)
        self.std_ = np.std(data_reshaped, axis=0)
        self.std_[self.std_ < 1e-8] = 1.0

    def transform(self, data):
        """标准化数据"""
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler has not been fitted yet.")

        original_shape = data.shape
        data_reshaped = data.reshape(-1, self.n_features)
        transformed = (data_reshaped - self.mean_) / self.std_
        return transformed.reshape(original_shape)

    def inverse_transform(self, data):
        """逆标准化数据"""
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler has not been fitted yet.")

        original_shape = data.shape
        data_reshaped = data.reshape(-1, self.n_features)
        inverted = (data_reshaped * self.std_) + self.mean_
        return inverted.reshape(original_shape)


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


def process_year_data(year, hdf5_dir):
    """处理单年份数据"""
    filepath = os.path.join(hdf5_dir, f"CRIM_SW2hr_AI_v1.2_{year}_DataDrivenRange_CN.hdf5")
    if not os.path.exists(filepath):
        print(f"Warning: File not found for year {year}: {filepath}")
        return None, None, None

    print(f"Processing year {year}...")

    try:
        with h5py.File(filepath, "r") as f:
            # 1. 加载 TEC 数据
            tec_data = f["/ionosphere/TEC"][:]
            fill_value = f["/ionosphere/TEC"].attrs.get("_FillValue", FILL_VALUE)
            tec_mask = tec_data == fill_value
            tec_data_original = tec_data.copy()
            tec_data[tec_mask] = np.nan
            tec_data_flat = tec_data.reshape(tec_data.shape[0], -1)

            # 2. 加载时间组件
            time_components = {comp: f[f"/coordinates/{comp}"][:] for comp in TIME_COMPONENTS}
            df_time = pd.DataFrame(time_components)

            # 3. 加载空间天气指数
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

            # 4. 生成周期性时间特征
            time_features_array = generate_cyclical_features(df_time)

            return tec_data_flat, tec_data_original.reshape(tec_data_original.shape[0], -1), sw_array, time_features_array

    except Exception as e:
        print(f"Error processing year {year}: {str(e)}")
        return None, None, None, None


def create_sequences(tec_scaled, tec_original, sw_scaled, time_features, p, s):
    """创建滑动窗口序列"""
    n_samples = tec_scaled.shape[0] - p - s + 1
    if n_samples <= 0:
        return None, None

    N = tec_scaled.shape[1]
    C_sw = sw_scaled.shape[1]
    C_time = time_features.shape[1]
    C_in = 1 + C_sw + C_time

    X = np.zeros((n_samples, p, N, C_in), dtype=np.float32)
    Y = np.zeros((n_samples, s, N, 1), dtype=np.float32)

    for i in range(n_samples):
        # TEC特征
        X[i, :, :, 0] = tec_scaled[i : i + p, :]

        # SW特征 (广播到所有节点)
        for j in range(C_sw):
            X[i, :, :, 1 + j] = sw_scaled[i : i + p, j : j + 1]

        # 时间特征 (广播到所有节点)
        for j in range(C_time):
            X[i, :, :, 1 + C_sw + j] = time_features[i : i + p, j : j + 1]

        # 目标值 (原始尺度)
        Y[i, :, :, 0] = tec_original[i + p : i + p + s, :]

    return X, Y


def save_final_dataset(X_all, Y_all, output_path):
    """保存最终合并的数据集"""
    print(f"Saving final dataset to {output_path}")
    print(f"X shape: {X_all.shape}, Y shape: {Y_all.shape}")

    # 使用压缩保存
    np.savez_compressed(output_path, x=X_all, y=Y_all)
    print(f"Dataset saved successfully to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="TEC Data Preprocessing")
    parser.add_argument("--hdf5_dir", type=str, default="data_preparation", help="Directory containing HDF5 files")
    parser.add_argument("--output_dir", type=str, default="./processed_tec_data", help="Output directory for processed data")
    parser.add_argument("--batch_process", action="store_true", help="Process and save in batches (for large datasets)")

    args = parser.parse_args()

    HDF5_DIR = args.hdf5_dir
    OUTPUT_DIR = args.output_dir
    SCALER_PATH = os.path.join(OUTPUT_DIR, "scaler.pkl")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== TEC数据预处理开始 ===")

    # 1. 收集所有训练集数据用于计算标准化参数
    print("Step 1: 计算标准化参数...")
    all_tec_train = []
    all_sw_train = []

    for year in YEARS_TRAIN:
        tec_flat, tec_orig, sw_array, time_feat = process_year_data(year, HDF5_DIR)
        if tec_flat is not None:
            all_tec_train.append(tec_flat)
            all_sw_train.append(sw_array)

    if not all_tec_train:
        raise ValueError("No training data found!")

    # 合并所有训练数据
    tec_train_combined = np.concatenate(all_tec_train, axis=0)
    sw_train_combined = np.concatenate(all_sw_train, axis=0)

    # 拟合标准化器
    tec_scaler = NodeScaler(N_NODES, fill_value=FILL_VALUE)
    tec_scaler.fit(tec_train_combined)

    sw_scaler = FeatureScaler(N_SW_INDICES)
    sw_scaler.fit(sw_train_combined)

    # 保存标准化器
    with open(SCALER_PATH, "wb") as f:
        pickle.dump({"tec_scaler": tec_scaler, "sw_scaler": sw_scaler}, f)
    print(f"Scaler saved to {SCALER_PATH}")

    # 2. 处理各个数据集
    for split_name, years in [("train", YEARS_TRAIN), ("val", YEARS_VAL), ("test", YEARS_TEST)]:
        print(f"\nStep 2: 处理{split_name}数据集...")

        X_all_split = []
        Y_all_split = []

        for year in years:
            tec_flat, tec_orig, sw_array, time_feat = process_year_data(year, HDF5_DIR)
            if tec_flat is None:
                continue

            # 应用标准化
            tec_scaled = tec_scaler.transform(tec_flat)
            sw_scaled = sw_scaler.transform(sw_array)

            # 创建序列
            X_year, Y_year = create_sequences(tec_scaled, tec_orig, sw_scaled, time_feat, HISTORY_LEN, FORECAST_LEN)

            if X_year is not None:
                X_all_split.append(X_year)
                Y_all_split.append(Y_year)
                print(f"  Year {year}: {X_year.shape[0]} sequences created")

        if X_all_split:
            # 合并所有年份数据
            X_combined = np.concatenate(X_all_split, axis=0)
            Y_combined = np.concatenate(Y_all_split, axis=0)

            # 保存最终数据集
            output_path = os.path.join(OUTPUT_DIR, f"{split_name}.npz")
            save_final_dataset(X_combined, Y_combined, output_path)
        else:
            print(f"  Warning: No data found for {split_name} split")

    print("\n=== 数据预处理完成 ===")
    print("生成的文件：")
    for filename in ["scaler.pkl", "train.npz", "val.npz", "test.npz"]:
        filepath = os.path.join(OUTPUT_DIR, filename)
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"  {filename}: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
