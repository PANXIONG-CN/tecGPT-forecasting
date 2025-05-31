import h5py
import numpy as np
import pandas as pd
from datetime import datetime
import math
import os
import pickle
from tqdm import tqdm
import gc  # 添加垃圾回收
import hydra
from omegaconf import DictConfig, OmegaConf

# --- 填充值 ---
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


def get_year_month_ranges(cfg):
    """从配置中获取年月范围"""
    # 训练集
    train_years = []
    for year in range(cfg.dataset.train_start_year, cfg.dataset.train_end_year + 1):
        start_month = cfg.dataset.train_start_month if year == cfg.dataset.train_start_year else 1
        end_month = cfg.dataset.train_end_month if year == cfg.dataset.train_end_year else 12
        train_years.extend([(year, month) for month in range(start_month, end_month + 1)])

    # 验证集
    val_years = []
    for year in range(cfg.dataset.val_start_year, cfg.dataset.val_end_year + 1):
        start_month = cfg.dataset.val_start_month if year == cfg.dataset.val_start_year else 1
        end_month = cfg.dataset.val_end_month if year == cfg.dataset.val_end_year else 12
        val_years.extend([(year, month) for month in range(start_month, end_month + 1)])

    # 测试集
    test_years = []
    for year in range(cfg.dataset.test_start_year, cfg.dataset.test_end_year + 1):
        start_month = cfg.dataset.test_start_month if year == cfg.dataset.test_start_year else 1
        end_month = cfg.dataset.test_end_month if year == cfg.dataset.test_end_year else 12
        test_years.extend([(year, month) for month in range(start_month, end_month + 1)])

    return train_years, val_years, test_years


def process_year_data(year, hdf5_dir, cfg):
    """处理单年份数据"""
    filepath = os.path.join(hdf5_dir, f"CRIM_SW2hr_AI_v1.2_{year}_DataDrivenRange_CN.hdf5")
    if not os.path.exists(filepath):
        print(f"Warning: File not found for year {year}: {filepath}")
        return None, None, None, None

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
            time_components = {
                "year": f["/coordinates/year"][:],
                "month": f["/coordinates/month"][:],
                "day": f["/coordinates/day"][:],
                "hour": f["/coordinates/hour"][:],
                "day_of_year": f["/coordinates/day_of_year"][:],
            }
            df_time = pd.DataFrame(time_components)

            # 3. 加载空间天气指数
            sw_indices = ["Kp_Index", "Dst_Index", "ap_Index", "F107_Index", "AE_Index"]
            sw_data = {}
            for idx_name in sw_indices:
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
            sw_array = df_sw[sw_indices].values

            # 4. 生成周期性时间特征（如果启用）
            if cfg.dataset.get("use_time_features", True):
                time_features_array = generate_cyclical_features(df_time)
            else:
                # 创建空的时间特征数组
                num_timesteps = len(df_time)
                time_features_array = np.empty((num_timesteps, 0), dtype=np.float32)

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
    C_time = time_features.shape[1]  # 如果use_time_features为false，这将是0
    C_in = 1 + C_sw + C_time

    X = np.zeros((n_samples, p, N, C_in), dtype=np.float32)
    Y = np.zeros((n_samples, s, N, 1), dtype=np.float32)

    for i in range(n_samples):
        # TEC特征
        X[i, :, :, 0] = tec_scaled[i : i + p, :]

        # SW特征 (广播到所有节点)
        current_feature_idx = 1
        for j in range(C_sw):
            X[i, :, :, current_feature_idx + j] = sw_scaled[i : i + p, j : j + 1]
        current_feature_idx += C_sw

        # 时间特征 (广播到所有节点) - 仅当C_time > 0时
        if C_time > 0:
            for j in range(C_time):
                X[i, :, :, current_feature_idx + j] = time_features[i : i + p, j : j + 1]

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


def generate_dataset_version_id(cfg):
    """生成数据集版本标识符"""
    train_start_short = str(cfg.dataset.train_start_year)[-2:]
    train_end_short = str(cfg.dataset.train_end_year)[-2:]
    val_start_short = str(cfg.dataset.val_start_year)[-2:]
    val_end_short = str(cfg.dataset.val_end_year)[-2:]
    test_start_short = str(cfg.dataset.test_start_year)[-2:]
    test_end_short = str(cfg.dataset.test_end_year)[-2:]

    time_feat_flag = "tf_on" if cfg.dataset.get("use_time_features", True) else "tf_off"

    dataset_version_id = f"tr{train_start_short}-{train_end_short}_v{val_start_short}-{val_end_short}_t{test_start_short}-{test_end_short}_h{cfg.dataset.history_len}_f{cfg.dataset.forecast_len}_{time_feat_flag}"

    return dataset_version_id


def generate_specific_config_file(cfg, dataset_version_id, unique_output_subdir):
    """生成特定的数据集配置文件"""
    # 创建指向特定预处理数据版本的配置
    specific_config = OmegaConf.create(
        {
            "data_dir": f"./processed_tec_data/{dataset_version_id}/",
            "scaler_path": f"./processed_tec_data/{dataset_version_id}/scaler.pkl",
            "num_nodes": cfg.dataset.num_nodes,
            "n_lat": cfg.dataset.n_lat,
            "n_lon": cfg.dataset.n_lon,
            "history_len": cfg.dataset.history_len,
            "forecast_len": cfg.dataset.forecast_len,
            "use_time_features": cfg.dataset.get("use_time_features", True),
            "tec_feat_dim": cfg.dataset.tec_feat_dim,
            "sw_feat_dim": cfg.dataset.sw_feat_dim,
            "time_feat_dim": cfg.dataset.time_feat_dim,
            "train_start_year": cfg.dataset.train_start_year,
            "train_start_month": cfg.dataset.train_start_month,
            "train_end_year": cfg.dataset.train_end_year,
            "train_end_month": cfg.dataset.train_end_month,
            "val_start_year": cfg.dataset.val_start_year,
            "val_start_month": cfg.dataset.val_start_month,
            "val_end_year": cfg.dataset.val_end_year,
            "val_end_month": cfg.dataset.val_end_month,
            "test_start_year": cfg.dataset.test_start_year,
            "test_start_month": cfg.dataset.test_start_month,
            "test_end_year": cfg.dataset.test_end_year,
            "test_end_month": cfg.dataset.test_end_month,
        }
    )

    # 保存特定配置文件
    specific_config_path = f"conf/dataset/tec_data_specific_{dataset_version_id}.yaml"

    # 添加注释头
    config_content = f"""# conf/dataset/tec_data_specific_{dataset_version_id}.yaml
# 此配置文件指向特定的预处理数据版本
# 由预处理脚本自动生成，请勿手动编辑

"""
    config_content += OmegaConf.to_yaml(specific_config)

    with open(specific_config_path, "w") as f:
        f.write(config_content)

    print(f"生成特定配置文件: {specific_config_path}")
    return specific_config_path


@hydra.main(config_path="../conf", config_name="config_preprocess", version_base=None)
def main(cfg: DictConfig) -> None:
    print("=== TEC数据预处理开始 (Hydra版本) ===")
    print(f"基础配置信息:\n{OmegaConf.to_yaml(cfg)}")

    # 生成唯一的数据集版本标识符
    dataset_version_id = generate_dataset_version_id(cfg)
    unique_output_subdir = os.path.join(cfg.dataset.data_dir, dataset_version_id)
    os.makedirs(unique_output_subdir, exist_ok=True)

    scaler_path = os.path.join(unique_output_subdir, "scaler.pkl")

    print(f"数据将被处理并保存到: {unique_output_subdir}")
    print(f"数据集版本标识符: {dataset_version_id}")

    # 生成特定的配置文件
    specific_config_path = generate_specific_config_file(cfg, dataset_version_id, unique_output_subdir)

    # 获取年月范围
    train_years, val_years, test_years = get_year_month_ranges(cfg)

    print(f"训练集年月范围: {len(train_years)} 月 ({train_years[:3]}...{train_years[-3:]})")
    print(f"验证集年月范围: {len(val_years)} 月 ({val_years[:3]}...{val_years[-3:]})")
    print(f"测试集年月范围: {len(test_years)} 月 ({test_years[:3]}...{test_years[-3:]})")

    # 获取所有需要的年份（用于处理HDF5文件）
    all_years = set()
    for year_month_list in [train_years, val_years, test_years]:
        for year, month in year_month_list:
            all_years.add(year)
    all_years = sorted(list(all_years))

    print(f"需要处理的年份: {all_years}")

    # 检查scaler是否已存在
    if os.path.exists(scaler_path):
        print("Scaler already exists, loading...")
        with open(scaler_path, "rb") as f:
            scalers = pickle.load(f)
            tec_scaler = scalers["tec_scaler"]
            sw_scaler = scalers["sw_scaler"]
    else:
        # 1. 收集所有训练集数据用于计算标准化参数
        print("Step 1: 计算标准化参数...")
        all_tec_train = []
        all_sw_train = []

        train_years_set = set([year for year, month in train_years])
        for year in train_years_set:
            tec_flat, tec_orig, sw_array, time_feat = process_year_data(year, cfg.hdf5_dir, cfg)
            if tec_flat is not None:
                # 这里我们简单地使用整年的数据来训练scaler
                # 在实际应用中，您可能希望进一步过滤到特定月份
                all_tec_train.append(tec_flat)
                all_sw_train.append(sw_array)

        if not all_tec_train:
            raise ValueError("No training data found!")

        # 合并所有训练数据
        tec_train_combined = np.concatenate(all_tec_train, axis=0)
        sw_train_combined = np.concatenate(all_sw_train, axis=0)

        # 拟合标准化器
        tec_scaler = NodeScaler(cfg.dataset.num_nodes, fill_value=FILL_VALUE)
        tec_scaler.fit(tec_train_combined)

        sw_scaler = FeatureScaler(cfg.dataset.sw_feat_dim)
        sw_scaler.fit(sw_train_combined)

        # 保存标准化器
        with open(scaler_path, "wb") as f:
            pickle.dump({"tec_scaler": tec_scaler, "sw_scaler": sw_scaler}, f)
        print(f"Scaler saved to {scaler_path}")

    # 2. 处理各个数据集
    splits_to_process = [("train", train_years), ("val", val_years), ("test", test_years)]

    for split_name, year_month_list in splits_to_process:
        print(f"\nStep 2: 处理{split_name}数据集...")

        X_all_split = []
        Y_all_split = []

        # 按年份组织数据处理
        years_in_split = set([year for year, month in year_month_list])
        for year in sorted(years_in_split):
            tec_flat, tec_orig, sw_array, time_feat = process_year_data(year, cfg.hdf5_dir, cfg)
            if tec_flat is None:
                continue

            # 应用标准化
            tec_scaled = tec_scaler.transform(tec_flat)
            sw_scaled = sw_scaler.transform(sw_array)

            # 创建序列
            X_year, Y_year = create_sequences(tec_scaled, tec_orig, sw_scaled, time_feat, cfg.dataset.history_len, cfg.dataset.forecast_len)

            if X_year is not None:
                X_all_split.append(X_year)
                Y_all_split.append(Y_year)
                print(f"  Year {year}: {X_year.shape[0]} sequences created")

            # 清理中间变量
            del tec_flat, tec_orig, sw_array, time_feat, tec_scaled, sw_scaled
            if X_year is not None:
                del X_year, Y_year
            gc.collect()

        if X_all_split:
            # 合并所有年份数据
            print(f"  合并{split_name}数据...")
            X_combined = np.concatenate(X_all_split, axis=0)
            Y_combined = np.concatenate(Y_all_split, axis=0)

            # 清理分割列表
            del X_all_split, Y_all_split
            gc.collect()

            # 保存最终数据集
            output_path = os.path.join(unique_output_subdir, f"{split_name}.npz")
            save_final_dataset(X_combined, Y_combined, output_path)

            # 清理合并数据
            del X_combined, Y_combined
            gc.collect()
        else:
            print(f"  Warning: No data found for {split_name} split")

    # 保存数据集生成配置的快照（将模板参数替换为具体值）
    config_snapshot_path = os.path.join(unique_output_subdir, "dataset_generation_params.yaml")

    # 创建配置副本并替换模板参数为具体值
    config_snapshot = OmegaConf.create(OmegaConf.to_yaml(cfg))

    # 替换模板参数为具体值
    config_snapshot.dataset.scaler_path = os.path.join(unique_output_subdir, "scaler.pkl")
    config_snapshot.dataset.version_id = dataset_version_id

    # 移除模板字段
    if "scaler_path_template" in config_snapshot.dataset:
        del config_snapshot.dataset.scaler_path_template
    if "version_id_template" in config_snapshot.dataset:
        del config_snapshot.dataset.version_id_template

    with open(config_snapshot_path, "w") as f:
        OmegaConf.save(config=config_snapshot, f=f)
    print(f"数据集生成参数保存到: {config_snapshot_path}")

    print("\n=== 数据预处理完成 ===")
    print("生成的文件：")
    for filename in ["scaler.pkl", "train.npz", "val.npz", "test.npz"]:
        filepath = os.path.join(unique_output_subdir, filename)
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"  {filename}: {size_mb:.1f} MB")

    print(f"\n--- 预处理完成总结 ---")
    print(f"生成的数据集版本: {dataset_version_id}")
    print(f"数据保存到: {os.path.abspath(unique_output_subdir)}")
    print(f"缩放器保存到: {os.path.abspath(scaler_path)}")
    print(f"特定配置文件: {os.path.abspath(specific_config_path)}")
    print(f"--- 在训练时使用配置: dataset=tec_data_specific_{dataset_version_id} ---")

    # 清理内存
    print("\n=== 清理内存 ===")
    try:
        # 清理可能存在的大型变量
        if "tec_scaler" in locals():
            del tec_scaler
        if "sw_scaler" in locals():
            del sw_scaler

        # 强制垃圾回收
        gc.collect()

        # 如果在CUDA环境中，清理GPU内存
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("GPU内存已清理")
        except ImportError:
            pass

        print("内存清理完成")
    except Exception as e:
        print(f"内存清理过程中出现警告: {e}")


if __name__ == "__main__":
    main()
