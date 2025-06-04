import numpy as np
import os
import torch
import pickle
from tqdm import tqdm  # 假设你可能在其他地方用


class DummyScaler:
    """A dummy scaler for testing purposes, mimicking NodeScaler/FeatureScaler structure."""

    def __init__(self, n_features_or_nodes):
        self.mean_ = np.zeros(n_features_or_nodes, dtype=np.float32)
        self.std_ = np.ones(n_features_or_nodes, dtype=np.float32)
        if isinstance(n_features_or_nodes, int) and n_features_or_nodes > 100:
            self.scaler_type = "node_scaler_dummy"
        else:
            self.scaler_type = "feature_scaler_dummy"

    def fit(self, data):
        """Dummy method, does nothing."""
        pass

    def transform(self, data):
        """Dummy transform."""
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler has not been fitted yet.")
        return (data - self.mean_) / self.std_

    def inverse_transform(self, data):
        """Dummy inverse transform."""
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler has not been fitted yet.")
        return data * self.std_ + self.mean_


class DataLoader(object):
    def __init__(self, data_dict=None, batch_size=32, shuffle=False):
        """
        内存映射数据加载器初始化
        Args:
            data_dict: 包含'x'和'y'键的字典，支持内存映射对象
            batch_size: 批处理大小
            shuffle: 是否打乱数据
        """
        self.batch_size = batch_size
        self.current_ind = 0
        self.should_shuffle = shuffle

        # 处理输入数据 - 支持内存映射对象
        if isinstance(data_dict, dict) and "x" in data_dict and "y" in data_dict:
            # 直接存储内存映射对象，不复制
            self.xs_orig = data_dict["x"]
            self.ys_orig = data_dict["y"]
        else:
            raise ValueError("data_dict必须是包含'x'和'y'键的字典")

        # 不复制数据，直接引用内存映射对象
        self.xs = self.xs_orig
        self.ys = self.ys_orig

        # 计算原始未填充的数据大小
        self.original_size_unpadded = self.xs_orig.shape[0]

        # 计算需要的填充数量
        self.num_padding = 0
        if self.original_size_unpadded > 0:
            self.num_padding = (batch_size - (self.original_size_unpadded % batch_size)) % batch_size

        # 逻辑大小（包括填充）
        self.size = self.original_size_unpadded + self.num_padding

        # 初始化索引数组用于打乱
        self.raw_indices = np.arange(self.original_size_unpadded)
        self.shuffled_indices = self.raw_indices.copy()

        self.num_batch = int(self.size // self.batch_size) if self.size > 0 else 0

        print(
            f"DataLoader初始化: 原始数据大小={self.original_size_unpadded}, 填充={self.num_padding}, "
            f"逻辑大小={self.size}, 批次数={self.num_batch}"
        )

    def shuffle_data(self):
        """打乱索引而不是直接打乱数据"""
        if self.original_size_unpadded > 0:
            np.random.shuffle(self.shuffled_indices)

    def get_iterator(self):
        self.current_ind = 0
        # 只有当should_shuffle为True时才打乱索引（用于训练集）
        if self.should_shuffle:
            self.shuffle_data()

        def _wrapper():
            if self.num_batch == 0:  # 如果没有数据或不够一个批次
                return
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))

                # 按需从内存映射对象构建批次
                batch_x_list = []
                batch_y_list = []

                for k_logical_idx in range(start_ind, end_ind):
                    if k_logical_idx < self.original_size_unpadded:
                        # 从真实数据获取（使用打乱后的索引）
                        actual_physical_idx = self.shuffled_indices[k_logical_idx]
                        sample_x = self.xs_orig[actual_physical_idx]
                        sample_y = self.ys_orig[actual_physical_idx]
                    else:
                        # 填充数据（使用原始数据的最后一个样本）
                        sample_x = self.xs_orig[self.original_size_unpadded - 1]
                        sample_y = self.ys_orig[self.original_size_unpadded - 1]

                    # 确保转换为内存中的数组
                    batch_x_list.append(np.array(sample_x))
                    batch_y_list.append(np.array(sample_y))

                # 构建批次
                x_i = np.stack(batch_x_list)
                y_i = np.stack(batch_y_list)

                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()


class StandardScaler:
    def __init__(self, scaler_path, device="cpu"):
        try:
            with open(scaler_path, "rb") as f:
                scaler_params = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Scaler file not found at: {scaler_path}. Please run preprocessing first.")

        # scaler_params 字典现在包含 'tec_scaler' (一个 NodeScaler 实例) 和 'sw_scaler' (一个 FeatureScaler 实例)
        tec_node_scaler = scaler_params.get("tec_scaler")
        sw_feature_scaler = scaler_params.get("sw_scaler")

        if tec_node_scaler is None:
            raise ValueError("tec_scaler not found in the loaded scaler file.")

        tec_mean = tec_node_scaler.mean_  # Shape: [N_nodes]
        tec_std = tec_node_scaler.std_  # Shape: [N_nodes]

        # SW 指数 scaler 参数
        self.sw_mean_np = None
        self.sw_std_np = None
        if sw_feature_scaler:
            self.sw_mean_np = sw_feature_scaler.mean_  # Shape: [N_SW_Indices]
            self.sw_std_np = sw_feature_scaler.std_  # Shape: [N_SW_Indices]

        self.device = device
        # 用于 TEC 逆变换 (输入 [B, N, S])
        self.tensor_mean_tec = torch.tensor(tec_mean, dtype=torch.float32, device=self.device).view(1, -1, 1)
        self.tensor_std_tec = torch.tensor(tec_std, dtype=torch.float32, device=self.device).view(1, -1, 1)

        self.np_mean_tec = tec_mean.astype(np.float32)
        self.np_std_tec = tec_std.astype(np.float32)

        print(f"Scaler loaded from {scaler_path}. TEC mean/std shape: {self.np_mean_tec.shape}")
        if self.sw_mean_np is not None:
            print(f"SW mean/std shape: {self.sw_mean_np.shape}")

    def inverse_transform_tec(self, data_tec_scaled):
        """只对 TEC 数据进行逆变换"""
        if not isinstance(data_tec_scaled, torch.Tensor):
            # 如果是 numpy，先转 tensor
            data_tec_scaled = torch.from_numpy(data_tec_scaled).to(self.device).float()

        # 确保设备匹配
        if data_tec_scaled.device.type != self.tensor_mean_tec.device.type:
            self.tensor_mean_tec = self.tensor_mean_tec.to(data_tec_scaled.device)
            self.tensor_std_tec = self.tensor_std_tec.to(data_tec_scaled.device)

        # data_tec_scaled: [B, N_nodes, S_forecast_len]
        if data_tec_scaled.shape[1] != self.tensor_mean_tec.shape[1]:
            raise ValueError(
                f"Dimension mismatch for inverse_transform_tec: data nodes {data_tec_scaled.shape[1]} vs scaler nodes {self.tensor_mean_tec.shape[1]}"
            )

        return data_tec_scaled * self.tensor_std_tec + self.tensor_mean_tec

    # transform_tec 和 transform_sw 方法不再需要，因为标准化在预处理中完成
    # 但可以保留 inverse_transform_sw 如果模型也预测SW指数 (当前场景不需要)


def load_from_files(x_file_path, y_file_path, use_mmap=True):
    """从npz文件加载x和y数据，支持内存映射"""
    try:
        mmap_mode = "r" if use_mmap else None
        x_data = np.load(x_file_path, mmap_mode=mmap_mode)
        y_data = np.load(y_file_path, mmap_mode=mmap_mode)
        # 假设npz文件中的数组名为"arr_0"，这是np.save默认行为
        x = x_data["arr_0"] if "arr_0" in x_data else x_data["x"]
        y = y_data["arr_0"] if "arr_0" in y_data else y_data["y"]
        mmap_info = " (memory mapped)" if use_mmap else ""
        print(f"从文件加载数据{mmap_info}: x shape {x.shape}, y shape {y.shape}")
        return {"x": x, "y": y}
    except FileNotFoundError:
        raise FileNotFoundError(f"数据文件不存在: {x_file_path} 或 {y_file_path}")
    except Exception as e:
        raise Exception(f"加载数据时出错: {e}")


def load_scaler(scaler_path):
    """加载标准化器"""
    return StandardScaler(scaler_path=scaler_path)


def load_dataset(dataset_dir, scaler_path, batch_size=32, target_device=None, load_test=True, load_train_val=True):
    """Load the preprocessed TEC dataset using memory mapping."""
    print(f"Loading preprocessed data from: {dataset_dir} (using memory mapping)")

    train_loader, val_loader = None, None

    # 加载训练和验证数据集（如果需要）
    if load_train_val:
        # 加载训练数据 (使用内存映射)
        try:
            train_data = np.load(os.path.join(dataset_dir, "train.npz"), mmap_mode="r")
            train_x = train_data["x"]
            train_y = train_data["y"]
            print(f"Loaded train data: x shape {train_x.shape}, y shape {train_y.shape} (memory mapped)")
            # 创建训练数据加载器
            train_loader = DataLoader({"x": train_x, "y": train_y}, batch_size=batch_size, shuffle=True)
        except Exception as e:
            raise Exception(f"训练数据加载失败: {e}")

        # 加载验证数据 (使用内存映射)
        try:
            val_data = np.load(os.path.join(dataset_dir, "val.npz"), mmap_mode="r")
            val_x = val_data["x"]
            val_y = val_data["y"]
            print(f"Loaded val data: x shape {val_x.shape}, y shape {val_y.shape} (memory mapped)")
            # 创建验证数据加载器
            val_loader = DataLoader({"x": val_x, "y": val_y}, batch_size=batch_size, shuffle=False)
        except Exception as e:
            raise Exception(f"验证数据加载失败: {e}")

    # 有条件地加载测试数据 (使用内存映射)
    test_loader = None
    if load_test:
        try:
            test_data = np.load(os.path.join(dataset_dir, "test.npz"), mmap_mode="r")
            test_x = test_data["x"]
            test_y = test_data["y"]
            print(f"Loaded test data: x shape {test_x.shape}, y shape {test_y.shape} (memory mapped)")
            # 创建测试数据加载器
            test_loader = DataLoader({"x": test_x, "y": test_y}, batch_size=batch_size, shuffle=False)
        except Exception as e:
            print(f"Warning: 测试数据加载失败: {e}")

    # 加载标准化器
    scaler = StandardScaler(scaler_path=scaler_path)

    return {"train_loader": train_loader, "val_loader": val_loader, "test_loader": test_loader, "scaler": scaler}


# --- 评价指标函数 ---
FILL_VALUE_MASK = -9999.0  # 与预处理脚本中的填充值一致


def MAE_torch(pred, true, mask_value=FILL_VALUE_MASK):
    if mask_value is not None:
        mask = torch.ne(true, mask_value)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
        if pred.numel() == 0:
            return torch.tensor(0.0, device=pred.device)
    return torch.mean(torch.abs(true - pred))


def MAPE_torch(pred, true, mask_value=FILL_VALUE_MASK):
    if mask_value is not None:
        mask = torch.ne(true, mask_value)
        mask = mask & torch.ne(true, 0.0)  # 额外确保真实值不为0
        if mask.sum() == 0:
            return torch.tensor(float("inf"), device=pred.device)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
        if pred.numel() == 0:
            return torch.tensor(float("inf"), device=pred.device)
    return torch.mean(torch.abs(torch.div((true - pred), true))) * 100.0  # 通常乘以100


def RMSE_torch(pred, true, mask_value=FILL_VALUE_MASK):
    if mask_value is not None:
        mask = torch.ne(true, mask_value)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
        if pred.numel() == 0:
            return torch.tensor(0.0, device=pred.device)
    return torch.sqrt(torch.mean((pred - true) ** 2))


def WMAPE_torch(pred, true, mask_value=FILL_VALUE_MASK):
    if mask_value is not None:
        mask = torch.ne(true, mask_value)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        pred_masked = torch.masked_select(pred, mask)
        true_masked = torch.masked_select(true, mask)
        if pred_masked.numel() == 0:
            return torch.tensor(0.0, device=pred.device)
    else:  # 如果没有 mask_value，直接使用原始数据
        pred_masked = pred
        true_masked = true

    sum_abs_true = torch.sum(torch.abs(true_masked))
    if sum_abs_true == 0:
        return torch.tensor(float("inf"), device=pred.device)

    loss = torch.sum(torch.abs(pred_masked - true_masked)) / sum_abs_true
    return loss * 100.0  # 通常乘以100


def metric(pred, real):
    # pred 和 real 都是原始尺度，形状通常是 [B, N, S] 或 [B*N, S] 或 [Samples, N_nodes]
    # 确保它们是 Tensors
    if not isinstance(pred, torch.Tensor):
        pred = torch.from_numpy(np.asarray(pred)).float()
    if not isinstance(real, torch.Tensor):
        real = torch.from_numpy(np.asarray(real)).float()

    if pred.device != real.device:  # 确保设备一致
        pred = pred.to(real.device)

    mae = MAE_torch(pred, real).item()
    mape = MAPE_torch(pred, real).item()
    rmse = RMSE_torch(pred, real).item()
    wmape = WMAPE_torch(pred, real).item()
    return mae, mape, rmse, wmape


# load_graph_data 函数保持不变 (如果其他基线模型需要)
# ...
