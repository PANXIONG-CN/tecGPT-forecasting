import numpy as np
import os
import torch
import pickle
from tqdm import tqdm  # 假设你可能在其他地方用


class DataLoader(object):
    def __init__(self, data_dict=None, batch_size=32, shuffle=False):
        """
        数据加载器初始化
        Args:
            data_dict: 包含'x'和'y'键的字典，或者直接传入x和y数组
            batch_size: 批处理大小
            shuffle: 是否打乱数据
        """
        self.batch_size = batch_size
        self.current_ind = 0
        self.should_shuffle = shuffle

        # 处理输入数据
        if isinstance(data_dict, dict) and "x" in data_dict and "y" in data_dict:
            self.xs_orig = np.array(data_dict["x"])
            self.ys_orig = np.array(data_dict["y"])
        else:
            raise ValueError("data_dict必须是包含'x'和'y'键的字典")

        # 复制数据用于处理
        self.xs = self.xs_orig.copy()
        self.ys = self.ys_orig.copy()

        # 计算填充
        self.size = len(self.xs)
        if self.size > 0:
            num_padding = (batch_size - (self.size % batch_size)) % batch_size
            if num_padding > 0:
                x_padding = np.repeat(self.xs[-1:], num_padding, axis=0)
                y_padding = np.repeat(self.ys[-1:], num_padding, axis=0)
                self.xs = np.concatenate([self.xs, x_padding], axis=0)
                self.ys = np.concatenate([self.ys, y_padding], axis=0)
                self.size = len(self.xs)

        self.num_batch = int(self.size // self.batch_size) if self.size > 0 else 0

    def shuffle_data(self):
        """打乱当前数据副本，而不是原始数据"""
        if self.size > 0:
            permutation = np.random.permutation(self.size)
            self.xs = self.xs[permutation]
            self.ys = self.ys[permutation]

    def get_iterator(self):
        self.current_ind = 0
        # 只有当should_shuffle为True时才打乱数据（用于训练集）
        if self.should_shuffle:
            self.shuffle_data()

        def _wrapper():
            if self.num_batch == 0:  # 如果没有数据或不够一个批次
                return
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind:end_ind, ...]
                y_i = self.ys[start_ind:end_ind, ...]
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


def load_from_files(x_file_path, y_file_path):
    """从npz文件加载x和y数据"""
    try:
        x_data = np.load(x_file_path)
        y_data = np.load(y_file_path)
        # 假设npz文件中的数组名为"arr_0"，这是np.save默认行为
        x = x_data["arr_0"] if "arr_0" in x_data else x_data["x"]
        y = y_data["arr_0"] if "arr_0" in y_data else y_data["y"]
        return {"x": x, "y": y}
    except FileNotFoundError:
        raise FileNotFoundError(f"数据文件不存在: {x_file_path} 或 {y_file_path}")
    except Exception as e:
        raise Exception(f"加载数据时出错: {e}")


def load_scaler(scaler_path):
    """加载标准化器"""
    return StandardScaler(scaler_path=scaler_path)


def load_dataset(dataset_dir, scaler_path, batch_size=32, target_device=None, load_test=True):
    """Load the preprocessed TEC dataset."""
    print(f"Loading preprocessed data from: {dataset_dir}")

    # 加载训练数据
    try:
        train_data = np.load(os.path.join(dataset_dir, "train.npz"))
        train_x = train_data["x"]
        train_y = train_data["y"]
        print(f"Loaded train data: x shape {train_x.shape}, y shape {train_y.shape}")
    except Exception as e:
        raise Exception(f"训练数据加载失败: {e}")

    # 加载验证数据
    try:
        val_data = np.load(os.path.join(dataset_dir, "val.npz"))
        val_x = val_data["x"]
        val_y = val_data["y"]
        print(f"Loaded val data: x shape {val_x.shape}, y shape {val_y.shape}")
    except Exception as e:
        raise Exception(f"验证数据加载失败: {e}")

    # 有条件地加载测试数据
    test_x, test_y = None, None
    if load_test:
        try:
            test_data = np.load(os.path.join(dataset_dir, "test.npz"))
            test_x = test_data["x"]
            test_y = test_data["y"]
            print(f"Loaded test data: x shape {test_x.shape}, y shape {test_y.shape}")
        except Exception as e:
            print(f"Warning: 测试数据加载失败: {e}")

    # 加载标准化器
    scaler = StandardScaler(scaler_path=scaler_path)

    # 创建数据加载器
    train_loader = DataLoader({"x": train_x, "y": train_y}, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader({"x": val_x, "y": val_y}, batch_size=batch_size, shuffle=False)

    test_loader = None
    if load_test and test_x is not None and test_y is not None:
        test_loader = DataLoader({"x": test_x, "y": test_y}, batch_size=batch_size, shuffle=False)

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
