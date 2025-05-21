import numpy as np
import os
import scipy.sparse as sp
import torch
import pickle


class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        self.batch_size = batch_size
        self.current_ind = 0

        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)

        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind:end_ind, ...]
                y_i = self.ys[start_ind:end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()


class StandardScaler:
    def __init__(self, mean, std, device="cpu"):
        self.np_mean = np.asarray(mean, dtype=np.float32)
        self.np_std = np.asarray(std, dtype=np.float32)

        self.tensor_mean = torch.tensor(mean, dtype=torch.float32, device=device).view(1, 1, -1, 1)
        self.tensor_std = torch.tensor(std, dtype=torch.float32, device=device).view(1, 1, -1, 1)

    def inverse_transform(self, data):
        """处理不同设备的数据"""
        if isinstance(data, torch.Tensor):
            return data * self.tensor_std + self.tensor_mean
        elif isinstance(data, np.ndarray):
            return data * self.np_std + self.np_mean
        raise TypeError("输入类型必须是torch.Tensor或numpy.ndarray")


def load_dataset(dataset_dir, batch_size, valid_batch_size=None, test_batch_size=None, target_device="cuda"):
    data = {}
    for category in ["train", "val", "test"]:
        cat_data = np.load(os.path.join(dataset_dir, category + ".npz"))

        data["x_" + category] = np.array(cat_data["x"], dtype=np.float32)  # (B, T, N, 5)
        data["y_" + category] = np.array(cat_data["y"], dtype=np.float32)  # (B, T, N, 1)

        data["x_" + category] = np.transpose(data["x_" + category], (0, 3, 2, 1))  # (B, 5, N, T)
        data["y_" + category] = np.transpose(data["y_" + category], (0, 3, 2, 1))  # (B, 1, N, T)

    TEC_INDEX = 0
    train_tec = data["x_train"][:, TEC_INDEX, :, :]  # [B, N, T]
    node_means = np.mean(train_tec, axis=(0, 2))  # [N]
    node_stds = np.std(train_tec, axis=(0, 2))  # [N]

    for category in ["train", "val", "test"]:
        x = data["x_" + category].copy()
        y = data["y_" + category].copy()

        for node_idx in range(x.shape[2]):
            mean = node_means[node_idx]
            std = node_stds[node_idx]
            x[:, TEC_INDEX, node_idx, :] = (x[:, TEC_INDEX, node_idx, :] - mean) / std

        y = (y - node_means.reshape(1, 1, -1, 1)) / node_stds.reshape(1, 1, -1, 1)

        data["x_" + category] = x
        data["y_" + category] = y

    data["train_loader"] = DataLoader(data["x_train"], data["y_train"], batch_size)
    data["val_loader"] = DataLoader(data["x_val"], data["y_val"], valid_batch_size or batch_size)
    data["test_loader"] = DataLoader(data["x_test"], data["y_test"], test_batch_size or batch_size)

    device = target_device if torch.cuda.is_available() else "cpu"
    data["scaler"] = StandardScaler(mean=node_means, std=node_stds, device=device)

    return data


def MAE_torch(pred, true, mask_value=None):
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(true - pred))


def MAPE_torch(pred, true, mask_value=None):
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs((true - pred) / true))


def RMSE_torch(pred, true, mask_value=None):
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.sqrt(torch.mean((pred - true) ** 2))


def WMAPE_torch(pred, true, mask_value=None):
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.sum(torch.abs(pred - true)) / torch.sum(torch.abs(true))


def metric(pred, real):
    mae = MAE_torch(pred, real, 0).item()
    mape = MAPE_torch(pred, real, 0).item()
    rmse = RMSE_torch(pred, real, 0).item()
    wmape = WMAPE_torch(pred, real, 0).item()
    return mae, mape, rmse, wmape


def load_graph_data(pkl_filename):
    """加载图数据，需要确保适配2911个节点的图结构"""
    try:
        with open(pkl_filename, "rb") as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError:
        with open(pkl_filename, "rb") as f:
            pickle_data = pickle.load(f, encoding="latin1")
    except Exception as e:
        print("Unable to load graph data:", e)
        raise

    if len(pickle_data) == 3:
        return pickle_data
    else:
        raise ValueError("Graph data format incorrect. Expected (sensor_ids, sensor_id_to_ind, adj_mx)")
