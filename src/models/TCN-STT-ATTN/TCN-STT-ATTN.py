import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- 定义模块 ----------------

# 位置编码模块
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (N, T, D)
        return x + self.pe[:, :x.size(1), :]

# TCN模块
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        # x: (N, C, T+padding)
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout):
        super(TemporalBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                     dilation=dilation_size, padding=(kernel_size - 1) * dilation_size,
                                     dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# SpatialAttention层
class SpatialAttention(nn.Module):
    def __init__(self, tec_features, aux_features, time_steps=48):
        super(SpatialAttention, self).__init__()
        self.time_steps = time_steps
        # 可学习的权重矩阵，形状为 (time_steps, aux_features, tec_features)
        self.weights = nn.Parameter(torch.ones(time_steps, aux_features, tec_features))
        # 使用 Xavier 初始化权重
        nn.init.xavier_uniform_(self.weights)

    def forward(self, tec, aux):
        """
        tec: Tensor of shape (batch, time_steps, tec_features)
        aux: Tensor of shape (batch, time_steps, aux_features)
        """
        # 使用 einsum 进行批量矩阵乘法
        # 'bta,taf->btf' 意味着：
        # b: batch
        # t: time_steps
        # a: aux_features
        # f: tec_features
        weighted_aux = torch.einsum('bta,taf->btf', aux, self.weights)  # (batch, T, F)
        # 计算注意力权重
        attention_weights = torch.sigmoid(weighted_aux)  # (batch, T, F)
        # 应用注意力权重到 TEC 特征
        tec_attended = tec * attention_weights  # 元素级相乘 (batch, T, F)
        return tec_attended, attention_weights

# TCN-SpatioTemporalTransformer模型（加入时空注意力机制）
class TCNSpatioTemporalTransformer(nn.Module):
    def __init__(self, tec_features, aux_features, num_channels, transformer_heads, transformer_layers, ff_hidden_size, output_size, dropout=0.2, time_steps=48):
        super(TCNSpatioTemporalTransformer, self).__init__()

        self.tec_features = tec_features
        self.aux_features = aux_features
        self.time_steps = time_steps

        # Spatial Attention with time-specific weights
        self.spatial_attention = SpatialAttention(tec_features, aux_features, time_steps=time_steps)

        # TCN
        self.tcn = TemporalConvNet(tec_features, num_channels, kernel_size=2, dropout=dropout)
        self.tcn_linear = nn.Linear(num_channels[-1], ff_hidden_size)

        # Transformer
        self.pos_encoder = PositionalEncoding(ff_hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=ff_hidden_size, nhead=transformer_heads,
                                                   dim_feedforward=ff_hidden_size, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        # Output Layer
        self.output_linear = nn.Linear(ff_hidden_size, output_size)

    def forward(self, x):
        """
        x: Tensor of shape (batch, time_steps, tec_features + aux_features)
        """
        # 拆分主特征和辅助特征
        tec = x[:, :, :self.tec_features]  # (batch, T, F)
        aux = x[:, :, self.tec_features:]  # (batch, T, A)

        # 应用 Spatial Attention
        tec_attended, attention_weights = self.spatial_attention(tec, aux)  # (batch, T, F), (batch, T, F)

        # Pass through TCN
        # TCN expects (N, C, T), so permute
        tec_attended_permuted = tec_attended.permute(0, 2, 1)  # (batch, F, T)
        tcn_out = self.tcn(tec_attended_permuted)  # (batch, C_out, T)
        tcn_out = tcn_out.permute(0, 2, 1)  # (batch, T, C_out)
        tcn_out = self.tcn_linear(tcn_out)  # (batch, T, ff_hidden_size)

        # Apply Positional Encoding
        tcn_out = self.pos_encoder(tcn_out)  # (batch, T, ff_hidden_size)

        # Prepare for Transformer (needs shape (T, N, D))
        transformer_input = tcn_out.permute(1, 0, 2)  # (T, batch, D)

        # Transformer
        transformer_out = self.transformer_encoder(transformer_input)  # (T, batch, D)
        transformer_out = transformer_out[-1, :, :]  # 取最后一个时间步的输出 (batch, D)

        # Output layer
        output = self.output_linear(transformer_out)  # (batch, output_size)

        return output, attention_weights  # 返回 attention_weights 以便保存

# 数据集类
class TEC_Dataset(Dataset):
    def __init__(self, X, aux, y):
        """
        初始化数据集

        参数:
            X (torch.Tensor): 主特征，形状为 (num_samples, 48, 255)
            aux (torch.Tensor): 辅助特征，形状为 (num_samples, 48, aux_features)
            y (torch.Tensor): 目标变量，形状为 (num_samples, 24 * 255)
        """
        self.X = X
        self.aux = aux
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # X[idx]: (48, 255)
        # aux[idx]: (48, aux_features)
        # 合并特征维度: (48, 255 + aux_features)
        X_merged = torch.cat((self.X[idx], self.aux[idx]), dim=-1)  # (48, 255 + aux_features)
        # y[idx]: (24, 255)，训练时需要展平为 (24*255)
        y_flat = self.y[idx].reshape(-1)  # (24*255,)
        return X_merged, y_flat  # 直接返回张量，避免转换警告

# ---------------- 辅助函数 ----------------

def load_data_with_aux(folder, aux_data, tec_features=255):
    """
    加载数据并与辅助因子合并

    参数:
        folder (str): 数据文件夹路径
        aux_data (pd.DataFrame): 辅助因子数据，索引为datetime
        tec_features (int): TEC特征维度

    返回:
        X (np.ndarray): 主特征，形状为 (num_samples, 48, 255)
        AUX (np.ndarray): 辅助特征，形状为 (num_samples, 48, aux_features)
        Y (np.ndarray): 目标变量，形状为 (num_samples, 24, 255)
        DATES (list): 日期列表
    """
    X, AUX, Y, DATES = [], [], [], []
    for file in sorted(os.listdir(folder)):
        if file.startswith("X_") and file.endswith(".npy"):
            date = file.replace("X_", "").replace(".npy", "")
            DATES.append(date)
            # 加载X
            X_data = np.load(os.path.join(folder, file)).astype(np.float32)  # (2, 6120)
            X_data = X_data.reshape(2, 24, tec_features)  # (2, 24, 255)

            # 加载y
            y_file = file.replace("X_", "y_")
            y_data = np.load(os.path.join(folder, y_file)).astype(np.float32)  # (24, 255)

            # 辅助因子与X对应的48小时数据
            end_time = pd.to_datetime(date)
            start_time = end_time - pd.Timedelta(hours=47)
            date_range = pd.date_range(start=start_time, end=end_time, freq='H')
            try:
                aux_values = aux_data.loc[date_range].values.astype(np.float32)  # (48, aux_features)
            except KeyError as e:
                raise KeyError(f"辅助因子数据中缺少日期范围 {date_range}, 错误: {e}")

            aux_features = aux_values.shape[1]
            aux_values = aux_values.reshape(2, 24, aux_features)  # (2, 24, aux_features)

            X.append(X_data)  # (2, 24, 255)
            AUX.append(aux_values)  # (2, 24, aux_features)
            Y.append(y_data)  # (24, 255)

    X = np.array(X).reshape(-1, 48, tec_features)  # (num_samples, 48, 255)
    AUX = np.array(AUX).reshape(-1, 48, AUX[0].shape[-1])  # (num_samples, 48, aux_features)
    Y = np.array(Y)  # (num_samples, 24, 255)
    return X, AUX, Y, DATES

def standardize_data(X_train, aux_train, y_train, X_val, aux_val, y_val, X_test, aux_test, y_test):
    """
    标准化数据

    参数:
        X_train, aux_train, y_train: 训练数据
        X_val, aux_val, y_val: 验证数据
        X_test, aux_test, y_test: 测试数据

    返回:
        标准化后的数据及对应的scaler对象
    """
    scaler_X = StandardScaler()
    scaler_aux = StandardScaler()
    scaler_y = StandardScaler()

    # 标准化 TEC 数据
    X_train_scaled = scaler_X.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val_scaled = scaler_X.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test_scaled = scaler_X.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    # 标准化辅助因子数据
    aux_train_scaled = scaler_aux.fit_transform(aux_train.reshape(-1, aux_train.shape[-1])).reshape(aux_train.shape)
    aux_val_scaled = scaler_aux.transform(aux_val.reshape(-1, aux_val.shape[-1])).reshape(aux_val.shape)
    aux_test_scaled = scaler_aux.transform(aux_test.reshape(-1, aux_test.shape[-1])).reshape(aux_test.shape)

    # 标准化 y 数据
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(len(y_train), -1))
    y_val_scaled = scaler_y.transform(y_val.reshape(len(y_val), -1))
    y_test_scaled = scaler_y.transform(y_test.reshape(len(y_test), -1))

    return (X_train_scaled, aux_train_scaled, y_train_scaled,
            X_val_scaled, aux_val_scaled, y_val_scaled,
            X_test_scaled, aux_test_scaled, y_test_scaled,
            scaler_X, scaler_aux, scaler_y)

def save_daily_results(y_true, y_pred, dates, true_folder, pred_folder):
    """
    保存测试结果到指定文件夹

    参数:
        y_true (np.ndarray): 真实值，形状为 (num_samples, 24 * 255)
        y_pred (np.ndarray): 预测值，形状为 (num_samples, 24 * 255)
        dates (list): 日期列表
        true_folder (str): 真实值保存文件夹路径
        pred_folder (str): 预测值保存文件夹路径
    """
    os.makedirs(true_folder, exist_ok=True)
    os.makedirs(pred_folder, exist_ok=True)

    for i, date in enumerate(dates):
        np.save(os.path.join(true_folder, f"{date}.npy"), y_true[i])
        np.save(os.path.join(pred_folder, f"{date}.npy"), y_pred[i])
    print(f"结果已保存：{len(dates)} 天的数据保存在 {true_folder} 和 {pred_folder} 文件夹中")

# ---------------- 主函数 ----------------
def main():
    # 数据路径配置
    train_folder = r"E:\全部程序\新数据改\电波科学学报\数据集\训练"
    val_folder = r"E:\全部程序\新数据改\电波科学学报\数据集\验证"
    test_folder = r"E:\全部程序\新数据改\电波科学学报\数据集\测试"
    aux_file = r"20000101-20231231(time+Kp+R+Dst+F107+Bz+ap+AE).csv"

    # 保存预测结果路径
    true_folder = "test_results/true"
    pred_folder = "test_results/pred"
    os.makedirs(true_folder, exist_ok=True)
    os.makedirs(pred_folder, exist_ok=True)

    # 读取辅助因子数据
    print("读取辅助因子数据...")
    aux_data = pd.read_csv(aux_file)
    if 'datetime' not in aux_data.columns:
        raise ValueError("辅助因子CSV文件中缺少 'datetime' 列。")
    aux_data['datetime'] = pd.to_datetime(aux_data['datetime'])
    aux_data.set_index('datetime', inplace=True)
    print("辅助因子数据读取完成。")

    # 加载数据
    print("加载训练、验证和测试数据...")
    X_train, aux_train, y_train, _ = load_data_with_aux(train_folder, aux_data)
    X_val, aux_val, y_val, _ = load_data_with_aux(val_folder, aux_data)
    X_test, aux_test, y_test, test_dates = load_data_with_aux(test_folder, aux_data)
    print("数据加载完成。")

    # 数据标准化
    print("进行数据标准化...")
    (X_train_scaled, aux_train_scaled, y_train_scaled,
     X_val_scaled, aux_val_scaled, y_val_scaled,
     X_test_scaled, aux_test_scaled, y_test_scaled,
     scaler_X, scaler_aux, scaler_y) = standardize_data(
        X_train, aux_train, y_train,
        X_val, aux_val, y_val,
        X_test, aux_test, y_test
    )
    print("数据标准化完成。")

    # 转换为张量
    print("转换为张量...")
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    aux_train_tensor = torch.tensor(aux_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)

    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    aux_val_tensor = torch.tensor(aux_val_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32)

    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    aux_test_tensor = torch.tensor(aux_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)
    print("张量转换完成。")

    # 创建 DataLoader
    print("创建 DataLoader...")
    train_dataset = TEC_Dataset(X_train_tensor, aux_train_tensor, y_train_tensor)
    val_dataset = TEC_Dataset(X_val_tensor, aux_val_tensor, y_val_tensor)
    test_dataset = TEC_Dataset(X_test_tensor, aux_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    print("DataLoader 创建完成。")

    # 模型参数定义
    tec_features = 255
    aux_features = aux_train.shape[-1]
    output_size = y_train_scaled.shape[1]  # 24 * 255

    num_channels = [256, 512, 256]
    transformer_heads = 8
    transformer_layers = 2
    ff_hidden_size = 512
    dropout = 0.2
    time_steps = 48  # 时间步数

    # 定义模型
    print("定义模型...")
    model = TCNSpatioTemporalTransformer(
        tec_features=tec_features,
        aux_features=aux_features,
        num_channels=num_channels,
        transformer_heads=transformer_heads,
        transformer_layers=transformer_layers,
        ff_hidden_size=ff_hidden_size,
        output_size=output_size,
        dropout=dropout,
        time_steps=time_steps
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"模型已加载到设备: {device}")

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=100, factor=0.5, verbose=True)

    # 训练和验证
    best_val_loss = float('inf')
    epochs = 500  # 根据需要调整
    print("开始训练...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs, _ = model(X_batch)  # (batch, 24*255)
            loss = criterion(outputs, y_batch)  # 已展平
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs, _ = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)

        # 学习率调度
        scheduler.step(val_loss)

        print(f"Epoch {epoch}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Best model saved!")

        # 可选：早停
        if epoch - list(scheduler.optimizer.state_dict()['state'].values())[0]['step'] > 100:
            print("Early stopping triggered.")
            break

    print("训练完成。")

    # 测试模型
    print("开始测试...")
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()

    y_pred = []
    y_true = []
    attention_weights_list = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs, attention_weights = model(X_batch)
            y_pred.append(outputs.cpu().numpy())
            y_true.append(y_batch.cpu().numpy())
            attention_weights_list.append(attention_weights.cpu().numpy())

    y_pred = np.concatenate(y_pred, axis=0)  # (num_test_samples, 24*255)
    y_true = np.concatenate(y_true, axis=0)  # (num_test_samples, 24*255)

    # 逆标准化
    y_pred = scaler_y.inverse_transform(y_pred)
    y_true = scaler_y.inverse_transform(y_true)

    # 保存测试结果
    print("保存测试结果...")
    save_daily_results(y_true, y_pred, test_dates, true_folder, pred_folder)

    # 保存 SpatialAttention 权重
    print("保存 SpatialAttention 权重...")
    spatial_attention_weights = model.spatial_attention.weights.detach().cpu().numpy()  # (48, aux_features, 255)
    np.save("spatial_attention_weights.npy", spatial_attention_weights)
    print("SpatialAttention 权重已保存为 spatial_attention_weights.npy")

    # 保存 Attention 权重（每个样本的注意力权重）
    print("保存 Attention 权重...")
    attention_weights_array = np.concatenate(attention_weights_list, axis=0)  # (num_test_samples, 48, 255)
    # 确保数值范围在 [0, 1]
    assert np.all(attention_weights_array >= 0) and np.all(attention_weights_array <= 1), "注意力权重不在 [0, 1] 范围内!"
    np.save("attention_weights.npy", attention_weights_array)
    print("Attention weights 已保存为 attention_weights.npy")

    # 评估模型
    print("评估模型...")
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"测试集 MSE: {mse:.4f}, MAE: {mae:.4f}")

if __name__ == "__main__":
    main()
