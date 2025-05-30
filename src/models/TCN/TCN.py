# 导入 NumPy 库，用于高效的数值计算和数组操作
import numpy as np

# 导入 pandas 库，用于数据处理和分析，特别是表格数据
import pandas as pd

# 导入 PyTorch 库，核心深度学习框架，提供张量计算和自动微分功能
import torch

# 从 PyTorch 中导入神经网络模块，包含各种神经网络层和损失函数
import torch.nn as nn

# 从 scikit-learn 导入 MinMaxScaler，用于将数据缩放到指定的范围（如 [0, 1]）
from sklearn.preprocessing import MinMaxScaler

# 从 scikit-learn 导入 mean_squared_error，用于计算均方误差，常用于回归模型的性能评估
from sklearn.metrics import mean_squared_error

# 导入 os 库，提供与操作系统交互的功能，如文件和目录操作
import os

# 从 PyTorch 导入 DataLoader 和 Dataset，用于数据加载和处理
from torch.utils.data import DataLoader, Dataset
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# 定义TCN模型
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCNModel(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCNModel, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        y1 = self.tcn(x)
        return self.linear(y1[:, :, -1])

# 数据集定义
class TEC_Dataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# 将数据保存为文件，文件命名为：2003_01_01.npy, 2003_01_02.npy, ... 2019_12_31.npy
# 加载数据文件
# data_dir = r'E:\中国数据\NPY\data\20030101-20191231(天)'  # 替换为实际的数据文件目录

# data_dir = r'D:\Desktop\深度学习重制版\重制数据\data2'  # 替换为实际的数据文件目录
data_dir = r'D:\Desktop\深度学习重制版\重制数据\21-22年数据新'  # 替换为实际的数据文件目录


# 获取所有文件列表
data_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npy')])

# 读取数据并创建训练和测试集
data = [np.load(f) for f in data_files]
data = np.array(data)  # 形状为（天数，24，17，15）

# 分割数据集
train_data = data[:-367]  # 训练集，假设用最后一年（365天）数据作为测试集
test_data = data[-367:]  # 测试集

# 计算训练集的全局最大值和最小值
train_data_min = train_data.min()
train_data_max = train_data.max()

# 数据标准化 (全局归一化)
train_data_scaled = (train_data - train_data_min) / (train_data_max - train_data_min)
test_data_scaled = (test_data - train_data_min) / (train_data_max - train_data_min)

# 保存归一化参数
np.save('scaler_min.npy', np.array([train_data_min]))
np.save('scaler_max.npy', np.array([train_data_max]))

# 创建序列
seq_length = 2
X_train, y_train = create_sequences(train_data_scaled, seq_length)
X_test, y_test = create_sequences(test_data_scaled, seq_length)

# 确认维度匹配
print(f'X_train shape: {X_train.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_test shape: {y_test.shape}')

# 调整数据形状
X_train = X_train.reshape((X_train.shape[0], seq_length, -1)).transpose(0, 2, 1)  # (batch_size, num_features, seq_length)
y_train = y_train.reshape((y_train.shape[0], -1))  # (batch_size, num_features)
X_test = X_test.reshape((X_test.shape[0], seq_length, -1)).transpose(0, 2, 1)  # (batch_size, num_features, seq_length)
y_test = y_test.reshape((y_test.shape[0], -1))  # (batch_size, num_features)

# 转换为PyTorch张量
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# 创建数据加载器
train_dataset = TEC_Dataset(X_train, y_train)
test_dataset = TEC_Dataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 模型定义
# input_size = 24 * 17 * 15  # 输入维度 (num_features)
###################################################
# input_size = 12 * 13 * 11  # 输入维度 (num_features)
input_size = 12 * 41 * 71  # 输入维度 (num_features)

# num_channels = [128] * 2
num_channels = [128, 256, 512] 
kernel_size = 2
# dropout = 0.3571606746344543
dropout = 0.3
# output_size = 24 * 17 * 15  # 输出维度


############################################
# output_size = 12 * 13 * 11  # 输出维度
output_size = 12 * 41 * 71  # 输出维度


model = TCNModel(input_size, output_size, num_channels, kernel_size, dropout).to('cuda')

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# 训练模型
# num_epochs = 200
num_epochs = 40
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to('cuda'), y_batch.to('cuda')
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}')

# 保存模型权重
torch.save(model.state_dict(), 'TCN_model.pth')

# 评估模型
model.eval()
with torch.no_grad():
    y_pred = []
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to('cuda'), y_batch.to('cuda')
        outputs = model(X_batch)
        # y_pred.append(outputs.cpu().numpy()) #
        y_pred.append(outputs.cpu()) #
        #y_pred.append(outputs.detach().cpu().numpy())
    y_pred = np.concatenate(y_pred, axis=0)
    # y_pred = np.concat(y_pred, axis=0)


# 反标准化预测结果
y_pred = y_pred * (train_data_max - train_data_min) + train_data_min
y_test_inv = y_test.cpu().numpy() * (train_data_max - train_data_min) + train_data_min

# 计算RMSE
# rmse = np.sqrt(mean_squared_error(y_test_inv.flatten(), y_pred.flatten()))
# print(f'RMSE = {rmse:.4f}')
# 计算各种评价指标
rmse = np.sqrt(mean_squared_error(y_test_inv.flatten(), y_pred.flatten()))
mae = mean_absolute_error(y_test_inv.flatten(), y_pred.flatten())
r2 = r2_score(y_test_inv.flatten(), y_pred.flatten())
rho, _ = stats.spearmanr(y_test_inv.flatten(), y_pred.flatten())
rho_squared = rho ** 2

print(f'RMSE = {rmse:.4f}')
print(f'MAE = {mae:.4f}')
print(f'R^2 = {r2:.4f}')
print(f'ρ^2 = {rho_squared:.4f}')
# 保存测试集上的预测结果和真实值
np.save('test_y_test_inv.npy', y_test_inv)
np.save('test_y_pred.npy', y_pred)

