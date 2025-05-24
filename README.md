# tecGPT-forecasting

**tecGPT-forecasting** 是一个基于预训练大语言模型(LLM)进行高分辨率区域电离层总电子含量(TEC)时空预测的深度学习项目。项目采用GPT-2作为核心架构，通过部分冻结注意力(PFA)策略进行高效微调，并设计了专门的嵌入模块来处理多模态输入数据。

## 特性

### 🚀 核心功能
- **基于LLM的时空预测**: 使用GPT-2作为骨干网络，将空间格点视为Token进行序列预测
- **部分冻结注意力(PFA)**: 高效的LLM微调策略，平衡性能与计算资源
- **多模态数据融合**: 整合历史TEC数据、空间天气指数和时间特征
- **端到端数据处理**: 从原始HDF5文件到模型训练的完整流程

### 🧠 模型架构
- **ST_LLM(tecGPT)**: 主模型类，整合多种嵌入层和PFA-GPT2
- **专用嵌入模块**:
  - `TecHistoryEmbedding`: 处理历史TEC序列
  - `TimeFeatureEmbedding`: 处理周期性时间特征
  - `SpatialEmbedding`: 基于格点位置的空间嵌入
  - `SpaceWeatherEmbedding`: 处理空间天气指数
  - `FusionLayer`: 多模态特征融合层

### 📊 实验管理
- **Weights & Biases集成**: 完整的实验跟踪和可视化
- **Optuna超参数优化**: 自动化超参数搜索
- **分布式训练支持**: 支持多GPU分布式训练(DDP)
- **混合精度训练**: 提高训练效率，减少显存占用

### 🔧 工程特性
- **模块化设计**: 可扩展的模型注册机制，便于添加新模型
- **标准化数据处理**: 统一的数据预处理和缩放流程
- **灵活的训练配置**: 支持命令行参数调整
- **完善的评估体系**: 多种评估指标和可视化

## 目录结构

```
tecGPT-forecasting/
│
├── data_preparation/         # 数据预处理
│   ├── preprocess_data.py    # 统一的数据预处理脚本
│   └── *.hdf5               # 原始HDF5数据文件
│
├── processed_tec_data/       # 预处理后的数据
│   ├── train.npz            # 训练集
│   ├── val.npz              # 验证集  
│   ├── test.npz             # 测试集
│   └── scaler.pkl           # 数据缩放器
│
├── src/                      # 源代码
│   ├── models/              # 模型定义
│   │   ├── __init__.py      # 模型注册机制
│   │   └── tecGPT/          # tecGPT模型
│   │       ├── __init__.py
│   │       ├── tec_gpt.py   # 主模型类
│   │       ├── pfa_llm.py   # PFA-GPT2实现
│   │       ├── embeddings.py # 嵌入层定义
│   │       └── gpt2/        # GPT-2预训练模型文件
│   │
│   ├── utils/               # 工具函数
│   │   ├── __init__.py
│   │   └── util.py          # 数据处理和评估工具
│   │
│   ├── tec_train.py         # 主训练脚本
│   ├── evaluate.py          # 模型评估脚本
│   └── ranger21.py          # Ranger优化器实现
│
├── logs/                     # 训练日志和模型检查点
├── wandb/                    # Weights & Biases本地文件
├── eval_results/             # 评估结果输出
│
├── README.md                 # 项目说明文档
├── requirements.txt          # Python依赖
├── .gitignore               # Git忽略文件
└── LICENSE                  # 开源协议
```

## 快速开始

### 1. 环境安装

```bash
# 克隆仓库
git clone https://github.com/your-username/tecGPT-forecasting.git
cd tecGPT-forecasting

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

#### 2.1 原始数据格式

项目使用的原始数据为HDF5格式文件，命名规则为：
```
CRIM_SW2hr_AI_v1.2_{YEAR}_DataDrivenRange_CN.hdf5
```

每个HDF5文件包含以下数据结构：

**电离层数据**:
- `/ionosphere/TEC`: 总电子含量数据 [时间, 纬度, 经度]
  - 形状: `[T, 41, 71]` (时间步数 × 纬度网格 × 经度网格)
  - 单位: TECU (Total Electron Content Units)
  - 填充值: -9999.0 (表示缺失数据)

**坐标信息**:
- `/coordinates/year`: 年份
- `/coordinates/month`: 月份  
- `/coordinates/day`: 日期
- `/coordinates/hour`: 小时 (偶数小时: 0, 2, 4, ..., 22)
- `/coordinates/day_of_year`: 年积日

**空间天气指数**:
- `/space_weather_indices/Kp_Index`: Kp指数 (地磁活动)
- `/space_weather_indices/Dst_Index`: Dst指数 (环电流强度)
- `/space_weather_indices/ap_Index`: ap指数 (地磁活动)
- `/space_weather_indices/F107_Index`: F10.7指数 (太阳射电流量)
- `/space_weather_indices/AE_Index`: AE指数 (极光电急流)

#### 2.2 数据预处理流程

将原始HDF5文件放入 `data_preparation/` 目录，然后运行统一的数据预处理脚本：

```bash
python data_preparation/preprocess_data.py --hdf5_dir data_preparation --output_dir ./processed_tec_data
```

**预处理步骤详解**:

1. **数据加载与质量控制**:
   - 从HDF5文件中提取TEC数据和元数据
   - 识别并标记填充值(-9999.0)
   - 对F10.7指数进行对数变换: `log10(max(F107, 1e-6))`
   - 使用前向填充和后向填充处理空间天气指数的缺失值

2. **特征工程**:
   
   **时间特征生成**:
   - 小时周期性编码: `sin(2π * hour / 24)`, `cos(2π * hour / 24)`
   - 星期周期性编码: `sin(2π * dayofweek / 7)`, `cos(2π * dayofweek / 7)`  
   - 年积日周期性编码: `sin(2π * dayofyear / days_in_year)`, `cos(2π * dayofyear / days_in_year)`

   **空间天气特征**:
   - Kp指数: 应用比例因子 (通常为0.1)
   - Dst, ap, AE指数: 保持原始值
   - F10.7指数: 对数变换后的值

3. **数据标准化**:
   
   **TEC数据 (节点级标准化)**:
   - 每个空间节点独立计算均值和标准差
   - 公式: `TEC_scaled = (TEC - mean_node) / std_node`
   - 只使用训练集数据计算标准化参数

   **空间天气数据 (特征级标准化)**:
   - 每个空间天气指数独立标准化
   - 公式: `SW_scaled = (SW - mean_feature) / std_feature`

4. **滑动窗口序列生成**:
   - 历史长度: P = 12 (24小时，每2小时一个点)
   - 预测长度: S = 12 (未来24小时)
   - 序列重叠生成，最大化数据利用率

5. **数据集划分**:
   - 训练集: 2013-2019年
   - 验证集: 2020-2021年  
   - 测试集: 2022-2025年

**输出数据格式**:
- X (输入): `[样本数, P=12, N=2911, C=12]`
  - C=12 特征: [1个TEC + 5个空间天气指数 + 6个时间特征]
- Y (目标): `[样本数, S=12, N=2911, 1]`
  - 原始尺度的TEC值，用于损失计算
- scaler.pkl: 包含标准化参数的文件

### 3. tecGPT模型架构与训练过程

#### 3.1 模型架构详解

tecGPT采用多模态嵌入 + PFA-GPT2 + 预测头的架构：

```
输入: [B, P=12, N=2911, C=12]
│
├── TEC历史嵌入 ────────── TecHistoryEmbedding([B, P, N] → [B, N, D])
├── 空间天气嵌入 ────────── SpaceWeatherEmbedding([B, P, 5] → [B, N, D])  
├── 时间特征嵌入 ────────── TimeFeatureEmbedding([B, P, 6] → [B, N, D])
└── 空间位置嵌入 ────────── SpatialEmbedding(网格坐标 → [1, N, D])
│
多模态融合层 ──────────── FusionLayer([B, N, 4*D] → [B, N, 768])
│
PFA-GPT2 ─────────────── GPT-2 with Partial Frozen Attention
│                         [B, N, 768] → [B, N, 768]
│
预测头 ──────────────── Linear层 ([B, N, 768] → [B, N, S=12])
│
输出: [B, N=2911, S=12]
```

**各模块详细说明**:

1. **TecHistoryEmbedding**:
   - 输入: 标准化后的历史TEC序列 `[B, P, N]`
   - 处理: MLP投影 `P → D_embed`
   - 输出: `[B, N, D_embed]`

2. **SpaceWeatherEmbedding**:  
   - 输入: 历史空间天气指数 `[B, P, 5]`
   - 处理: 展平为 `[B, P*5]`，然后MLP投影，广播到所有节点
   - 输出: `[B, N, D_embed]`

3. **TimeFeatureEmbedding**:
   - 输入: 周期性时间特征 `[B, P, 6]` 
   - 处理: 取最后时刻特征 `[B, 6]`，MLP投影，广播到所有节点
   - 输出: `[B, N, D_embed]`

4. **SpatialEmbedding**:
   - 输入: 网格坐标(预定义)
   - 处理: 纬度/经度Embedding查表 + MLP投影
   - 输出: `[1, N, D_embed]` (批次间共享)

5. **FusionLayer**:
   - 输入: 连接所有嵌入 `[B, N, 4*D_embed]`
   - 处理: MLP投影到GPT-2维度
   - 输出: `[B, N, 768]`

6. **PFA-GPT2**:
   - 加载预训练GPT-2模型
   - 替换位置嵌入层适应2911个节点
   - 应用部分冻结注意力策略:
     - 冻结前(L-U)层的MLP和注意力权重
     - 保持所有LayerNorm可训练
     - 解冻后U层的注意力机制
   - 支持梯度检查点减少显存

7. **预测头**:
   - 输入: `[B, N, 768]`
   - 处理: Linear(768 → 384) + GELU + LayerNorm + Dropout + Linear(384 → S)  
   - 输出: `[B, N, S]` (标准化尺度)

#### 3.2 训练过程

**数据流程**:
```
原始HDF5 → 预处理 → NPZ文件 → DataLoader → 模型训练
```

**训练配置**:
- 损失函数: MAE (在原始尺度计算)
- 优化器: AdamW / Ranger21 (支持梯度中心化)
- 学习率调度: ReduceLROnPlateau  
- 早停策略: 验证集MAE无改善时停止
- 混合精度训练: 自动启用以节省显存
- 梯度裁剪: 防止梯度爆炸

**训练流程**:
1. 加载预处理数据和标准化器
2. 初始化tecGPT模型，应用PFA策略
3. 每个epoch:
   - 训练阶段: 前向传播 → 逆标准化 → 损失计算 → 反向传播
   - 验证阶段: 评估所有指标 (MAE, RMSE, MAPE, WMAPE)
   - 保存最佳模型 (基于验证MAE)
4. 测试评估: 逐时间步预测并计算分层指标

### 4. 模型训练

#### 基础训练
```bash
# 使用默认参数训练
python src/tec_train.py

# 自定义参数训练
python src/tec_train.py --epochs 50 --batch_size 16 --learning_rate 1e-4 --use_wandb
```

#### 分布式训练
```bash
# 双GPU训练
python src/tec_train.py --use_ddp --world_size 2 --use_amp
```

#### 超参数优化
```bash
# 使用Optuna进行超参数搜索
python src/tec_train.py --use_optuna --optuna_trials 20 --use_wandb
```

### 5. 模型评估

```bash
# 评估训练好的模型
python src/evaluate.py --model_path logs/tecGPT_YYYYMMDD-HHMMSS/best_model.pth
```

## 详细配置

### 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_name` | tecGPT | 模型类型选择 |
| `--d_embed` | 128 | 嵌入维度 |
| `--d_llm` | 768 | LLM隐藏维度 |
| `--llm_layers_to_use` | 6 | 使用的GPT层数 |
| `--U_unfrozen_mha` | 2 | 解冻的注意力层数 |
| `--dropout_embed` | 0.1 | 嵌入层dropout |
| `--dropout_llm_out` | 0.1 | LLM输出dropout |

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--batch_size` | 16 | 批次大小 |
| `--learning_rate` | 5e-5 | 学习率 |
| `--epochs` | 100 | 训练轮数 |
| `--patience` | 20 | 早停耐心值 |
| `--optimizer_type` | AdamW | 优化器类型 |
| `--weight_decay` | 0.01 | 权重衰减 |
| `--clip_grad_norm` | 1.0 | 梯度裁剪 |

### 数据参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--history_len` | 12 | 输入序列长度 |
| `--forecast_len` | 12 | 预测序列长度 |
| `--num_nodes` | 2911 | 空间节点数(41×71) |
| `--tec_feat_dim` | 1 | TEC特征维度 |
| `--sw_feat_dim` | 5 | 空间天气特征维度 |
| `--time_feat_dim` | 6 | 时间特征维度 |

## 实验管理

### Weights & Biases

启用wandb进行实验跟踪：

```bash
# 基础wandb使用
python src/tec_train.py --use_wandb --wandb_project "your-project" --wandb_entity "your-entity"

# 设置实验组
python src/tec_train.py --use_wandb --wandb_group "experiment_group_1"
```

### Optuna超参数优化

```bash
# 使用SQLite存储优化历史
python src/tec_train.py --use_optuna --optuna_storage "sqlite:///optuna.db" --optuna_trials 50

# 自定义study名称
python src/tec_train.py --use_optuna --optuna_study_name "tecGPT_optimization_v1"
```

## 性能优化

### 内存优化
- **梯度检查点**: 自动启用以减少GPU内存使用
- **混合精度训练**: 使用`--use_amp`参数
- **批次大小调整**: 根据GPU内存自动调整

### 训练加速
- **分布式训练**: 支持多GPU DDP训练
- **数据并行**: 高效的数据加载和预处理
- **模型并行**: PFA策略减少可训练参数

## 模型扩展

### 添加新模型

1. 在`src/models/`下创建新的模型目录
2. 实现模型类并继承相应的基类
3. 在`src/models/__init__.py`中注册新模型：

```python
# 在MODEL_REGISTRY中添加新模型
MODEL_REGISTRY = {
    'tecGPT': ST_LLM,
    'your_new_model': YourNewModel,  # 新增
}
```

4. 在`tec_train.py`的`create_model`函数中添加模型创建逻辑

### 自定义损失函数

在`utils/util.py`中定义新的损失函数，并在训练脚本中引用。

## 评估指标

项目支持多种评估指标：

- **MAE** (Mean Absolute Error): 平均绝对误差
- **RMSE** (Root Mean Squared Error): 均方根误差  
- **MAPE** (Mean Absolute Percentage Error): 平均绝对百分比误差
- **WMAPE** (Weighted Mean Absolute Percentage Error): 加权平均绝对百分比误差

所有指标都在原始物理量纲上计算，确保结果的可解释性。

## 常见问题

### Q: 如何处理OOM错误？
A: 
- 减小批次大小(`--batch_size`)
- 启用混合精度训练(`--use_amp`)
- 减少模型层数(`--llm_layers_to_use`)
- 启用梯度检查点(默认开启)

### Q: 如何恢复中断的训练？
A: 训练会自动保存检查点到`logs/`目录，可以从最新检查点恢复。

### Q: 如何调整数据集划分？
A: 修改`data_preparation/preprocess_data.py`中的年份范围常量。

### Q: 如何使用自定义的GPT-2模型？
A: 修改`--local_gpt2_path`参数指向你的GPT-2模型文件目录。

## 数据说明

### 输入数据格式
- **TEC历史数据**: 标准化后的电离层总电子含量
- **空间天气指数**: Kp, Dst, AE, F10.7, Ap指数(标准化)
- **时间特征**: 小时、星期、年积日的周期性编码(sin/cos)

### 输出数据格式  
- **TEC预测**: 原始量纲的未来TEC值
- **形状**: `[批次大小, 节点数, 预测长度]`

## 许可证

本项目采用MIT许可证。详情请见[LICENSE](LICENSE)文件。

## 贡献指南

欢迎提交Issue和Pull Request！请确保：

1. 代码符合项目风格
2. 添加必要的测试
3. 更新相关文档
4. 提交前运行完整测试

## 引用

如果本项目对您的研究有帮助，请考虑引用：

```bibtex
@software{tecgpt_forecasting,
  title={tecGPT-forecasting: LLM-based Ionospheric TEC Prediction},
  author={Your Name},
  year={2024},
  url={https://github.com/your-username/tecGPT-forecasting}
}
```

## 联系方式

- 项目主页: [GitHub Repository](https://github.com/PANXIONG-CN/tecGPT-forecasting)
- 问题反馈: [GitHub Issues](https://github.com/PANXIONG-CN/tecGPT-forecasting/issues)
- 邮箱: xiong.pan@gmail.com
