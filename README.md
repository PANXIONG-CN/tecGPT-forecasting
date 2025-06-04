# tecGPT-forecasting

**tecGPT-forecasting** 是一个基于预训练大语言模型(LLM)进行高分辨率区域电离层总电子含量(TEC)时空预测的深度学习项目。项目采用GPT-2作为核心架构，通过部分冻结注意力(PFA)策略进行高效微调，并设计了专门的嵌入模块来处理多模态输入数据。

## 🌟 项目亮点

### 🚀 核心创新
- **LLM时空预测范式**: 首次将GPT-2应用于电离层TEC预测，将空间格点视为Token序列
- **部分冻结注意力(PFA)**: 创新的LLM微调策略，在保持预训练知识的同时适应时空预测任务
- **多模态数据融合**: 巧妙整合历史TEC数据、空间天气指数和周期性时间特征
- **端到端处理流程**: 从原始HDF5文件到模型部署的完整自动化流程

### 🧠 先进架构
- **tecGPT模型**: 专为时空预测设计的GPT-2变体，支持2911个空间节点的并行预测
- **智能嵌入系统**:
  - `TecHistoryEmbedding`: 历史TEC序列的时序建模
  - `SpatialEmbedding`: 基于经纬度的空间位置编码
  - `SpaceWeatherEmbedding`: 空间天气指数的全局特征提取
  - `TimeFeatureEmbedding`: 周期性时间特征的语义表示
  - `FusionLayer`: 多模态特征的深度融合

### 📊 企业级实验管理
- **Hydra配置管理**: 灵活的配置系统，支持配置继承和动态覆盖
- **Optuna超参数优化**: 自动化贝叶斯优化，支持并行搜索和早停策略
- **实验跟踪**: 支持Weights & Biases集成(通过备用脚本)
- **分布式训练**: 支持多GPU DDP训练，线性扩展训练效率

### 🔧 工程优势
- **模块化设计**: 可扩展的模型注册机制，便于集成新的预测模型
- **内存优化**: 支持内存映射数据加载、梯度检查点和混合精度训练
- **容错机制**: 完善的异常处理、自动恢复和资源清理
- **标准化流程**: 统一的数据预处理、模型训练和评估接口

## 目录结构

```
tecGPT-forecasting/
│
├── conf/                     # Hydra配置文件
│   ├── config.yaml          # 主配置文件
│   ├── config_optuna_test.yaml # Optuna测试配置
│   ├── config_preprocess.yaml  # 数据预处理配置
│   ├── dataset/             # 数据集配置
│   ├── model/               # 模型配置
│   ├── trainer/             # 训练器配置
│   └── hydra/               # Hydra特定配置
│
├── data_preparation/         # 数据预处理
│   ├── __init__.py
│   ├── preprocess_data.py    # 统一的数据预处理脚本
│   ├── preprocess_data_hydra.py # Hydra版本预处理脚本
│   └── *.hdf5               # 原始HDF5数据文件
│
├── processed_tec_data/       # 预处理后的数据
│   ├── train.npz            # 训练集
│   ├── val.npz              # 验证集
│   ├── test.npz             # 测试集
│   ├── scaler.pkl           # 数据缩放器
│   └── tr13-21_v22-23_t23-25_*/  # 特定配置的数据集
│
├── small_test_data/          # 小规模测试数据
│   ├── train.npz
│   ├── val.npz
│   ├── test.npz
│   └── scaler.pkl
│
├── src/                      # 源代码
│   ├── __init__.py
│   ├── models/              # 模型定义
│   │   ├── __init__.py      # 模型注册机制
│   │   ├── tecGPT/          # tecGPT模型
│   │   │   ├── __init__.py
│   │   │   ├── tec_gpt.py   # 主模型类
│   │   │   ├── pfa_llm.py   # PFA-GPT2实现
│   │   │   ├── embeddings.py # 嵌入层定义
│   │   │   └── gpt2/        # GPT-2预训练模型文件
│   │   ├── ConvLSTM/        # ConvLSTM基线模型
│   │   ├── TCN/             # TCN基线模型
│   │   └── TCN-STT-ATTN/    # TCN-STT-ATTN基线模型
│   │
│   ├── utils/               # 工具函数
│   │   ├── __init__.py
│   │   └── util.py          # 数据处理和评估工具
│   │
│   ├── tec_train.py         # 主训练脚本(Hydra)
│   ├── tec_train_argparse_backup.py # 备用训练脚本(支持wandb)
│   └── evaluate.py          # 模型评估脚本
│
├── logs/                     # 训练日志和模型检查点
│   ├── tecGPT-forecasting/  # 项目日志目录
│   └── *.db                 # Optuna数据库文件
│
├── wandb/                    # Weights & Biases本地文件(可选)
│
├── README.md                 # 项目说明文档
├── requirements.txt          # Python依赖
└── LICENSE                  # 开源协议
```

## 🛠️ 技术栈

### 核心框架
- **PyTorch**: 深度学习框架，支持GPU加速和分布式训练
- **Transformers**: Hugging Face预训练模型库，提供GPT-2模型
- **Hydra**: 配置管理框架，支持动态配置和多任务运行

### 数据处理
- **NumPy & Pandas**: 高效的数值计算和数据处理
- **HDF5 (h5py)**: 大规模科学数据存储格式
- **Scikit-learn**: 数据预处理和评估指标

### 实验管理
- **Hydra**: 配置管理和多任务运行
- **Optuna**: 自动化超参数优化
- **Weights & Biases**: 实验跟踪(可选，通过备用脚本)

### 性能优化
- **混合精度训练**: 自动混合精度(AMP)减少显存占用
- **分布式训练**: DistributedDataParallel多GPU并行
- **内存映射**: 大数据集的高效加载
- **梯度检查点**: 减少训练时的内存峰值

## 🚀 快速开始

### 1. 环境配置

#### 方式一：使用conda环境(推荐)
```bash
# 克隆仓库
git clone https://github.com/PANXIONG-CN/tecGPT-forecasting.git
cd tecGPT-forecasting

# 创建conda环境
conda create -n tecgpt python=3.9
conda activate tecgpt

# 安装PyTorch (根据您的CUDA版本调整)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 安装其他依赖
pip install -r requirements.txt
```

#### 方式二：使用pip环境
```bash
# 克隆仓库
git clone https://github.com/PANXIONG-CN/tecGPT-forecasting.git
cd tecGPT-forecasting

# 创建虚拟环境
python -m venv tecgpt_env
source tecgpt_env/bin/activate  # Linux/Mac
# 或 tecgpt_env\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 环境验证

```bash
# 验证PyTorch和CUDA
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU数量: {torch.cuda.device_count()}')"

# 验证Transformers
python -c "from transformers import GPT2Model; print('Transformers导入成功')"

# 验证Hydra
python -c "import hydra; print('Hydra导入成功')"
```

### 3. 数据准备

#### 3.1 原始数据格式

项目使用的原始数据为HDF5格式文件，包含中国区域的电离层TEC数据和空间天气指数。

**文件命名规则**:
```
CRIM_SW2hr_AI_v1.2_{YEAR}_DataDrivenRange_CN.hdf5
```
其中 `{YEAR}` 为2013-2025年。

**数据结构详解**:

**🌍 电离层数据**:
- **路径**: `/ionosphere/TEC`
- **形状**: `[T, 41, 71]` (时间步数 × 纬度网格 × 经度网格)
- **覆盖范围**:
  - 纬度: 15°N - 55°N (1°分辨率)
  - 经度: 70°E - 140°E (1°分辨率)
- **时间分辨率**: 2小时 (00, 02, 04, ..., 22 UTC)
- **单位**: TECU (Total Electron Content Units, 1 TECU = 10¹⁶ electrons/m²)
- **填充值**: -9999.0 (表示缺失或无效数据)

**📅 时间坐标**:
- `/coordinates/year`: 年份 (2013-2025)
- `/coordinates/month`: 月份 (1-12)
- `/coordinates/day`: 日期 (1-31)
- `/coordinates/hour`: 小时 (0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22)
- `/coordinates/day_of_year`: 年积日 (1-365/366)

**🌌 空间天气指数**:
- `/space_weather_indices/Kp_Index`: **Kp指数** - 全球地磁活动指数 (0-9)
- `/space_weather_indices/Dst_Index`: **Dst指数** - 环电流强度指数 (nT)
- `/space_weather_indices/ap_Index`: **ap指数** - 地磁活动指数 (nT)
- `/space_weather_indices/F107_Index`: **F10.7指数** - 太阳射电流量 (sfu)
- `/space_weather_indices/AE_Index`: **AE指数** - 极光电急流指数 (nT)

#### 3.2 数据预处理流程

**📁 数据准备**:
1. 将原始HDF5文件放入 `data_preparation/` 目录
2. 确保文件命名符合规范
3. 运行统一的数据预处理脚本

**🔧 预处理命令**:

```bash
# 基础预处理 (使用默认参数)
python data_preparation/preprocess_data.py

# 自定义输出目录
python data_preparation/preprocess_data.py \
    --hdf5_dir data_preparation \
    --output_dir ./processed_tec_data

# 使用Hydra配置进行预处理
python data_preparation/preprocess_data_hydra.py \
    --config-name config_preprocess \
    hdf5_dir=data_preparation \
    output_dir=./processed_tec_data
```

**🔄 预处理步骤详解**:

**步骤1: 数据加载与质量控制**
- 📥 从HDF5文件中提取TEC数据和时空坐标
- 🔍 识别并标记填充值(-9999.0)
- 🧮 对F10.7指数进行对数变换: `log10(max(F107, 1e-6))`
- 🔧 使用前向/后向填充处理空间天气指数缺失值
- ✅ 数据完整性检查和异常值检测

**步骤2: 智能特征工程**

*时间特征生成*:
- ⏰ **小时周期性**: `sin(2π × hour/24)`, `cos(2π × hour/24)`
- 📅 **星期周期性**: `sin(2π × dayofweek/7)`, `cos(2π × dayofweek/7)`
- 🗓️ **年积日周期性**: `sin(2π × dayofyear/days_in_year)`, `cos(2π × dayofyear/days_in_year)`

*空间天气特征处理*:
- 🌍 **Kp指数**: 应用比例因子(0.1)，将整数值转换为浮点数
- ⚡ **Dst指数**: 保持原始值，直接使用无额外处理
- 🌪️ **ap/AE指数**: 保持原始值，直接使用无额外处理
- ☀️ **F10.7指数**: 填充值处理 + 前向/后向填充 + 对数变换 `log10(max(F107, 1e-6))`

**步骤3: 高级数据标准化**

*TEC数据 (节点级标准化)*:
- 🎯 每个空间节点(2911个)独立计算统计量
- 📊 公式: `TEC_scaled = (TEC - μ_node) / σ_node`
- 🔒 仅使用训练集数据计算标准化参数
- 🛡️ 处理零方差节点的特殊情况

*空间天气数据 (特征级标准化)*:
- 📈 每个空间天气指数独立标准化
- 📊 公式: `SW_scaled = (SW - μ_feature) / σ_feature`
- 🔄 保持时间序列的相对关系

**步骤4: 滑动窗口序列生成**
- ⏮️ **历史长度**: P = 12 (过去24小时，2小时间隔)
- ⏭️ **预测长度**: S = 12 (未来24小时)
- 🔄 **重叠策略**: 滑动步长=1，最大化数据利用
- 📏 **序列验证**: 确保时间连续性和数据完整性

**步骤5: 智能数据集划分**
- 🏋️ **训练集**: 2013年1月 - 2021年12月 (9年，约75%数据)
- 🎯 **验证集**: 2022年1月 - 2023年6月 (1.5年，约15%数据)
- 🧪 **测试集**: 2023年7月 - 2025年4月 (1.75年，约10%数据)
- ⚖️ **时间分布**: 保持季节性和太阳活动周期平衡，避免数据泄露

**📦 输出数据格式**:

*输入特征 X*: `[样本数, P=12, N=2911, C=12]`
- **C=12维特征组成**:
  - 🌍 TEC历史值 (1维)
  - 🌌 空间天气指数 (5维): Kp, Dst, ap, F10.7, AE
  - ⏰ 时间特征 (6维): hour_sin/cos, dayofweek_sin/cos, dayofyear_sin/cos

*目标变量 Y*: `[样本数, S=12, N=2911, 1]`
- 📊 原始尺度的TEC值，用于损失计算和评估
- 🎯 保持物理意义，便于结果解释

*标准化器*: `scaler.pkl`
- 🔧 NodeScaler: TEC数据的节点级标准化参数
- 📈 FeatureScaler: 空间天气数据的特征级标准化参数
- 💾 支持模型部署时的数据逆变换

### 4. tecGPT模型架构与训练

#### 4.1 🏗️ 模型架构详解

tecGPT采用**多模态嵌入 + PFA-GPT2 + 预测头**的创新架构，专为时空预测任务设计：

```
📥 输入: [B, P=12, N=2911, C=12]
│
├── 🌍 TEC历史嵌入 ────────── TecHistoryEmbedding([B, P, N] → [B, N, D])
├── 🌌 空间天气嵌入 ────────── SpaceWeatherEmbedding([B, P, 5] → [B, N, D])
├── ⏰ 时间特征嵌入 ────────── TimeFeatureEmbedding([B, P, 6] → [B, N, D])
└── 📍 空间位置嵌入 ────────── SpatialEmbedding(网格坐标 → [1, N, D])
│
🔗 多模态融合层 ──────────── FusionLayer([B, N, 4*D] → [B, N, 768])
│
🧠 PFA-GPT2 ─────────────── GPT-2 with Partial Frozen Attention
│                         [B, N, 768] → [B, N, 768]
│
🎯 预测头 ──────────────── MLP ([B, N, 768] → [B, N, S=12])
│
📤 输出: [B, N=2911, S=12]
```

**🔧 核心模块详解**:

**1. TecHistoryEmbedding (历史TEC嵌入)**
- 📊 **输入**: 标准化历史TEC序列 `[B, P=12, N=2911]`
- 🔄 **处理**: 深度MLP `P → 2D → D_embed`，包含GELU激活和LayerNorm
- 📈 **输出**: `[B, N, D_embed]` - 每个节点的历史模式表示
- 🎯 **作用**: 捕获每个空间位置的时序演化模式

**2. SpaceWeatherEmbedding (空间天气嵌入)**
- 🌌 **输入**: 历史空间天气指数 `[B, P=12, 5]`
- 🔄 **处理**: 展平 → MLP投影 → 广播到所有节点
- 📡 **输出**: `[B, N, D_embed]` - 全局空间天气影响
- 🎯 **作用**: 建模全球空间天气对局地TEC的影响

**3. TimeFeatureEmbedding (时间特征嵌入)**
- ⏰ **输入**: 周期性时间特征 `[B, P=12, 6]`
- 🔄 **处理**: 取最新时刻 → MLP投影 → 广播到所有节点
- 📅 **输出**: `[B, N, D_embed]` - 时间周期性表示
- 🎯 **作用**: 编码日、周、年的周期性变化模式

**4. SpatialEmbedding (空间位置嵌入)**
- 📍 **输入**: 预定义的网格坐标 (41×71)
- 🔄 **处理**: 纬度/经度Embedding查表 + MLP融合
- 🗺️ **输出**: `[1, N, D_embed]` - 空间位置编码
- 🎯 **作用**: 为每个网格点提供地理位置信息

**5. FusionLayer (多模态融合层)**
- 🔗 **输入**: 连接所有嵌入 `[B, N, 4×D_embed]`
- 🔄 **处理**: 深度MLP投影到GPT-2维度，包含残差连接
- 🎭 **输出**: `[B, N, 768]` - 统一的多模态表示
- 🎯 **作用**: 智能融合不同模态的信息

**6. PFA-GPT2 (部分冻结注意力GPT-2)**
- 🧠 **基础**: 预训练GPT-2模型 (12层Transformer)
- 🔧 **改进**:
  - 扩展位置嵌入支持2911个节点
  - 应用部分冻结注意力(PFA)策略
  - 支持梯度检查点节省显存
- ❄️ **冻结策略**:
  - 前(L-U)层: 冻结MLP和注意力权重
  - 所有层: 保持LayerNorm可训练
  - 后U层: 解冻注意力机制进行微调
- 🎯 **作用**: 利用预训练语言知识进行时空建模

**7. 预测头 (Prediction Head)**
- 📊 **输入**: `[B, N, 768]` - GPT-2输出特征
- 🔄 **处理**:
  ```
  Linear(768 → 384) → GELU → LayerNorm → Dropout → Linear(384 → S=12)
  ```
- 📈 **输出**: `[B, N, S=12]` - 未来12步预测(标准化尺度)
- 🎯 **作用**: 将语义特征转换为具体的TEC预测值

#### 4.2 🚀 训练过程详解

**📊 数据流程**:
```
原始HDF5 → 特征工程 → 标准化 → NPZ文件 → 内存映射加载 → 模型训练
```

**⚙️ 训练配置**:
- 🎯 **损失函数**: RMSE (在原始物理尺度计算，保持可解释性)
- 🔧 **优化器**: AdamW / Ranger21 (支持梯度中心化和自适应学习率)
- 📈 **学习率调度**: ReduceLROnPlateau (验证损失平台期自动降低学习率)
- ⏹️ **早停策略**: 验证RMSE连续20个epoch无改善时停止
- ⚡ **混合精度训练**: 自动启用AMP节省显存并加速训练
- ✂️ **梯度裁剪**: 防止梯度爆炸，稳定训练过程

**🔄 训练流程**:

**阶段1: 初始化**
1. 📥 加载预处理数据和标准化器
2. 🏗️ 初始化tecGPT模型，应用PFA冻结策略
3. 🔧 配置优化器、调度器和损失函数
4. 📊 初始化实验跟踪和日志记录

**阶段2: 训练循环**
每个epoch包含以下步骤：
- **训练阶段**:
  - 🔄 前向传播: 输入 → 嵌入 → GPT-2 → 预测头
  - 🔙 逆标准化: 将预测结果转换回原始物理尺度
  - 📊 损失计算: 在原始尺度上计算RMSE损失
  - ⬅️ 反向传播: 梯度计算和参数更新
- **验证阶段**:
  - 📈 评估所有指标: MAE, RMSE, MAPE, WMAPE
  - 💾 保存最佳模型 (基于验证RMSE)
  - 📊 记录训练指标和实验日志

**阶段3: 测试评估**
- 🧪 加载最佳模型进行最终测试
- 📏 逐时间步预测并计算分层指标
- 📊 生成详细的评估报告

### 5. 🎯 模型训练指南

> **📝 注意**: 当前主要训练脚本使用Hydra配置管理。如需Weights & Biases实验跟踪，请使用备用脚本 `src/tec_train_argparse_backup.py`。

#### 5.1 基础训练

**使用Hydra配置文件(推荐)**:
```bash
# 使用默认配置训练
python src/tec_train.py

# 使用特定配置文件
python src/tec_train.py --config-name config_optuna_test

# 动态覆盖配置参数
python src/tec_train.py trainer.epochs=50 trainer.batch_size=16 trainer.learning_rate=1e-4

# 指定数据集配置
python src/tec_train.py dataset=small_test_data trainer.epochs=10

# 使用特定数据集配置进行训练
python src/tec_train.py dataset=tec_data_specific_tr13-21_v22-23_t23-25_h12_f12_tf_off

# 启用实验跟踪 (通过Hydra配置)
python src/tec_train.py project_name="my-tecgpt-experiments"
```

**使用命令行参数(备用方式)**:
```bash
# 基础训练
python src/tec_train_argparse_backup.py

# 自定义参数训练 (备用脚本支持wandb)
python src/tec_train_argparse_backup.py \
    --epochs 50 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --use_wandb
```

#### 5.2 分布式训练

```bash
# 双GPU分布式训练
python src/tec_train.py trainer.use_ddp=true trainer.world_size=2 trainer.use_amp=true

# 四GPU分布式训练
python src/tec_train.py trainer.use_ddp=true trainer.world_size=4 trainer.use_amp=true

# 使用备用脚本进行分布式训练
python src/tec_train_argparse_backup.py --use_ddp --world_size 2 --use_amp
```

#### 5.3 超参数优化

**使用Optuna进行自动优化**:
```bash
# 基础Optuna优化
python src/tec_train.py --multirun use_optuna=true optuna_trials=20

# 使用小数据集进行快速Optuna测试
python src/tec_train.py --multirun \
    dataset=small_test_data \
    use_optuna=true \
    optuna_trials=10 \
    trainer.epochs=5

# 结合实验跟踪的Optuna优化
python src/tec_train.py --multirun \
    use_optuna=true \
    optuna_trials=50 \
    project_name="tecgpt-optuna-optimization"

# 自定义Optuna存储和数据集
python src/tec_train.py --multirun \
    dataset=tec_data_basic \
    use_optuna=true \
    optuna_trials=30 \
    optuna_storage="sqlite:///my_optuna_study.db" \
    optuna_study_name="tecgpt_optimization_v2"
```

#### 5.4 数据集配置选择

项目提供了多种预配置的数据集选项：

| 数据集配置 | 描述 | 适用场景 |
|-----------|------|----------|
| `small_test_data` | 小规模测试数据集 | 快速测试、调试、Optuna优化 |
| `tec_data_basic` | 基础数据集配置 | 标准训练 |
| `tec_data_v1` | 版本1数据集 | 特定实验配置 |
| `tec_data_specific_tr13-21_v22-23_t23-25_h12_f12_tf_off` | 特定时间范围配置 | 精确的时间划分实验 |
| `tec_data_specific_tr13-21_v22-23_t23-25_h36_f12_tf_off` | 36小时历史长度配置 | 长历史序列实验 |

**使用示例**:
```bash
# 快速测试
python src/tec_train.py dataset=small_test_data trainer.epochs=5

# 标准训练
python src/tec_train.py dataset=tec_data_basic

# 长历史序列实验
python src/tec_train.py dataset=tec_data_specific_tr13-21_v22-23_t23-25_h36_f12_tf_off
```

### 6. 📊 模型评估

#### 6.1 评估训练好的模型

```bash
# 基础评估
python src/evaluate.py --model_path logs/tecGPT_YYYYMMDD-HHMMSS/best_model.pth

# 自定义评估配置
python src/evaluate.py \
    --model_path logs/tecGPT_20241201-143022/best_model.pth \
    --data_dir ./processed_tec_data \
    --batch_size 64 \
    --output_dir ./eval_results
```

#### 6.2 评估输出

评估脚本将生成以下输出：
- 📊 **控制台输出**: 每个预测时间步的详细指标
- 📁 **CSV文件**: `evaluation_results.csv` - 完整的评估结果
- 📈 **平均指标**: 所有时间步的平均性能指标

## ⚙️ 详细配置

### 7.1 模型参数配置

#### 核心模型参数

| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| `model.model_name` | tecGPT | - | 模型类型选择 |
| `model.d_embed` | 128 | 64-512 | 嵌入层维度，影响模型表达能力 |
| `model.d_llm` | 768 | 512-1024 | GPT-2隐藏维度，与预训练模型匹配 |
| `model.llm_layers_to_use` | 6 | 1-12 | 使用的GPT-2层数，平衡性能与效率 |
| `model.U_unfrozen_mha` | 2 | 1-6 | PFA策略中解冻的注意力层数 |
| `model.dropout_embed` | 0.1 | 0.0-0.5 | 嵌入层dropout率，防止过拟合 |
| `model.dropout_llm_out` | 0.1 | 0.0-0.5 | LLM输出层dropout率 |
| `model.enable_gradient_checkpointing_llm` | true | - | 是否启用梯度检查点节省显存 |

#### GPT-2配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model.local_gpt2_path` | `src/models/tecGPT/gpt2` | 本地GPT-2模型路径 |
| `model.gpt2_model_name` | gpt2 | GPT-2模型规格 (gpt2/gpt2-medium/gpt2-large) |

### 7.2 训练参数配置

#### 基础训练参数

| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| `trainer.batch_size` | 16 | 1-128 | 批次大小，根据GPU显存调整 |
| `trainer.learning_rate` | 5e-5 | 1e-6-1e-3 | 学习率，影响收敛速度和稳定性 |
| `trainer.epochs` | 100 | 10-500 | 最大训练轮数 |
| `trainer.patience` | 20 | 5-50 | 早停耐心值，验证指标无改善的容忍轮数 |
| `trainer.optimizer_type` | AdamW | AdamW/Ranger | 优化器类型 |
| `trainer.weight_decay` | 0.01 | 0.0-0.1 | 权重衰减，L2正则化强度 |
| `trainer.clip_grad_norm` | 1.0 | 0.1-10.0 | 梯度裁剪阈值，防止梯度爆炸 |

#### 高级训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `trainer.use_amp` | true | 是否使用混合精度训练 |
| `trainer.use_ddp` | false | 是否使用分布式数据并行 |
| `trainer.world_size` | 2 | 分布式训练的GPU数量 |
| `trainer.print_every_epochs` | 1 | 打印训练日志的频率 |

### 7.3 数据集参数配置

#### 数据维度参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `dataset.history_len` | 12 | 输入历史序列长度 (24小时) |
| `dataset.forecast_len` | 12 | 预测序列长度 (24小时) |
| `dataset.num_nodes` | 2911 | 空间节点总数 (41×71) |
| `dataset.n_lat` | 41 | 纬度网格数 |
| `dataset.n_lon` | 71 | 经度网格数 |
| `dataset.tec_feat_dim` | 1 | TEC特征维度 |
| `dataset.sw_feat_dim` | 5 | 空间天气特征维度 |
| `dataset.time_feat_dim` | 6 | 时间特征维度 |

#### 数据路径参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `dataset.data_dir` | `./processed_tec_data` | 预处理数据目录 |
| `dataset.scaler_path` | `./processed_tec_data/scaler.pkl` | 标准化器文件路径 |
| `dataset.use_time_features` | true | 是否使用时间特征 |

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
