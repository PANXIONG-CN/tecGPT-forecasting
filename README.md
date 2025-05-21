# tecGPT-forecasting

**tecGPT-forecasting** 是一个旨在利用预训练语言模型 (LLM)，特别是基于 GPT-2 架构的模型，进行高分辨率区域电离层总电子含量 (Total Electron Content, TEC) 时空预测的科研项目。项目采用了部分冻结注意力 (Partial Freezing Attention, PFA) 策略以高效微调 LLM，并设计了专门的嵌入模块来处理多样化的输入数据，包括历史TEC图、空间天气指数和时间特征。

本项目使用 [Hydra](https://hydra.cc/) 进行配置管理，[Weights & Biases (Wandb)](https://wandb.ai/) 进行实验跟踪和可视化，并集成 [Optuna](https://optuna.org/) 进行超参数调优。代码采用 PyTorch 框架实现，并提供了 Docker 环境以确保可复现性。

## 特性

*   **基于 LLM 的时空预测**: 使用 GPT-2 作为核心，将空间格点视为 Token 进行序列预测。
*   **部分冻结注意力 (PFA)**: 高效微调 LLM 的策略，平衡性能与计算资源。
    *   冻结 LLM 的前 F 层 (MHA+FFN，LayerNorm 可选可训练)。
    *   解冻后 U 层的 MHA 和 LayerNorm，但保持这些层的 MLP (FFN) 部分冻结。
*   **专用嵌入模块**:
    *   `TecHistoryEmbedding`: 处理各节点的历史TEC序列。
    *   `TimeFeatureEmbedding`: 处理周期编码的时间特征。
    *   `SpatialEmbedding`: 基于格点经纬度索引的可学习空间嵌入。
    *   `SpaceWeatherEmbedding`: 处理历史空间天气指数。
    *   `FusionLayer`: 将上述异构嵌入融合为 LLM 的输入。
*   **端到端数据处理**:
    *   从年度 HDF5 文件加载原始数据 (`/ionosphere/TEC`, `/coordinates/*`, `/space_weather_indices/*`)。
    *   预处理脚本 (`preprocess_hdf5.py`) 执行特征工程 (Kp 指数缩放, F10.7 对数转换, 周期时间编码)、数据标准化 (节点级TEC，特征级SW指数) 并保存缩放器 (`scaler.pkl`)，最后生成滑动窗口样本。
    *   输出 `train.npz`, `val.npz`, `test.npz` 文件，格式为 `X: [Num_Samples, P, N_nodes, C_in]`, `Y: [Num_Samples, S, N_nodes, 1]`。
*   **灵活的实验管理**:
    *   **Hydra**: 用于全面的配置管理 (数据、模型、训练器、日志等)。
    *   **Wandb**: 集成用于实验跟踪、日志记录和可视化。
    *   **Optuna**: 通过 Hydra 插件进行超参数优化。
*   **标准化训练与评估流程**:
    *   `train.py`: Hydra 驱动的训练脚本。
    *   `evaluate.py`: Hydra 驱动的评估脚本，加载检查点在测试集上评估。
    *   `tune.py`: Hydra 驱动的超参数调优脚本。
    *   `TecTrainer`: 封装训练和评估的核心逻辑，包括早停、模型检查点保存、指标计算等。
*   **可复现环境**:
    *   `environment.yaml`: Conda 环境定义。
    *   `requirements.txt`: Pip 依赖列表。
    *   `Dockerfile` 和 `.dockerignore`: 用于构建 Docker 镜像。

## 目录结构

```
tecGPT-forecasting/
│
├── conf/                     # Hydra 配置文件
│   ├── config.yaml           # 主配置文件
│   ├── data/default.yaml     # 数据处理与加载配置
│   ├── model/tec_gpt.yaml    # 模型结构与参数配置
│   ├── trainer/default.yaml  # 训练器参数配置
│   ├── hydra/                # Hydra 内部组件配置 (如日志)
│   └── tune.yaml             # Optuna 超参数调优配置
│
├── data_preparation/         # HDF5 原始数据预处理脚本
│   └── preprocess_hdf5.py
│   └── (示例 HDF5 数据可放于此或外部指定路径)
│
├── processed_data/           # 预处理后的 .npz 数据和 scaler.pkl (被 .gitignore 忽略)
│
├── src/                      # 项目源代码
│   ├── datasets/             # PyTorch Dataset 和 DataModule
│   │   └── tec_dataset.py
│   ├── models/               # 模型定义
│   │   ├── __init__.py
│   │   ├── tec_gpt.py        # TecGPT 主模型 (ST_LLM)
│   │   ├── embeddings.py     # 各种嵌入层
│   │   └── pfa_llm.py        # PFA-GPT2 实现
│   ├── trainers/             # 训练器类
│   │   └── tec_trainer.py
│   ├── utils/                # 辅助工具、指标、缩放器等
│   │   ├── __init__.py
│   │   ├── scaler.py         # 数据缩放器 (用于逆转换)
│   │   ├── metrics.py        # 评估指标函数
│   │   └── helpers.py        # 通用辅助函数 (如 seed_everything)
│   ├── train.py              # 主训练脚本
│   ├── evaluate.py           # 主评估脚本
│   └── tune.py               # 主超参数调优脚本
│
├── outputs/                  # Hydra 默认输出目录 (被 .gitignore 忽略)
├── multirun/                 # Hydra 多重运行输出目录 (被 .gitignore 忽略)
├── wandb/                    # Wandb 本地文件 (被 .gitignore 忽略)
├── gpt2_cache_docker/        # Dockerfile 中预下载模型的缓存 (可选, 被 .gitignore 忽略)
│
├── README.md                 # 本文件
├── environment.yaml          # Conda 环境描述文件
├── requirements.txt          # Pip 需求列表
├── Dockerfile                # Docker 镜像定义
├── .dockerignore             # Docker 构建时忽略的文件
├── .gitignore                # Git 版本控制忽略的文件
└── LICENSE                   # 项目许可证
```

## 安装与设置

### 1. Conda 环境 (推荐)

建议使用 Conda 创建和管理项目环境：

```bash
# 从 environment.yaml 创建 conda 环境
conda env create -f environment.yaml

# 激活环境
conda activate tecgpt
```

### 2. Pip 依赖

如果不使用 Conda，可以先创建虚拟环境，然后通过 `pip` 安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate  # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Docker 环境

项目提供了 `Dockerfile` 用于构建可复现的 Docker 镜像：

```bash
# 构建 Docker 镜像 (在项目根目录下运行)
docker build -t tecgpt-forecasting .

# 运行 Docker 容器 (示例)
# docker run -it --rm \
#   -v $(pwd)/conf:/app/conf \          # 挂载配置文件
#   -v $(pwd)/processed_data:/app/processed_data \  # 挂载处理后的数据
#   -v $(pwd)/data_preparation/your_hdf5_data:/app/raw_data \ # 挂载原始HDF5数据 (如果需要预处理)
#   -v $(pwd)/outputs:/app/outputs \      # 挂载输出目录
#   --gpus all \                         # 如果使用 GPU
#   tecgpt-forecasting \
#   python src/train.py data.raw_data_dir=/app/raw_data # 示例：运行训练，并覆盖原始数据路径
```
**注意**: Dockerfile 中的路径和挂载卷需要根据你的实际使用情况进行调整，特别是数据路径和模型缓存路径 (`gpt2_cache_docker`)。

## 数据准备

### 1. 数据源

*   原始数据以年度 HDF5 文件的形式提供。
*   每个 HDF5 文件应包含：
    *   TEC 数据: `/ionosphere/TEC` (形状 `[N_times, 41, 71]`, 通常 `N_times=1` 代表单个时间点)
    *   时间坐标: `/coordinates/{year, month, day, hour, day_of_year}` (标量)
    *   空间天气指数: `/space_weather_indices/{Kp_Index, Dst_Index, AE_Index, f107_Index, Ap_Index}` (标量)
*   时间分辨率：假设为 2 小时。

### 2. 预处理脚本 (`data_preparation/preprocess_hdf5.py`)

此脚本负责将原始 HDF5 数据转换为模型训练所需的格式。

**主要步骤**:

1.  **加载数据**: 从指定的年度 HDF5 文件中加载 TEC、时间和空间天气数据。
2.  **特征工程**:
    *   Kp 指数乘以 0.1。
    *   F10.7 指数取 log10。
    *   生成周期性时间特征 (小时、星期几、年积日的 sin/cos 编码，共6个)。
3.  **数据标准化**:
    *   **TEC 数据**: 对每个空间格点 (node) 的 TEC 值，使用训练集 (2013-2019) 计算均值和标准差，并进行 z-score 标准化。
    *   **空间天气指数**: 对每个空间天气指数，使用训练集计算均值和标准差，并进行 z-score 标准化。
    *   所有缩放器参数 (均值、标准差) 保存到 `scaler.pkl` 文件中。
4.  **样本生成**:
    *   使用滑动窗口方法创建输入序列 (X) 和目标序列 (Y)。
    *   **输入 X**: 包含历史 P 步的标准化后的 TEC (1维)、标准化后的空间天气指数 (5维)、周期编码的时间特征 (6维)。总共12个输入特征。形状为 `[Num_Samples, P, N_nodes, C_in=12]`。
    *   **输出 Y**: 包含未来 S 步的**原始尺度**的 TEC 值。形状为 `[Num_Samples, S, N_nodes, 1]`。
5.  **数据划分与保存**:
    *   数据集划分: Train (2013-2019), Val (2020-2021), Test (2022-2025.4)。
    *   将处理好的数据分别保存为 `train.npz`, `val.npz`, `test.npz` 到 `processed_data/` 目录。

**运行预处理**:

在运行脚本前，需要修改 `data_preparation/preprocess_hdf5.py` 文件底部的 `example_config` 中的以下路径：
*   `data_root_dir`: 指向你的 HDF5 文件存放的根目录。HDF5 文件应按年份存放在子目录中 (例如 `your_hdf5_data/2013/file.hdf5`)，或者脚本中的文件查找逻辑需要根据你的实际情况调整。
*   `output_dir`: 指定预处理后的 `.npz` 文件和 `scaler.pkl` 的保存位置 (通常是 `tecGPT-forecasting/processed_data/`)。

```bash
# 切换到 data_preparation 目录
cd data_preparation

# 运行预处理脚本
python preprocess_hdf5.py
```
脚本中包含生成虚拟 HDF5 数据的逻辑，如果找不到真实的 HDF5 文件，它会尝试创建一些用于测试。请确保检查日志输出以确认数据加载和处理是否符合预期。

## 配置管理 (Hydra)

本项目使用 [Hydra](https://hydra.cc/) 进行配置管理。

*   所有配置文件都位于 `conf/` 目录下。
*   `conf/config.yaml` 是主配置文件，它通过 `defaults` 列表组合了数据、模型和训练器的配置。
*   各个组件的详细配置在对应的 YAML 文件中：
    *   `conf/data/default.yaml`: 数据集参数、路径、P/S 窗口大小等。
    *   `conf/model/tec_gpt.yaml`: 模型架构参数、嵌入维度、LLM配置等。
    *   `conf/trainer/default.yaml`: 训练参数、优化器、学习率调度器、早停、检查点等。
*   **覆盖配置**: 可以在命令行中轻松覆盖任何配置参数。例如：
    ```bash
    python src/train.py trainer.epochs=50 data.batch_size=16 model.llm_config.f_layers_to_freeze=4
    ```

## 训练模型

### 脚本: `src/train.py`

主训练脚本通过 Hydra 加载配置，实例化数据模块、模型、缩放器和训练器，然后启动训练过程。

### 运行训练

在项目根目录下运行：

```bash
python src/train.py [配置覆盖参数...]
```
例如，训练10个 epoch，并指定一个 Wandb 实体：
```bash
python src/train.py trainer.epochs=10 wandb.entity="your_wandb_username_or_team"
```

### 特性

*   **Wandb 集成**: 自动记录配置、训练/验证损失、指标和学习率到 Weights & Biases。
*   **检查点**:
    *   自动保存在 Hydra 输出目录的 `checkpoints/` 子目录下。
    *   保存最新的检查点 (`...-last.ckpt`)。
    *   根据验证集上的监控指标保存 top-k 最佳检查点。
*   **从检查点恢复训练**:
    ```bash
    python src/train.py resume_from_checkpoint=/path/to/your/checkpoint.ckpt
    # 或者，如果检查点在之前的 outputs 目录中：
    # python src/train.py resume_from_checkpoint=outputs/YYYY-MM-DD/HH-MM-SS/checkpoints/model-last.ckpt
    ```
*   **早停**: 如果验证集上的指标在一定 epoch 内没有改善，则提前停止训练。

## 评估模型

### 脚本: `src/evaluate.py`

主评估脚本用于加载训练好的模型检查点，并在测试集上评估其性能。

### 运行评估

在项目根目录下运行，并指定要评估的检查点路径：

```bash
python src/evaluate.py checkpoint_path=/path/to/your/model-best.ckpt [配置覆盖参数...]
```
例如：
```bash
python src/evaluate.py checkpoint_path=outputs/some_run_dir/checkpoints/tecllm-epoch=XX-...ckpt wandb.entity="your_entity"
```

### 特性

*   加载指定的模型检查点。
*   在测试集上计算以下指标 (均在原始物理尺度上)：
    *   MAE (Mean Absolute Error)
    *   RMSE (Root Mean Squared Error)
    *   WMAPE (Weighted Mean Absolute Percentage Error)
    *   R² (R-squared / Coefficient of Determination)
    *   可选的逐预测步长 (per-step) MAE 和 RMSE。
*   将评估结果打印到控制台，保存到 Hydra 输出目录下的 `test_metrics.yaml`，并可选地记录到 Wandb。

## 超参数优化 (Optuna)

### 脚本: `src/tune.py` 和 `conf/tune.yaml`

项目集成了 Optuna 用于超参数优化 (HPO)，通过 `hydra-optuna-sweeper` 插件。

*   `conf/tune.yaml`: 定义 Optuna sweeper 的配置 (如优化方向、试验次数、采样器、剪枝器) 以及要调优的超参数及其范围/选择。
*   `src/tune.py`: 包含 Optuna 的 `objective` 函数，该函数会为每一组超参数组合运行一次训练 (可能是缩短版)，并返回要优化的指标。

### 运行超参数优化

要启动 HPO sweep，请使用 `-m` (multirun) 标志运行 `src/tune.py`：

```bash
python src/tune.py -m [配置覆盖参数...]
```
例如，运行20次试验，每次试验训练3个 epoch：
```bash
python src/tune.py -m hydra.sweeper.n_trials=20 trainer.epochs=3 wandb.entity="your_entity"
```

### 特性

*   与 Hydra 无缝集成。
*   支持 Optuna 的各种采样器和剪枝器。
*   可以将整个 sweep 或每个 trial 的结果记录到 Wandb。

## 核心代码模块

*   **`src/models/`**:
    *   `tec_gpt.py`: 定义了 `ST_LLM` (TecGPT) 主模型类，整合了所有嵌入层、PFA-LLM 和预测头。
    *   `pfa_llm.py`: 实现了 `PFA_GPT2` 类，应用部分冻结注意力策略到 `transformers` GPT-2 模型。
    *   `embeddings.py`: 包含所有自定义的嵌入模块 (`TecHistoryEmbedding`, `TimeFeatureEmbedding`, `SpatialEmbedding`, `SpaceWeatherEmbedding`, `FusionLayer`)。
*   **`src/datasets/tec_dataset.py`**:
    *   `TECDataset`: PyTorch `Dataset` 类，用于加载 `preprocess_hdf5.py` 生成的 `.npz` 文件。
    *   `TECDataModule`: 组织数据加载、准备和提供 `DataLoader`。
*   **`src/trainers/tec_trainer.py`**:
    *   `TecTrainer`: 封装了完整的训练、验证和测试循环，包括优化器/调度器配置、损失计算、指标计算 (处理尺度转换)、检查点保存、早停和 Wandb 日志记录。
*   **`src/utils/`**:
    *   `scaler.py`: `StandardScaler` 类，用于从 `scaler.pkl` 加载参数并对模型输出进行逆转换以评估指标。
    *   `metrics.py`: 实现各种评估指标 (MAE, RMSE, WMAPE, R²) 的 PyTorch 函数，支持掩码。
    *   `helpers.py`: 包含通用辅助函数，如 `seed_everything()` 和 `get_device()`。

## 待办事项 / 未来工作

*   [ ] 针对特定区域或事件优化模型配置。
*   [ ] 探索更高级的 LLM 架构或微调策略。
*   [ ] 实现更复杂的空间天气指数或地磁活动特征的嵌入方式。
*   [ ] 扩展到全球 TEC 预测。
*   [ ] 添加更详细的预测结果可视化。

## 许可证

本项目采用 MIT 许可证。详情请见 [LICENSE](LICENSE) 文件。

## 联系方式/引用

*   (请在此处添加您的联系方式或项目引用信息) 