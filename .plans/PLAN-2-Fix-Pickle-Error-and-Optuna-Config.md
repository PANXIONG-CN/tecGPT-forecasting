# 计划：修复Pickle反序列化错误并调整Optuna与Hydra的配置

**ID:** 2
**状态:** [✅] 已完成 (2025-06-04)
**模块:** `src/utils/util.py`, `create_scaler.py`, `conf/config_optuna_test.yaml`
**目标:** 解决 `DummyNodeScaler` 相关的Pickle反序列化错误，并确保Hydra正确配置以使用Optuna Sweeper进行小数据集测试。

## 1. 前期准备与分析

*   **[x] 审查用户提供的研究结果**：已详细阅读并理解用户关于 `DummyNodeScaler` 反序列化错误、Hydra及Optuna Sweeper配置问题的分析和解决方案。
*   **[ ] 确认关键文件**：
    *   `src/utils/util.py`：将在此处统一定义 `DummyScaler`。
    *   `create_scaler.py`：将修改此类以从 `src/utils/util.py` 导入 `DummyScaler`。
    *   `conf/config_optuna_test.yaml`：将修改此配置文件以正确指定Optuna sweeper并启用Optuna。
    *   `small_test_data/scaler.pkl`：需要根据修复方案重新生成此文件。
    *   `src/tec_train.py`：作为测试的执行入口。

## 2. 详细实施步骤

### 操作 2.1: 解决 `DummyNodeScaler` Pickle 反序列化错误

*   **[ ] 任务 2.1.1: 将 `DummyScaler` 类定义移动到 `src/utils/util.py`**
    *   **位置:** `src/utils/util.py`
    *   **修改内容:**
        1.  在文件顶部（例如，在 `DataLoader` 类定义之前）添加 `DummyScaler` 类的完整定义，如用户研究中所述。
            ```python
            # src/utils/util.py
            import numpy as np # 确保导入
            # ... 其他导入 ...

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

            # class DataLoader(object):
            # ... (现有代码)
            ```
    *   **预期结果:** `DummyScaler` 类在 `pickle.load` 操作时对 `src/utils/util.StandardScaler` 可见。

*   **[ ] 任务 2.1.2: 修改 `create_scaler.py` 以从 `src/utils/util.py` 导入 `DummyScaler`**
    *   **位置:** `create_scaler.py` (位于项目根目录)
    *   **修改内容:**
        1.  删除本地（`create_scaler.py` 文件内部）的 `DummyScaler` 类定义。
        2.  添加从 `src.utils.util` 导入 `DummyScaler` 的语句。
            ```python
            # create_scaler.py
            import pickle
            import numpy as np
            from src.utils.util import DummyScaler # <--- 新增导入

            # (此处原有的 DummyScaler 类定义需要被删除)

            scaler_data = {"tec_scaler": DummyScaler(2911), "sw_scaler": DummyScaler(5)}

            with open("./small_test_data/scaler.pkl", "wb") as f:
                pickle.dump(scaler_data, f)

            print("Scaler created successfully using DummyScaler from utils")
            ```
    *   **预期结果:** `create_scaler.py` 使用 `src/utils/util.py` 中定义的 `DummyScaler` 来生成 `scaler.pkl` 文件。

### 操作 2.2: 解决 Hydra 与 Optuna Sweeper 配置问题

*   **[ ] 任务 2.2.1: 修改 `conf/config_optuna_test.yaml`**
    *   **位置:** `conf/config_optuna_test.yaml`
    *   **修改内容:**
        1.  在 `defaults` 列表中添加 `- hydra/sweeper: test_optuna` 来明确指定使用的sweeper配置。
        2.  确认 `use_optuna: true`。
        3.  确认 `optuna_trials` 设置为合理的值（例如3次用于快速测试）。
        4.  确认 `optuna_study_name` 具有唯一的模板。
        5.  确认 `optuna_storage` 指向一个有效的 SQLite 数据库路径 (例如 `sqlite:///./logs/optuna_test_study.db`)。
            ```yaml
            # conf/config_optuna_test.yaml
            # 专门用于Optuna测试的配置
            defaults:
              - model: tecGPT
              - dataset: small_test_data 
              - trainer: test 
              - hydra/sweeper: test_optuna # <--- 确保此行存在并正确
              - _self_

            # ... (project_name, seed, device 保持不变) ...

            hydra:
              run:
                dir: "." 
              sweep:
                dir: ./logs/optuna_test/${now:%Y-%m-%d_%H-%M-%S}_${optuna_study_name}
                subdir: trial_${hydra.job.num}
              # ... (job_logging, output_subdir 保持不变) ...

            # Optuna 配置 - 明确启用
            use_optuna: true
            optuna_trials: 3 
            optuna_study_name: "test_study_${now:%Y%m%d_%H%M%S}"
            optuna_storage: "sqlite:///./logs/optuna_test_study.db" # 数据库路径
            ```
    *   **预期结果:** 当使用此配置文件运行 `src/tec_train.py --multirun` 时，Hydra 会加载 `conf/hydra/sweeper/test_optuna.yaml`，从而使用 `OptunaSweeper`，Optuna会执行指定次数的trials，并将结果存储在指定的数据库中。

### 操作 2.3: 测试与验证流程

*   **[ ] 任务 2.3.1: 重新生成 `small_test_data/scaler.pkl` 文件**
    *   **操作:** 运行修改后的 `python create_scaler.py` 命令。
    *   **验证:** 确认 `./small_test_data/scaler.pkl` 文件被成功创建或更新。

*   **[ ] 任务 2.3.2: 运行 Optuna 测试**
    *   **操作:** 在项目根目录下执行以下命令：
        ```bash
        python src/tec_train.py --config-name config_optuna_test --multirun trainer.epochs=1 optuna_trials=2
        ```
        (使用较少的 `optuna_trials` 和 `trainer.epochs` 以加速测试)。
    *   **验证:**
        1.  **日志输出**: 检查终端输出，确认没有出现 `Can't get attribute 'DummyNodeScaler'` 或其他Pickle相关的错误。确认Optuna的trial日志正常输出。
        2.  **目录结构**:
            *   检查 `logs/optuna_test/` 目录下是否生成了类似 `YYYY-MM-DD_HH-MM-SS_test_study_YYYYMMDD_HHMMSS/` 的文件夹。
            *   在该文件夹内，是否生成了 `trial_0`, `trial_1` 等子目录（根据 `optuna_trials` 数量）。
        3.  **数据库文件**: 检查 `logs/optuna_test_study.db` 文件是否被创建（或 `optuna_storage` 指定的其他路径）。
        4.  **Hydra Sweeper 配置**: 检查任意一个trial目录 (例如 `trial_0/.hydra/`) 下的 `config.yaml` 或 `multirun.yaml`，确认 `hydra.sweeper._target_` 的值是 `hydra_plugins.hydra_optuna_sweeper.OptunaSweeper`。

## 3. 潜在风险与应对

*   **Python环境与Pickle兼容性**:
    *   **风险**: 极小概率下，如果生成 `scaler.pkl` 的Python环境与运行 `tec_train.py` 的环境有显著差异（例如，不同的Python主版本或Numpy版本），可能仍有反序列化问题。
    *   **应对**: 确保在同一Python虚拟环境（或容器）中执行 `create_scaler.py` 和 `src/tec_train.py`。
*   **`create_scaler.py` 的运行路径**:
    *   **风险**: 如果 `create_scaler.py` 不是在项目根目录下运行，`from src.utils.util import DummyScaler` 可能会失败。
    *   **应对**: 明确指导用户在项目根目录下运行 `python create_scaler.py`。
*   **`scaler.pkl` 文件来源混淆**:
    *   **风险**: 用户研究中提到 `scaler.pkl` 可能由 `create_small_test_dataset.py`（使用 `NodeScaler` 和 `FeatureScaler`）或 `create_scaler.py`（计划修改为使用 `DummyScaler`）生成。如果混淆，测试可能不会针对预期的 `DummyScaler` 错误。
    *   **应对**: 本计划的任务 2.1.2 明确修改 `create_scaler.py` 使用 `DummyScaler`，任务 2.3.1 指导运行此脚本。这确保了 `scaler.pkl` 包含的是 `DummyScaler` 实例，与待修复的错误场景一致。
*   **Optuna数据库写入权限**:
    *   **风险**: 如果 `logs/` 目录或其子目录没有写入权限，Optuna无法保存数据库。
    *   **应对**: 确保运行用户对 `./logs/` 目录有写入权限。

## 4. 测试策略

*   **组件测试 (隐式)**:
    *   `create_scaler.py` 能够成功运行并生成 `scaler.pkl`，表明 `DummyScaler` 导入和使用正确。
    *   `src/tec_train.py` 在初始化 `StandardScaler` 时能够成功加载由 `create_scaler.py` 生成的 `scaler.pkl` 而不抛出 `DummyNodeScaler` 错误。
*   **集成测试 (核心)**:
    *   执行 `src/tec_train.py --config-name config_optuna_test --multirun ...` 命令。
    *   **成功标准**:
        1.  程序完成所有trials而没有因Pickle错误或Hydra配置错误提前终止。
        2.  Optuna正确执行了指定数量的trials。
        3.  SQLite数据库 (`.db`) 文件被创建并包含study数据。
        4.  Hydra的输出目录结构符合预期。
        5.  日志中明确显示 `OptunaSweeper` 被使用。

## 5. Mermaid图表

对于此特定修复任务，流程相对直接，Mermaid图表可能不是必需的。关键在于确保文件修改和测试命令的正确性。

## 6. 清理与后续

*   **[ ] 确认所有测试通过后，可以考虑是否保留 `DummyScaler` 或将其重命名以更清晰地表示其测试用途。**
*   **[ ] 如果 `create_small_test_dataset.py` 也生成 `scaler.pkl`，确保用户清楚何时使用哪个脚本生成的 `scaler.pkl`，或者统一scaler的生成逻辑。**

**重要提示**：在应用任何修改前，建议备份相关文件。
