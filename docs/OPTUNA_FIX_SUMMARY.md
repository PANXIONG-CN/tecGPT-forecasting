# Optuna 配置修复总结

## 问题
用户遇到错误：`hydra/sweeper appears more than once in the final defaults list`

## 根本原因
在主配置文件 `conf/config.yaml` 中已经指定了默认的 sweeper，而命令行又重复指定了 sweeper。

## 解决方案

### 1. 修复了配置文件格式
- 修改了 `conf/hydra/sweeper/basic_optuna.yaml` 和 `simple_optuna.yaml`
- 添加了正确的 `@package hydra/sweeper` 头部
- 移动了 `_target_` 字段到顶层

### 2. 修复了主配置文件
- 在 `conf/config.yaml` 中添加了默认的 sweeper 配置
- 这样可以避免 "Missing mandatory value" 错误

### 3. 正确的使用方法

现在您可以使用以下命令：

```bash
# 方法1：使用默认的 basic_optuna sweeper
python src/tec_train.py --multirun dataset=tec_data_specific_tr13-21_v22-23_t23-25_h36_f12_tf_off use_optuna=true optuna_trials=2 trainer.epochs=1

# 方法2：如果要使用不同的 sweeper，使用 override 语法
python src/tec_train.py --multirun dataset=tec_data_specific_tr13-21_v22-23_t23-25_h36_f12_tf_off use_optuna=true optuna_trials=2 'override hydra/sweeper: simple_optuna' trainer.epochs=1
```

### 4. 已安装的依赖
- optuna==3.5.0
- hydra-optuna-sweeper

### 5. 配置文件状态
- ✅ `conf/config.yaml` - 包含默认 sweeper
- ✅ `conf/hydra/sweeper/basic_optuna.yaml` - 修复格式
- ✅ `conf/hydra/sweeper/simple_optuna.yaml` - 简化测试版本
- ✅ `conf/hydra/sweeper/test_optuna.yaml` - PLAN-2 的测试版本

## 推荐命令

对于您的具体需求，推荐使用：

```bash
python src/tec_train.py --multirun dataset=tec_data_specific_tr13-21_v22-23_t23-25_h36_f12_tf_off use_optuna=true optuna_trials=2 trainer.epochs=1
```

这将使用默认的 `basic_optuna` sweeper 配置，包含以下优化参数：
- trainer.learning_rate: loguniform(1e-5, 1e-3)
- trainer.batch_size: categorical(8, 12, 16, 32, 48)
- model.d_embed: categorical(128, 192, 256)
- model.llm_layers_to_use: int(4, 8)
- model.U_unfrozen_mha: int(2, 4)
- trainer.optimizer_type: categorical('AdamW', 'Ranger')
