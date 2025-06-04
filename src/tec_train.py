# src/tec_train.py

import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import time
import os
import sys
import warnings
import random
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import optuna
import json
from datetime import datetime
import traceback
import gc
import logging

# Hydra imports
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode

# 将项目根目录添加到Python路径中
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 修改后的import语句，适应新的模块结构
from utils import util
from models.tecGPT.tec_gpt import tecGPT
from models import get_model_class, MODEL_REGISTRY
from models.tecGPT.ranger21 import Ranger
from data_preparation.preprocess_data import NodeScaler, FeatureScaler

# 增加CUDA内存配置，尝试避免内存碎片化
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64,expandable_segments:True"
# 添加更多内存管理选项
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # 异步执行以提高性能

# 清理CUDA缓存
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    # 设置内存分数以避免OOM
    torch.cuda.set_per_process_memory_fraction(0.9)  # 使用90%的GPU内存


def get_dataset_version_for_path(cfg: DictConfig) -> str:
    """
    辅助函数，用于从配置中提取一个用于路径的数据集标识符。
    优先使用 cfg.dataset.version_id (如果已由 preprocess_data_hydra.py 生成并写入特定数据集配置)
    其次尝试 cfg.dataset_version_id (如果手动设置在cfg顶层)
    否则尝试从 cfg.dataset.data_dir 推断。
    """
    # 优先级1: 特定数据集配置文件中明确写入的 version_id (由预处理脚本生成)
    if hasattr(cfg.dataset, "version_id") and cfg.dataset.version_id:
        return str(cfg.dataset.version_id)

    # 优先级2: 如果在主配置的dataset部分直接定义了version_id
    dataset_config_name = HydraConfig.get().runtime.choices.get("dataset", "unknown")
    if dataset_config_name.startswith("tec_data_specific_"):
        # 从cfg.dataset.data_dir推断，因为version_id应该在cfg.dataset里，而不是从文件名猜测
        if hasattr(cfg.dataset, "data_dir"):
            # 例如 data_dir: ./processed_tec_data/tr13-21_v22-23_t23-25_h36_f12_tf_off/
            data_dir_parts = cfg.dataset.data_dir.strip("/").split("/")
            if len(data_dir_parts) > 0 and data_dir_parts[-1].startswith("tr"):
                return data_dir_parts[-1]

    # 优先级3: 从 cfg.dataset.data_dir 推断（作为后备）
    if hasattr(cfg.dataset, "data_dir"):
        data_dir_parts = cfg.dataset.data_dir.strip("/").split("/")
        if len(data_dir_parts) > 1 and data_dir_parts[-2] == "processed_tec_data":
            version_candidate = data_dir_parts[-1]
            if "tr" in version_candidate and ("_v" in version_candidate or "_h" in version_candidate):
                return version_candidate

    return "unknown_dataset_version"


def get_effective_output_dir(cfg: DictConfig) -> str:
    """
    根据运行模式决定最终的输出目录
    """
    hydra_cfg = HydraConfig.get()

    if hydra_cfg.mode == RunMode.MULTIRUN and cfg.get("use_optuna", False):
        # Optuna sweep trial: Hydra自动管理输出目录 (e.g., hydra.sweep.dir/hydra.sweep.subdir)
        # hydra_cfg.runtime.output_dir 已经是 trial_X 目录
        return hydra_cfg.runtime.output_dir
    else:
        # 单次运行 (包括 use_optuna=True 但没有 --multirun, 或者 use_optuna=False)
        base_log_dir = f"./logs/{cfg.project_name}/{cfg.model.model_name}/SINGLE_RUNS"

        current_time = datetime.now()
        # 使用更短的时间戳格式，避免在路径中出现冒号
        time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")

        dataset_version_str = get_dataset_version_for_path(cfg)

        run_type_prefix = "optuna_single_trial" if cfg.get("use_optuna", False) else "run"

        # 确保路径组件不包含非法字符，特别是 dataset_version_str
        dataset_version_str_sanitized = dataset_version_str.replace("/", "_").replace(":", "-")

        final_dir = os.path.join(base_log_dir, f"{time_str}_{run_type_prefix}_{dataset_version_str_sanitized}")
        return final_dir


def seed_environment(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_optuna_storage(config):
    """确保 Optuna 存储路径有效"""
    if config.use_optuna and config.optuna_storage and config.optuna_storage.startswith("sqlite:///"):
        # 提取数据库文件路径
        db_path = config.optuna_storage.replace("sqlite:///", "")
        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        print(f"Optuna SQLite 存储目录已创建: {os.path.dirname(os.path.abspath(db_path))}")
    return config.optuna_storage


def setup_optuna_study(config):
    """设置 Optuna study 并确保 SQLite 存储正常工作"""
    try:
        # 确保存储目录存在
        if config.optuna_storage and config.optuna_storage.startswith("sqlite:///"):
            db_path = config.optuna_storage.replace("sqlite:///", "")
            os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

        # 创建存储对象
        storage = None
        if config.optuna_storage:
            storage = optuna.storages.RDBStorage(
                url=config.optuna_storage, engine_kwargs={"connect_args": {"timeout": 60, "check_same_thread": False}}
            )
            logging.info(f"成功创建 Optuna RDBStorage: {config.optuna_storage}")

        # 创建 study
        study = optuna.create_study(
            study_name=config.optuna_study_name,
            storage=storage,
            direction="minimize",
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=config.seed),
        )

        logging.info(f"成功创建 Optuna study: {config.optuna_study_name}")
        return study

    except Exception as e:
        logging.error(f"创建 Optuna study 失败: {e}")
        logging.warning("回退到内存存储...")

        # 回退到内存存储
        study = optuna.create_study(study_name=config.optuna_study_name, direction="minimize", sampler=optuna.samplers.TPESampler(seed=config.seed))
        return study


class SimpleTrainer:
    """一个简化的训练器类，用于组织训练逻辑"""

    def __init__(self, model, scaler, optimizer, scheduler, loss_fn, cfg, device, logger, rank=0):
        self.model = model.to(device)
        self.cfg = cfg
        self.logger = logger

        # 如果使用DDP，包装模型
        self.use_ddp = cfg.get("use_ddp", False)
        self.rank = rank
        if self.use_ddp:
            self.model = DDP(self.model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

        self.scaler = scaler
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.best_val_rmse = float("inf")
        self.epochs_no_improve = 0

        # 如果启用混合精度训练，初始化GradScaler
        self.use_amp = cfg.trainer.use_amp
        if self.use_amp and str(self.device) != "cpu":
            self.grad_scaler = GradScaler()
            if rank == 0:
                self.logger.info("Using Automatic Mixed Precision (AMP) training.")

    def _run_epoch(self, dataloader, is_training=True):
        if is_training:
            self.model.train()
        else:
            self.model.eval()

        epoch_losses = []
        epoch_metrics = {"mae": [], "mape": [], "rmse": [], "wmape": []}

        desc = "Train" if is_training else "Val"
        iterator = dataloader.get_iterator()

        # 主进程显示进度条
        if not self.use_ddp or self.rank == 0:
            iterator = tqdm(iterator, desc=f"Epoch {self.current_epoch} {desc}", leave=False)

        for batch_x, batch_y_raw in iterator:
            batch_x = torch.FloatTensor(batch_x).to(self.device)
            batch_y_raw = torch.FloatTensor(batch_y_raw).to(self.device)

            if is_training:
                self.optimizer.zero_grad()

            # 使用混合精度训练
            with torch.set_grad_enabled(is_training), autocast(enabled=self.use_amp and is_training):
                # 如果使用DDP，直接使用封装的模型
                if self.use_ddp:
                    pred_scaled = self.model.module(batch_x) if is_training else self.model.module(batch_x)
                else:
                    pred_scaled = self.model(batch_x)  # Output: [B, N, S] (standardized)

                pred_raw = self.scaler.inverse_transform_tec(pred_scaled)  # Output: [B, N, S] (raw)
                target_raw = batch_y_raw.squeeze(-1).permute(0, 2, 1)  # -> [B, N, S] (raw)

                loss = self.loss_fn(pred_raw, target_raw, mask_value=util.FILL_VALUE_MASK)

            if is_training:
                if self.use_amp:
                    # 使用GradScaler处理梯度
                    self.grad_scaler.scale(loss).backward()
                    if self.cfg.trainer.clip_grad_norm > 0:
                        self.grad_scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.trainer.clip_grad_norm)
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                else:
                    # 原始FP32训练
                    loss.backward()
                    if self.cfg.trainer.clip_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.trainer.clip_grad_norm)
                    self.optimizer.step()

            epoch_losses.append(loss.item())
            # 计算指标时不需要梯度
            with torch.no_grad():
                m = util.metric(pred_raw, target_raw)
                epoch_metrics["mae"].append(m[0])
                epoch_metrics["mape"].append(m[1])
                epoch_metrics["rmse"].append(m[2])
                epoch_metrics["wmape"].append(m[3])

        avg_loss = np.mean(epoch_losses)
        avg_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}

        # 如果使用DDP，同步所有GPU上的损失和指标
        if self.use_ddp:
            metrics_tensor = torch.tensor(
                [avg_loss, avg_metrics["mae"], avg_metrics["mape"], avg_metrics["rmse"], avg_metrics["wmape"]], device=self.device
            )
            dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
            metrics_tensor /= self.cfg.get("world_size", 1)
            avg_loss = metrics_tensor[0].item()
            avg_metrics["mae"] = metrics_tensor[1].item()
            avg_metrics["mape"] = metrics_tensor[2].item()
            avg_metrics["rmse"] = metrics_tensor[3].item()
            avg_metrics["wmape"] = metrics_tensor[4].item()

        return avg_loss, avg_metrics

    def train(self, train_loader, val_loader):
        training_history = []
        current_output_dir = self.cfg.output_dir_for_this_run

        if not self.use_ddp or self.rank == 0:
            self.logger.info(f"Starting training for {self.cfg.trainer.epochs} epochs...")

        for epoch in range(1, self.cfg.trainer.epochs + 1):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            train_loss, train_metrics = self._run_epoch(train_loader, is_training=True)
            val_loss, val_metrics = self._run_epoch(val_loader, is_training=False)

            epoch_duration = time.time() - epoch_start_time

            # 只在主进程中打印信息和保存模型
            if not self.use_ddp or self.rank == 0:
                if epoch % self.cfg.trainer.print_every_epochs == 0 or epoch == 1:
                    self.logger.info(
                        f"Epoch {epoch}/{self.cfg.trainer.epochs} [{epoch_duration:.2f}s] - "
                        f"Train RMSE: {train_loss:.4f} (MAE metric: {train_metrics['mae']:.4f}), "
                        f"Val RMSE: {val_loss:.4f} (MAE metric: {val_metrics['mae']:.4f})"
                    )

                training_history.append(
                    {
                        "epoch": epoch,
                        "time": epoch_duration,
                        "train_rmse": train_loss,
                        "train_mae": train_metrics["mae"],
                        "train_rmse_metric": train_metrics["rmse"],
                        "train_mape": train_metrics["mape"],
                        "train_wmape": train_metrics["wmape"],
                        "val_rmse": val_loss,
                        "val_mae": val_metrics["mae"],
                        "val_rmse_metric": val_metrics["rmse"],
                        "val_mape": val_metrics["mape"],
                        "val_wmape": val_metrics["wmape"],
                    }
                )

                if self.scheduler:
                    self.scheduler.step(val_loss)

                if val_loss < self.best_val_rmse:
                    self.logger.info(f"Validation RMSE improved ({self.best_val_rmse:.4f} --> {val_loss:.4f}). Saving model...")
                    self.best_val_rmse = val_loss

                    # 保存模型时，如果使用DDP，保存module
                    model_save_path = os.path.join(current_output_dir, "best_model.pth")
                    if self.use_ddp:
                        torch.save(self.model.module.state_dict(), model_save_path)
                    else:
                        torch.save(self.model.state_dict(), model_save_path)

                    self.epochs_no_improve = 0
                else:
                    self.epochs_no_improve += 1
                    self.logger.info(f"No improvement in validation RMSE for {self.epochs_no_improve} epochs.")
                    if self.epochs_no_improve >= self.cfg.trainer.patience:
                        self.logger.info(f"Early stopping triggered at epoch {epoch}.")
                        break

            # 如果使用DDP，需要同步所有进程是否需要早停
            if self.use_ddp:
                stop_tensor = torch.tensor([1 if self.epochs_no_improve >= self.cfg.trainer.patience else 0], device=self.device)
                dist.broadcast(stop_tensor, src=0)
                if stop_tensor.item() == 1:
                    break

            torch.cuda.empty_cache()

        if not self.use_ddp or self.rank == 0:
            self.logger.info(f"\nTraining finished. Best validation RMSE: {self.best_val_rmse:.4f}")
        return training_history


def create_model(cfg: DictConfig):
    """模型工厂函数：根据Hydra配置创建相应的模型实例"""
    model_name = cfg.model.model_name
    model_class = get_model_class(model_name)

    if model_name == "tecGPT":
        model = model_class(
            input_len=cfg.dataset.history_len,
            output_len=cfg.dataset.forecast_len,
            num_nodes=cfg.dataset.num_nodes,
            n_lat=cfg.dataset.n_lat,
            n_lon=cfg.dataset.n_lon,
            tec_feat_dim=cfg.dataset.tec_feat_dim,
            sw_feat_dim=cfg.dataset.sw_feat_dim,
            time_feat_dim=cfg.dataset.time_feat_dim,
            use_time_features=cfg.dataset.get("use_time_features", True),
            d_embed=cfg.model.d_embed,
            d_llm=cfg.model.d_llm,
            llm_model_local_path=cfg.model.local_gpt2_path,
            llm_layers_to_use=cfg.model.llm_layers_to_use,
            U_unfrozen_mha=cfg.model.U_unfrozen_mha,
            dropout_embed=cfg.model.dropout_embed,
            dropout_llm_out=cfg.model.dropout_llm_out,
            enable_gradient_checkpointing_llm=cfg.model.enable_gradient_checkpointing_llm,
            device=str(cfg.device),
        )
    else:
        # 为未来的模型预留接口
        raise NotImplementedError(f"Model {model_name} not yet implemented")

    return model


def run_training(cfg: DictConfig, logger: logging.Logger):
    """主训练函数"""
    seed_environment(cfg.seed)
    logger.info(f"Environment seeded with {cfg.seed}")

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 启用内存优化选项
    torch.backends.cudnn.benchmark = True

    # 确保CUDA缓存被清空
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("CUDA cache cleared")

    logger.info("Loading dataset...")
    try:
        dataset_loaders = util.load_dataset(
            dataset_dir=cfg.dataset.data_dir,
            scaler_path=cfg.dataset.scaler_path,
            batch_size=cfg.trainer.batch_size,
            target_device=str(device),
            load_test=False,  # 训练时不加载测试数据
        )
        train_loader = dataset_loaders["train_loader"]
        val_loader = dataset_loaders["val_loader"]
        scaler = dataset_loaders["scaler"]
        logger.info("Dataset loaded.")
    except Exception as e:
        logger.error(f"数据加载失败: {e}")
        return float("inf")

    logger.info("Instantiating tecGPT model...")
    try:
        model = create_model(cfg)
        logger.info(f"Model instantiated. Trainable parameters: {model.param_num(trainable_only=True):,}")
        logger.info(f"Model instantiated. Total parameters: {model.param_num(trainable_only=False):,}")
    except Exception as e:
        logger.error(f"模型初始化失败: {e}")
        return float("inf")

    # 优化器
    if cfg.trainer.optimizer_type == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=cfg.trainer.learning_rate, weight_decay=cfg.trainer.weight_decay)
    elif cfg.trainer.optimizer_type == "Ranger":
        optimizer = Ranger(model.parameters(), lr=cfg.trainer.learning_rate, weight_decay=cfg.trainer.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer_type: {cfg.trainer.optimizer_type}")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=cfg.trainer.patience // 2, verbose=True)
    loss_fn = util.RMSE_torch

    trainer = SimpleTrainer(model, scaler, optimizer, scheduler, loss_fn, cfg, device, logger)
    training_history = trainer.train(train_loader, val_loader)

    # --- Final Evaluation on Test Set ---
    logger.info("\nLoading best model for final evaluation on test set...")
    try:
        model.load_state_dict(torch.load(os.path.join(cfg.output_dir_for_this_run, "best_model.pth"), map_location=device))
        logger.info("Best model loaded for testing.")
    except FileNotFoundError:
        logger.warning("best_model.pth not found. Evaluating with the last model state (which might not be the best).")

    model.eval()

    # 释放训练和验证数据集以节省内存
    logger.info("Releasing training and validation datasets to free memory...")
    del train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()

    # 现在加载测试数据集
    logger.info("Loading test dataset for evaluation...")
    try:
        test_dataset = util.load_dataset(
            dataset_dir=cfg.dataset.data_dir,
            scaler_path=cfg.dataset.scaler_path,
            batch_size=cfg.trainer.batch_size,
            target_device=str(device),
            load_test=True,  # 只加载测试数据
            load_train_val=False,  # 不加载训练和验证数据
        )
        test_loader = test_dataset["test_loader"]
        logger.info("Test dataset loaded.")
    except Exception as e:
        logger.error(f"测试数据加载失败: {e}")
        return trainer.best_val_rmse

    test_predictions_all_horizons = [[] for _ in range(cfg.dataset.forecast_len)]
    test_targets_all_horizons = [[] for _ in range(cfg.dataset.forecast_len)]

    with torch.no_grad():
        for batch_x_test, batch_y_raw_test in tqdm(test_loader.get_iterator(), desc="Testing"):
            batch_x_test = torch.FloatTensor(batch_x_test).to(device)
            batch_y_raw_test = torch.FloatTensor(batch_y_raw_test).to(device)

            pred_scaled_test = model(batch_x_test)  # [B, N, S]
            pred_raw_test = scaler.inverse_transform_tec(pred_scaled_test)  # [B, N, S]
            target_raw_test = batch_y_raw_test.squeeze(-1).permute(0, 2, 1)  # [B, N, S]

            for s_idx in range(cfg.dataset.forecast_len):
                test_predictions_all_horizons[s_idx].append(pred_raw_test[:, :, s_idx].cpu())
                test_targets_all_horizons[s_idx].append(target_raw_test[:, :, s_idx].cpu())

    test_results_per_horizon = []
    logger.info("\n--- Final Test Results (Per Horizon) ---")
    for s_idx in range(cfg.dataset.forecast_len):
        preds_h = torch.cat(test_predictions_all_horizons[s_idx], dim=0)
        targets_h = torch.cat(test_targets_all_horizons[s_idx], dim=0)

        m_test = util.metric(preds_h, targets_h)
        horizon_results = {"horizon": s_idx + 1, "mae": m_test[0], "mape": m_test[1], "rmse": m_test[2], "wmape": m_test[3]}
        test_results_per_horizon.append(horizon_results)
        logger.info(f"Horizon {s_idx+1:02d} - MAE: {m_test[0]:.4f}, RMSE: {m_test[2]:.4f}, MAPE: {m_test[1]:.2f}%, WMAPE: {m_test[3]:.2f}%")

    pd.DataFrame(training_history).to_csv(os.path.join(cfg.output_dir_for_this_run, "training_log.csv"), index=False)
    df_test_results = pd.DataFrame(test_results_per_horizon)
    df_test_results.to_csv(os.path.join(cfg.output_dir_for_this_run, "test_results_per_horizon.csv"), index=False)

    avg_test_metrics = df_test_results.drop(columns=["horizon"]).mean()
    logger.info("\n--- Average Test Results (All Horizons) ---")
    logger.info(f"Avg MAE:   {avg_test_metrics['mae']:.4f}")
    logger.info(f"Avg RMSE:  {avg_test_metrics['rmse']:.4f}")
    logger.info(f"Avg MAPE:  {avg_test_metrics['mape']:.2f}%")
    logger.info(f"Avg WMAPE: {avg_test_metrics['wmape']:.2f}%")

    logger.info(f"\nBest validation RMSE achieved during training: {trainer.best_val_rmse:.4f}")
    logger.info(f"Full results saved to: {cfg.output_dir_for_this_run}")

    # 最后清理内存
    if torch.cuda.is_available():
        logger.info("Cleaning up CUDA memory...")
        model = model.cpu()
        del model, test_loader, scaler
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("Memory cleanup complete")

    return trainer.best_val_rmse


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> float:
    """Hydra主函数"""
    hydra_cfg = HydraConfig.get()
    is_optuna_multirun = cfg.get("use_optuna", False) and hydra_cfg.mode == RunMode.MULTIRUN

    # 1. 决定并创建输出目录
    output_dir_for_this_run = get_effective_output_dir(cfg)
    os.makedirs(output_dir_for_this_run, exist_ok=True)

    # 暂时不切换目录，先测试基本功能
    # original_cwd = os.getcwd()
    # os.chdir(output_dir_for_this_run)

    # 2. 配置日志记录
    # 获取一个名为当前模块的logger，而不是root logger
    logger = logging.getLogger(__name__)

    # 清除此logger的旧处理器，以防在Jupyter等环境中重复运行导致重复添加
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.INFO)

    # 文件处理器
    log_file_path = os.path.join(output_dir_for_this_run, "run_output.log")
    fh = logging.FileHandler(log_file_path, mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(process)d][%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    # 控制台处理器
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(sh)

    # 阻止日志事件传递给root logger，特别是Hydra的默认控制台输出，避免重复
    logger.propagate = False

    logger.info(f"=== tecGPT Training Run (PID: {os.getpid()}) ===")
    logger.info(f"Hydra Run Mode: {hydra_cfg.mode}")
    logger.info(f"Output directory for this run: {output_dir_for_this_run}")
    logger.info(f"Hydra's .hydra config dir: {os.path.join(output_dir_for_this_run, '.hydra')}")

    # 确保 Optuna 存储路径有效
    if cfg.get("use_optuna", False):
        cfg.optuna_storage = setup_optuna_storage(cfg)
        logger.info(f"Optuna 存储设置完成: {cfg.optuna_storage}")

    # 3. 更新配置中可能被Optuna修改的动态项和内存优化
    if is_optuna_multirun and cfg.trainer.batch_size > 2:
        original_batch_size = cfg.trainer.batch_size
        with open_dict(cfg):
            cfg.trainer.batch_size = min(original_batch_size, 2)  # 进一步减小Optuna时的batch size
        logger.info(f"Optuna multirun mode: Batch size reduced from {original_batch_size} to {cfg.trainer.batch_size}")

    # 自动启用梯度检查点以节省内存
    if cfg.device == "cuda" and not cfg.model.get("enable_gradient_checkpointing_llm", False):
        logger.info("Automatically enabling gradient checkpointing for LLM.")
        with open_dict(cfg):
            cfg.model.enable_gradient_checkpointing_llm = True

    # 自动启用混合精度训练以节省内存
    if cfg.device == "cuda" and not cfg.trainer.get("use_amp", False):
        logger.info("Automatically enabling mixed precision training.")
        with open_dict(cfg):
            cfg.trainer.use_amp = True

    # 对于单次运行，如果batch size太大，自动减小
    if not is_optuna_multirun and cfg.trainer.batch_size > 2:
        original_batch_size = cfg.trainer.batch_size
        with open_dict(cfg):
            cfg.trainer.batch_size = min(original_batch_size, 2)  # 保守的batch size
        logger.info(f"Reduced batch size for memory safety: {original_batch_size} -> {cfg.trainer.batch_size}")

    logger.info(f"Effective Configuration for this run:\n{OmegaConf.to_yaml(cfg)}")

    # 将 output_dir_for_this_run 添加到cfg中
    with open_dict(cfg):
        cfg.output_dir_for_this_run = output_dir_for_this_run

    # 4. 执行训练
    best_val_rmse_result = float("inf")
    try:
        best_val_rmse_result = run_training(cfg, logger)
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"CUDA Out of Memory error: {e}")
        logger.error("Try reducing batch_size, model.llm_layers_to_use, or model.d_embed")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if is_optuna_multirun:
            return float("inf")
        raise
    except RuntimeError as e:
        if "CUDA" in str(e) or "out of memory" in str(e).lower():
            logger.error(f"CUDA Runtime error: {e}")
            logger.error("Try reducing model size or batch size")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if is_optuna_multirun:
                return float("inf")
        raise
    except Exception as e:
        logger.error(f"Training run failed with error: {e}", exc_info=True)
        if is_optuna_multirun:
            return float("inf")
        raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # 移动.hydra目录到正确位置（如果存在且不在Optuna multirun模式下）
        if not is_optuna_multirun:
            project_hydra_dir = ".hydra"
            target_hydra_dir = os.path.join(output_dir_for_this_run, ".hydra")
            if os.path.exists(project_hydra_dir) and not os.path.exists(target_hydra_dir):
                try:
                    import shutil

                    shutil.move(project_hydra_dir, target_hydra_dir)
                    logger.info(f"Moved .hydra directory to: {target_hydra_dir}")
                except Exception as e:
                    logger.warning(f"Failed to move .hydra directory: {e}")

        logger.info(f"Run finished. All artifacts in: {output_dir_for_this_run}")
        # 显式关闭文件处理器，确保日志完全写入
        for handler in logger.handlers:
            handler.close()
            logger.removeHandler(handler)

    # 5. Optuna 返回值
    if is_optuna_multirun:
        # 确保返回有效的float值
        if best_val_rmse_result is None or np.isnan(best_val_rmse_result) or np.isinf(best_val_rmse_result):
            logger.info("Warning: Returning penalty value for failed trial")
            return 999.0  # 返回大的惩罚值而不是inf
        logger.info(f"Returning validation RMSE: {best_val_rmse_result}")
        return float(best_val_rmse_result)

    return 0.0  # 单次运行返回0.0表示成功


if __name__ == "__main__":
    torch.cuda.empty_cache()
    script_start_time = time.time()
    main()
    total_time = (time.time() - script_start_time) / 60
    print(f"\nTotal script execution time: {total_time:.2f} minutes")
