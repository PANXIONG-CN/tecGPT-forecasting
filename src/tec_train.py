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
import wandb
import json
from datetime import datetime
import traceback
import gc

# Hydra imports
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

# 将项目根目录添加到Python路径中
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 修改后的import语句，适应新的模块结构
from utils import util
from models.tecGPT.tec_gpt import tecGPT
from models import get_model_class, MODEL_REGISTRY
from ranger21 import Ranger
from data_preparation.preprocess_data import NodeScaler, FeatureScaler

# 增加CUDA内存配置，尝试避免内存碎片化
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"

torch.cuda.empty_cache()


def seed_environment(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Environment seeded with {seed}")


class SimpleTrainer:
    """一个简化的训练器类，用于组织训练逻辑"""

    def __init__(self, model, scaler, optimizer, scheduler, loss_fn, cfg, device, rank=0):
        self.model = model.to(device)
        self.cfg = cfg

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
        self.use_wandb = cfg.wandb.enable and (not self.use_ddp or self.rank == 0)

        # 如果启用混合精度训练，初始化GradScaler
        self.use_amp = cfg.trainer.use_amp
        if self.use_amp:
            self.grad_scaler = GradScaler()
            if rank == 0:
                print("Using Automatic Mixed Precision (AMP) training")

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
        hydra_output_dir = HydraConfig.get().runtime.output_dir

        if not self.use_ddp or self.rank == 0:
            print(f"Starting training for {self.cfg.trainer.epochs} epochs...")

        for epoch in range(1, self.cfg.trainer.epochs + 1):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            train_loss, train_metrics = self._run_epoch(train_loader, is_training=True)
            val_loss, val_metrics = self._run_epoch(val_loader, is_training=False)

            epoch_duration = time.time() - epoch_start_time

            # 记录到wandb
            if self.use_wandb:
                wandb.log(
                    {
                        "epoch": epoch,
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
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                        "epoch_time": epoch_duration,
                    }
                )

            # 只在主进程中打印信息和保存模型
            if not self.use_ddp or self.rank == 0:
                if epoch % self.cfg.trainer.print_every_epochs == 0 or epoch == 1:
                    print(
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
                    print(f"Validation RMSE improved ({self.best_val_rmse:.4f} --> {val_loss:.4f}). Saving model...")
                    self.best_val_rmse = val_loss

                    # 保存模型时，如果使用DDP，保存module
                    if self.use_ddp:
                        torch.save(self.model.module.state_dict(), os.path.join(hydra_output_dir, "best_model.pth"))
                    else:
                        torch.save(self.model.state_dict(), os.path.join(hydra_output_dir, "best_model.pth"))

                    # 记录最佳模型到wandb
                    if self.use_wandb:
                        wandb.run.summary["best_val_rmse"] = self.best_val_rmse
                        wandb.run.summary["best_epoch"] = epoch

                    self.epochs_no_improve = 0
                else:
                    self.epochs_no_improve += 1
                    print(f"No improvement in validation RMSE for {self.epochs_no_improve} epochs.")
                    if self.epochs_no_improve >= self.cfg.trainer.patience:
                        print(f"Early stopping triggered at epoch {epoch}.")
                        break

            # 如果使用DDP，需要同步所有进程是否需要早停
            if self.use_ddp:
                stop_tensor = torch.tensor([1 if self.epochs_no_improve >= self.cfg.trainer.patience else 0], device=self.device)
                dist.broadcast(stop_tensor, src=0)
                if stop_tensor.item() == 1:
                    break

            torch.cuda.empty_cache()

        if not self.use_ddp or self.rank == 0:
            print(f"\nTraining finished. Best validation RMSE: {self.best_val_rmse:.4f}")
        return training_history


def create_model(cfg: DictConfig):
    """模型工厂函数：根据Hydra配置创建相应的模型实例"""
    model_name = cfg.model.model_name
    model_class = get_model_class(model_name)
    print(f"Creating model: {model_name}")

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


def run_training(cfg: DictConfig):
    """主训练函数"""
    seed_environment(cfg.seed)

    # 获取Hydra输出目录
    hydra_output_dir = HydraConfig.get().runtime.output_dir
    print(f"Logs and models will be saved to: {hydra_output_dir}")

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 启用内存优化选项
    torch.backends.cudnn.benchmark = True

    # 确保CUDA缓存被清空
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("CUDA cache cleared")

    print("Loading dataset...")
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
        print("Dataset loaded.")
    except Exception as e:
        print(f"数据加载失败: {e}")
        return

    print("Instantiating tecGPT model...")
    try:
        model = create_model(cfg)
        print(f"Model instantiated. Trainable parameters: {model.param_num(trainable_only=True):,}")
        print(f"Model instantiated. Total parameters: {model.param_num(trainable_only=False):,}")
    except Exception as e:
        print(f"模型初始化失败: {e}")
        return

    # 优化器
    if cfg.trainer.optimizer_type == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=cfg.trainer.learning_rate, weight_decay=cfg.trainer.weight_decay)
    elif cfg.trainer.optimizer_type == "Ranger":
        optimizer = Ranger(model.parameters(), lr=cfg.trainer.learning_rate, weight_decay=cfg.trainer.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer_type: {cfg.trainer.optimizer_type}")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=cfg.trainer.patience // 2, verbose=True)
    loss_fn = util.RMSE_torch

    trainer = SimpleTrainer(model, scaler, optimizer, scheduler, loss_fn, cfg, device)
    training_history = trainer.train(train_loader, val_loader)

    # --- Final Evaluation on Test Set ---
    print("\nLoading best model for final evaluation on test set...")
    try:
        model.load_state_dict(torch.load(os.path.join(hydra_output_dir, "best_model.pth"), map_location=device))
        print("Best model loaded for testing.")
    except FileNotFoundError:
        print("Warning: best_model.pth not found. Evaluating with the last model state (which might not be the best).")

    model.eval()

    # 现在加载测试数据集
    print("Loading test dataset for evaluation...")
    try:
        test_dataset = util.load_dataset(
            dataset_dir=cfg.dataset.data_dir,
            scaler_path=cfg.dataset.scaler_path,
            batch_size=cfg.trainer.batch_size,
            target_device=str(device),
            load_test=True,  # 只加载测试数据
        )
        test_loader = test_dataset["test_loader"]
        print("Test dataset loaded.")
    except Exception as e:
        print(f"测试数据加载失败: {e}")
        return

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
    print("\n--- Final Test Results (Per Horizon) ---")
    for s_idx in range(cfg.dataset.forecast_len):
        preds_h = torch.cat(test_predictions_all_horizons[s_idx], dim=0)
        targets_h = torch.cat(test_targets_all_horizons[s_idx], dim=0)

        m_test = util.metric(preds_h, targets_h)
        horizon_results = {"horizon": s_idx + 1, "mae": m_test[0], "mape": m_test[1], "rmse": m_test[2], "wmape": m_test[3]}
        test_results_per_horizon.append(horizon_results)
        print(f"Horizon {s_idx+1:02d} - MAE: {m_test[0]:.4f}, RMSE: {m_test[2]:.4f}, MAPE: {m_test[1]:.2f}%, WMAPE: {m_test[3]:.2f}%")

        # 记录测试结果到wandb
        if cfg.wandb.enable:
            wandb.log(
                {
                    f"test_mae_h{s_idx+1}": m_test[0],
                    f"test_rmse_h{s_idx+1}": m_test[2],
                    f"test_mape_h{s_idx+1}": m_test[1],
                    f"test_wmape_h{s_idx+1}": m_test[3],
                    "horizon": s_idx + 1,
                }
            )

    pd.DataFrame(training_history).to_csv(os.path.join(hydra_output_dir, "training_log.csv"), index=False)
    df_test_results = pd.DataFrame(test_results_per_horizon)
    df_test_results.to_csv(os.path.join(hydra_output_dir, "test_results_per_horizon.csv"), index=False)

    avg_test_metrics = df_test_results.drop(columns=["horizon"]).mean()
    print("\n--- Average Test Results (All Horizons) ---")
    print(f"Avg MAE:   {avg_test_metrics['mae']:.4f}")
    print(f"Avg RMSE:  {avg_test_metrics['rmse']:.4f}")
    print(f"Avg MAPE:  {avg_test_metrics['mape']:.2f}%")
    print(f"Avg WMAPE: {avg_test_metrics['wmape']:.2f}%")

    # 记录平均测试指标到wandb
    if cfg.wandb.enable:
        wandb.log(
            {
                "test_avg_mae": avg_test_metrics["mae"],
                "test_avg_rmse": avg_test_metrics["rmse"],
                "test_avg_mape": avg_test_metrics["mape"],
                "test_avg_wmape": avg_test_metrics["wmape"],
            }
        )

    print(f"\nBest validation RMSE achieved during training: {trainer.best_val_rmse:.4f}")
    print(f"Full results saved to: {hydra_output_dir}")

    # 最后清理内存
    if torch.cuda.is_available():
        print("Cleaning up CUDA memory...")
        model = model.cpu()
        del model, train_loader, val_loader, test_loader, scaler
        torch.cuda.empty_cache()
        gc.collect()
        print("Memory cleanup complete")

    return trainer.best_val_rmse  # 返回最佳验证RMSE，用于Optuna优化


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Hydra主函数"""
    print("=== tecGPT Training with Hydra ===")

    # 启用内存优化
    if cfg.device == "cuda" and not cfg.model.get("enable_gradient_checkpointing_llm", False):
        print("Automatically enabling gradient checkpointing for better memory efficiency")
        cfg.model.enable_gradient_checkpointing_llm = True

    # 当使用Optuna时，自动减小批量大小以节省内存
    if cfg.get("use_optuna", False) and cfg.trainer.batch_size > 4:
        original_batch_size = cfg.trainer.batch_size
        cfg.trainer.batch_size = min(cfg.trainer.batch_size, 4)  # 限制为最大4
        print(f"Optuna mode detected: Reducing batch size from {original_batch_size} to {cfg.trainer.batch_size} to save memory")

    print(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # 初始化wandb
    if cfg.wandb.enable:
        hydra_output_dir = HydraConfig.get().runtime.output_dir
        wandb.login(key="b5cc72abb4a307ad59f85bd6e32cb2a636769051")
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.run_name,
            group=cfg.wandb.group,
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            dir=hydra_output_dir,
            job_type="train",
        )
        print("W&B initialized")

    try:
        # 运行训练
        best_val_rmse = run_training(cfg)

        # 如果使用wandb，记录最终结果并关闭
        if cfg.wandb.enable:
            wandb.run.summary["final_best_val_rmse"] = best_val_rmse
            wandb.finish()

        # 最终清理内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    except Exception as e:
        print(f"Training failed with error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        if cfg.wandb.enable and wandb.run is not None:
            wandb.finish()
        # 出错时也清理内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        raise


if __name__ == "__main__":
    torch.cuda.empty_cache()
    script_start_time = time.time()
    main()
    print(f"\nTotal script execution time: {(time.time() - script_start_time)/60:.2f} minutes")
