# src/tec_train.py

import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import argparse
import time
import os
import sys  # 添加sys模块导入
import warnings
import random
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler  # 添加混合精度训练支持
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import optuna  # 导入optuna进行超参数优化
import wandb  # 导入wandb进行实验追踪
import json
from datetime import datetime
import traceback
import gc  # 导入gc模块

# 将项目根目录添加到Python路径中
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 修改后的import语句，适应新的模块结构
from utils import util
from models.tecGPT.tec_gpt import ST_LLM
from models import get_model, MODEL_REGISTRY
from ranger21 import Ranger
from data_preparation.preprocess_data import NodeScaler, FeatureScaler  # 解决pickle加载问题

# 增加CUDA内存配置，尝试避免内存碎片化
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1" # 仅用于调试，正常训练时注释掉

torch.cuda.empty_cache()


def parse_args():
    """参数解析配置"""
    parser = argparse.ArgumentParser(description="tecGPT Training Script")
    # --- 路径和设备 ---
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument(
        "--data_dir", type=str, default="./processed_tec_data", help="Path to the directory of processed NPZ files (e.g., ./processed_tec_data)"
    )
    parser.add_argument("--scaler_path", type=str, default="./processed_tec_data/scaler.pkl", help="Path to the saved scaler.pkl file")
    parser.add_argument(
        "--save_dir", type=str, default=f"./logs/tecGPT_{time.strftime('%Y%m%d-%H%M%S')}", help="Directory to save logs and model checkpoints"
    )
    # --- 数据参数 ---
    parser.add_argument("--num_nodes", type=int, default=2911, help="Number of spatial nodes (grid points)")
    parser.add_argument("--n_lat", type=int, default=41, help="Number of latitude grid points")
    parser.add_argument("--n_lon", type=int, default=71, help="Number of longitude grid points")
    parser.add_argument("--history_len", type=int, default=12, help="Input sequence length (P)")
    parser.add_argument("--forecast_len", type=int, default=12, help="Prediction length (S)")
    # --- 特征维度 (C_in = tec + sw + time) ---
    parser.add_argument("--tec_feat_dim", type=int, default=1, help="Dimension of TEC feature in X input (after selection)")
    parser.add_argument("--sw_feat_dim", type=int, default=5, help="Number of Space Weather features in X input")
    parser.add_argument("--time_feat_dim", type=int, default=6, help="Number of cyclical Time features in X input")

    # --- 模型选择 ---
    parser.add_argument("--model_name", type=str, default="tecGPT", choices=["tecGPT"], help="Model to use for training")

    # --- 模型超参数 (tecGPT) ---
    parser.add_argument("--d_embed", type=int, default=128, help="Dimension for sub-embeddings")
    parser.add_argument("--d_llm", type=int, default=768, help="LLM hidden dimension (n_embd for GPT2)")
    parser.add_argument(
        "--local_gpt2_path",
        type=str,
        default="/home/panxiong/tecGPT-forecasting/src/models/tecGPT/gpt2",  # 更新路径
        help="Path to local GPT-2 model files directory (config.json, pytorch_model.bin)",
    )
    parser.add_argument("--llm_layers_to_use", type=int, default=6, help="Number of GPT layers to USE from pretrained")
    parser.add_argument("--U_unfrozen_mha", type=int, default=2, help="Number of UNfrozen MHA/LN layers in PFA (from end)")
    parser.add_argument("--dropout_embed", type=float, default=0.1, help="Dropout for embedding layers")
    parser.add_argument("--dropout_llm_out", type=float, default=0.1, help="Dropout after LLM and before prediction head")
    parser.add_argument("--enable_gradient_checkpointing_llm", action="store_true", help="Enable gradient checkpointing in LLM")

    # --- 训练器参数 ---
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW / Ranger")  # Ranger默认0，AdamW常用0.01
    parser.add_argument("--clip_grad_norm", type=float, default=1.0, help="Gradient clipping norm value")
    parser.add_argument("--print_every_epochs", type=int, default=1, help="Frequency to print training logs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--optimizer_type", type=str, default="AdamW", choices=["AdamW", "Ranger"], help="Optimizer to use")

    # 添加混合精度训练选项
    parser.add_argument("--use_amp", action="store_true", help="使用混合精度训练 (Automatic Mixed Precision)")

    # 添加分布式训练选项
    parser.add_argument("--use_ddp", action="store_true", help="使用DistributedDataParallel进行分布式训练")
    parser.add_argument("--world_size", type=int, default=2, help="分布式训练的总进程数(GPU数)")
    parser.add_argument("--local_rank", type=int, default=-1, help="分布式训练的本地进程序号")

    # 添加Optuna超参数优化选项
    parser.add_argument("--use_optuna", action="store_true", help="使用Optuna进行超参数优化")
    parser.add_argument("--optuna_trials", type=int, default=20, help="Optuna优化的试验次数")
    parser.add_argument("--optuna_study_name", type=str, default=f"tecGPT_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}", help="Optuna study名称")
    parser.add_argument("--optuna_storage", type=str, default=None, help="Optuna storage URL，例如: sqlite:///optuna.db")

    # 添加Weights & Biases (wandb) 记录选项
    parser.add_argument("--use_wandb", action="store_true", help="使用wandb记录训练过程")
    parser.add_argument("--wandb_project", type=str, default="tecGPT-forecasting", help="wandb项目名称")
    parser.add_argument("--wandb_entity", type=str, default=None, help="wandb实体/组织名称")
    parser.add_argument("--wandb_group", type=str, default=None, help="wandb运行组")

    return parser.parse_args()


# 初始化分布式环境
def setup_ddp(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    dist.destroy_process_group()


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

    def __init__(self, model, scaler, optimizer, scheduler, loss_fn, args, device, rank=0):
        self.model = model.to(device)

        # 如果使用DDP，包装模型
        self.use_ddp = args.use_ddp
        self.rank = rank
        if self.use_ddp:
            self.model = DDP(self.model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

        self.scaler = scaler
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.args = args
        self.device = device
        self.best_val_mae = float("inf")
        self.epochs_no_improve = 0
        self.use_wandb = args.use_wandb and (not self.use_ddp or self.rank == 0)

        # 如果启用混合精度训练，初始化GradScaler
        self.use_amp = args.use_amp
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
        epoch_metrics = {"mape": [], "rmse": [], "wmape": []}

        desc = "Train" if is_training else "Val"
        iterator = dataloader.get_iterator()

        # 主进程显示进度条
        if not self.use_ddp or self.rank == 0:
            iterator = tqdm(iterator, desc=f"Epoch {self.current_epoch} {desc}", leave=False)

        for batch_x, batch_y_raw in iterator:
            batch_x = torch.FloatTensor(batch_x).to(self.device)  # [B, P, N, C_in]
            batch_y_raw = torch.FloatTensor(batch_y_raw).to(self.device)  # [B, S, N, 1]

            if is_training:
                self.optimizer.zero_grad()

            # 使用混合精度训练
            with torch.set_grad_enabled(is_training), autocast(enabled=self.use_amp and is_training):  # 控制梯度计算和混合精度
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
                    if self.args.clip_grad_norm > 0:
                        self.grad_scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                else:
                    # 原始FP32训练
                    loss.backward()
                    if self.args.clip_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)
                    self.optimizer.step()

            epoch_losses.append(loss.item())
            # 计算指标时不需要梯度
            with torch.no_grad():
                m = util.metric(pred_raw, target_raw)
                epoch_metrics["mape"].append(m[1])
                epoch_metrics["rmse"].append(m[2])
                epoch_metrics["wmape"].append(m[3])

        avg_loss = np.mean(epoch_losses)
        avg_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}

        # 如果使用DDP，同步所有GPU上的损失和指标
        if self.use_ddp:
            # 创建包含所有指标的tensor
            metrics_tensor = torch.tensor([avg_loss, avg_metrics["mape"], avg_metrics["rmse"], avg_metrics["wmape"]], device=self.device)

            # 跨所有进程同步指标
            dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
            metrics_tensor /= self.args.world_size

            # 更新本地指标
            avg_loss = metrics_tensor[0].item()
            avg_metrics["mape"] = metrics_tensor[1].item()
            avg_metrics["rmse"] = metrics_tensor[2].item()
            avg_metrics["wmape"] = metrics_tensor[3].item()

        return avg_loss, avg_metrics

    def train(self, train_loader, val_loader):
        training_history = []
        if not self.use_ddp or self.rank == 0:
            print(f"Starting training for {self.args.epochs} epochs...")

        for epoch in range(1, self.args.epochs + 1):
            self.current_epoch = epoch  # 用于tqdm描述
            epoch_start_time = time.time()

            train_loss, train_metrics = self._run_epoch(train_loader, is_training=True)
            val_loss, val_metrics = self._run_epoch(val_loader, is_training=False)

            epoch_duration = time.time() - epoch_start_time

            # 记录到wandb
            if self.use_wandb:
                wandb.log(
                    {
                        "epoch": epoch,
                        "train_mae": train_loss,
                        "train_rmse": train_metrics["rmse"],
                        "train_mape": train_metrics["mape"],
                        "train_wmape": train_metrics["wmape"],
                        "val_mae": val_loss,
                        "val_rmse": val_metrics["rmse"],
                        "val_mape": val_metrics["mape"],
                        "val_wmape": val_metrics["wmape"],
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                        "epoch_time": epoch_duration,
                    }
                )

            # 只在主进程中打印信息和保存模型
            if not self.use_ddp or self.rank == 0:
                if epoch % self.args.print_every_epochs == 0 or epoch == 1:
                    print(
                        f"Epoch {epoch}/{self.args.epochs} [{epoch_duration:.2f}s] - "
                        f"Train MAE: {train_loss:.4f} (RMSE: {train_metrics['rmse']:.4f}), "
                        f"Val MAE: {val_loss:.4f} (RMSE: {val_metrics['rmse']:.4f})"
                    )

                training_history.append(
                    {
                        "epoch": epoch,
                        "time": epoch_duration,
                        "train_mae": train_loss,
                        "train_rmse": train_metrics["rmse"],
                        "train_mape": train_metrics["mape"],
                        "train_wmape": train_metrics["wmape"],
                        "val_mae": val_loss,
                        "val_rmse": val_metrics["rmse"],
                        "val_mape": val_metrics["mape"],
                        "val_wmape": val_metrics["wmape"],
                    }
                )

                if self.scheduler:
                    self.scheduler.step(val_loss)  # ReduceLROnPlateau 监控 val_loss

                if val_loss < self.best_val_mae:
                    print(f"Validation MAE improved ({self.best_val_mae:.4f} --> {val_loss:.4f}). Saving model...")
                    self.best_val_mae = val_loss

                    # 保存模型时，如果使用DDP，保存module
                    if self.use_ddp:
                        torch.save(self.model.module.state_dict(), os.path.join(self.args.save_dir, "best_model.pth"))
                    else:
                        torch.save(self.model.state_dict(), os.path.join(self.args.save_dir, "best_model.pth"))

                    # 记录最佳模型到wandb
                    if self.use_wandb:
                        wandb.run.summary["best_val_mae"] = self.best_val_mae
                        wandb.run.summary["best_epoch"] = epoch

                    self.epochs_no_improve = 0
                else:
                    self.epochs_no_improve += 1
                    print(f"No improvement in validation MAE for {self.epochs_no_improve} epochs.")
                    if self.epochs_no_improve >= self.args.patience:
                        print(f"Early stopping triggered at epoch {epoch}.")
                        break

            # 如果使用DDP，需要同步所有进程是否需要早停
            if self.use_ddp:
                # 创建一个表示是否需要早停的tensor
                stop_tensor = torch.tensor([1 if self.epochs_no_improve >= self.args.patience else 0], device=self.device)
                dist.broadcast(stop_tensor, src=0)  # 从rank 0广播早停信号
                if stop_tensor.item() == 1:
                    break

            torch.cuda.empty_cache()

        if not self.use_ddp or self.rank == 0:
            print(f"\nTraining finished. Best validation MAE: {self.best_val_mae:.4f}")
        return training_history


def train_ddp(rank, world_size, args):
    try:
        # 设置分布式环境
        setup_ddp(rank, world_size)

        # 为每个进程设置不同的种子
        seed = args.seed + rank
        seed_environment(seed)

        # 只在主进程初始化wandb
        if rank == 0 and args.use_wandb:
            wandb_run_name = f"tecGPT_ddp_{time.strftime('%Y%m%d-%H%M%S')}"
            config = vars(args).copy()
            config.update(get_model_config(args))
            wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=wandb_run_name, group=args.wandb_group, config=config)

        if rank == 0:
            os.makedirs(args.save_dir, exist_ok=True)
            print(f"Logs and models will be saved to: {args.save_dir}")

        device = torch.device(f"cuda:{rank}")

        # 加载数据集 - 为DDP调整batch size
        # 每个GPU处理原batch size的数据
        per_gpu_batch_size = args.batch_size // world_size
        if per_gpu_batch_size < 1:
            per_gpu_batch_size = 1
            if rank == 0:
                print(f"Warning: Batch size per GPU is less than 1, setting to 1")

        if rank == 0:
            print(f"Loading dataset with batch size {per_gpu_batch_size} per GPU...")

        # 加载数据集
        dataset_loaders = util.load_dataset(
            dataset_dir=args.data_dir, scaler_path=args.scaler_path, batch_size=per_gpu_batch_size, target_device=str(device)
        )

        train_loader = dataset_loaders["train_loader"]
        val_loader = dataset_loaders["val_loader"]
        test_loader = dataset_loaders["test_loader"]
        scaler = dataset_loaders["scaler"]

        if rank == 0:
            print("Dataset loaded.")

        # 初始化模型
        if rank == 0:
            print("Instantiating tecGPT model...")

        # 启用梯度检查点以减少GPU内存使用
        args.enable_gradient_checkpointing_llm = True
        if rank == 0:
            print("Gradient checkpointing enabled to reduce memory usage")

        model = create_model(args)

        if rank == 0:
            print(f"Model instantiated. Trainable parameters: {model.param_num(trainable_only=True):,}")
            print(f"Model instantiated. Total parameters: {model.param_num(trainable_only=False):,}")

        # 优化器
        if args.optimizer_type == "AdamW":
            optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        elif args.optimizer_type == "Ranger":
            optimizer = Ranger(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer_type: {args.optimizer_type}")

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=args.patience // 2, verbose=(rank == 0))
        loss_fn = util.MAE_torch

        trainer = SimpleTrainer(model, scaler, optimizer, scheduler, loss_fn, args, device, rank)
        training_history = trainer.train(train_loader, val_loader)

        # 只在rank 0进行测试评估
        if rank == 0:
            # --- Final Evaluation on Test Set ---
            print("\nLoading best model for final evaluation on test set...")
            try:
                model.load_state_dict(torch.load(os.path.join(args.save_dir, "best_model.pth"), map_location=device))
                print("Best model loaded for testing.")
            except FileNotFoundError:
                print("Warning: best_model.pth not found. Evaluating with the last model state (which might not be the best).")

            model.eval()  # 确保模型在评估模式
            test_predictions_all_horizons = [[] for _ in range(args.forecast_len)]
            test_targets_all_horizons = [[] for _ in range(args.forecast_len)]

            with torch.no_grad():
                for batch_x_test, batch_y_raw_test in tqdm(test_loader.get_iterator(), desc="Testing"):
                    batch_x_test = torch.FloatTensor(batch_x_test).to(device)
                    batch_y_raw_test = torch.FloatTensor(batch_y_raw_test).to(device)

                    pred_scaled_test = model(batch_x_test)  # [B, N, S]
                    pred_raw_test = scaler.inverse_transform_tec(pred_scaled_test)  # [B, N, S]
                    target_raw_test = batch_y_raw_test.squeeze(-1).permute(0, 2, 1)  # [B, N, S]

                    for s_idx in range(args.forecast_len):
                        test_predictions_all_horizons[s_idx].append(pred_raw_test[:, :, s_idx].cpu())
                        test_targets_all_horizons[s_idx].append(target_raw_test[:, :, s_idx].cpu())

            test_results_per_horizon = []
            print("\n--- Final Test Results (Per Horizon) ---")
            for s_idx in range(args.forecast_len):
                preds_h = torch.cat(test_predictions_all_horizons[s_idx], dim=0)
                targets_h = torch.cat(test_targets_all_horizons[s_idx], dim=0)

                m_test = util.metric(preds_h, targets_h)  # metric 函数现在处理 Tensor
                horizon_results = {"horizon": s_idx + 1, "mae": m_test[0], "mape": m_test[1], "rmse": m_test[2], "wmape": m_test[3]}
                test_results_per_horizon.append(horizon_results)
                print(f"Horizon {s_idx+1:02d} - MAE: {m_test[0]:.4f}, RMSE: {m_test[2]:.4f}, MAPE: {m_test[1]:.2f}%, WMAPE: {m_test[3]:.2f}%")

                # 记录测试结果到wandb
                if args.use_wandb:
                    wandb.log(
                        {
                            f"test_mae_h{s_idx+1}": m_test[0],
                            f"test_rmse_h{s_idx+1}": m_test[2],
                            f"test_mape_h{s_idx+1}": m_test[1],
                            f"test_wmape_h{s_idx+1}": m_test[3],
                            "horizon": s_idx + 1,
                        }
                    )

            pd.DataFrame(training_history).to_csv(os.path.join(args.save_dir, "training_log.csv"), index=False)
            df_test_results = pd.DataFrame(test_results_per_horizon)
            df_test_results.to_csv(os.path.join(args.save_dir, "test_results_per_horizon.csv"), index=False)

            avg_test_metrics = df_test_results.drop(columns=["horizon"]).mean()  # 排除 horizon 列再计算平均值
            print("\n--- Average Test Results (All Horizons) ---")
            print(f"Avg MAE:   {avg_test_metrics['mae']:.4f}")
            print(f"Avg RMSE:  {avg_test_metrics['rmse']:.4f}")
            print(f"Avg MAPE:  {avg_test_metrics['mape']:.2f}%")
            print(f"Avg WMAPE: {avg_test_metrics['wmape']:.2f}%")

            # 记录平均测试指标到wandb
            if args.use_wandb:
                wandb.log(
                    {
                        "test_avg_mae": avg_test_metrics["mae"],
                        "test_avg_rmse": avg_test_metrics["rmse"],
                        "test_avg_mape": avg_test_metrics["mape"],
                        "test_avg_wmape": avg_test_metrics["wmape"],
                    }
                )
                wandb.finish()

            print(f"\nBest validation MAE achieved during training: {trainer.best_val_mae:.4f}")
            print(f"Full results saved to: {args.save_dir}")

        # 清理分布式环境
        cleanup_ddp()
    except Exception as e:
        if rank == 0:
            print(f"DDP训练在进程 {rank} 遇到错误: {e}")
            print(f"堆栈跟踪: {traceback.format_exc()}")
        # 尝试清理分布式环境
        try:
            cleanup_ddp()
        except:
            pass
        # 抛出异常让主进程处理
        raise e


def objective(trial, args):
    """
    Optuna优化的目标函数
    """
    # 设置随机种子
    seed = args.seed
    seed_environment(seed)

    # 强制清理GPU内存
    torch.cuda.empty_cache()

    # 重要：在Optuna优化期间禁用DDP，不管原始参数是什么
    original_use_ddp = args.use_ddp  # 保存原始值
    args.use_ddp = False  # 在trial期间禁用DDP

    # 保存原始批量大小，然后设置为较小的值以节省内存
    original_batch_size = args.batch_size
    args.batch_size = 12  # 为Optuna优化使用更合理的批量大小（进一步调整为12）

    # 更新超参数
    # 学习率 (log尺度搜索)
    args.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)

    # 优化器选择
    args.optimizer_type = trial.suggest_categorical("optimizer_type", ["AdamW", "Ranger"])

    # 权重衰减
    args.weight_decay = trial.suggest_float("weight_decay", 1e-5, 0.1, log=True)

    # 梯度裁剪
    args.clip_grad_norm = trial.suggest_float("clip_grad_norm", 0.1, 3.0)

    # 模型相关超参数 - 减小搜索范围以降低内存使用
    args.d_embed = trial.suggest_int("d_embed", 64, 128, step=32)  # 减小上限
    args.dropout_embed = trial.suggest_float("dropout_embed", 0.0, 0.4)  # 增加丢弃率上限
    args.dropout_llm_out = trial.suggest_float("dropout_llm_out", 0.0, 0.4)  # 增加丢弃率上限

    # GPT-2层的使用和解冻 - 减少层数
    args.llm_layers_to_use = trial.suggest_int("llm_layers_to_use", 3, 5)  # 减小上限
    args.U_unfrozen_mha = trial.suggest_int("U_unfrozen_mha", 1, 2)  # 减小上限

    # 启用梯度检查点以减少内存使用
    args.enable_gradient_checkpointing_llm = True

    # 强制开启混合精度训练以节省内存
    args.use_amp = True

    # 优化模型保存路径
    args.save_dir = os.path.join(args.save_dir, f"trial_{trial.number}")
    os.makedirs(args.save_dir, exist_ok=True)

    # 配置wandb
    if args.use_wandb:
        run_name = f"trial_{trial.number}"
        # 初始化wandb，用于追踪单个trial
        config = vars(args).copy()
        config.update(get_model_config(args))
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            group=args.wandb_group or args.optuna_study_name,
            config=config,
            reinit=True,
        )
        # 记录当前trial的超参数
        for key, value in trial.params.items():
            wandb.log({f"param/{key}": value})

    # 加载数据集
    print(f"Loading dataset for trial {trial.number}...")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 修改 util.load_dataset 的调用方式，以便后续清理
    loaded_data_dict = {}  # 用于接收 util.load_dataset 返回的完整字典

    # 创建 DataLoader 实例
    train_loader = util.DataLoader(
        np.load(os.path.join(args.data_dir, "train.npz"))["x"],
        np.load(os.path.join(args.data_dir, "train.npz"))["y"],
        args.batch_size,
        shuffle_on_init=True,
    )
    val_loader = util.DataLoader(
        np.load(os.path.join(args.data_dir, "val.npz"))["x"], np.load(os.path.join(args.data_dir, "val.npz"))["y"], args.batch_size
    )
    # Scaler 仍然像以前一样加载
    scaler_device = str(device) if torch.cuda.is_available() else "cpu"
    scaler = util.StandardScaler(scaler_path=args.scaler_path, device=scaler_device)

    # 初始化模型
    print(f"Instantiating model for trial {trial.number}...")

    model = create_model(args)

    print(f"Trial {trial.number}: Model trainable parameters: {model.param_num(trainable_only=True):,}")

    # 优化器
    if args.optimizer_type == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer_type == "Ranger":
        optimizer = Ranger(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer_type: {args.optimizer_type}")

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=args.patience // 2, verbose=True)

    # 损失函数
    loss_fn = util.MAE_torch

    # 减少训练轮数以加快优化
    original_epochs = args.epochs
    args.epochs = min(args.epochs, 20)  # 最多20轮用于超参数搜索（从30减少到20）
    args.patience = min(args.patience, 8)  # 减小早停耐心值（从10减少到8）

    # 训练
    try:
        trainer = SimpleTrainer(model, scaler, optimizer, scheduler, loss_fn, args, device)
        training_history = trainer.train(train_loader, val_loader)
        best_val_mae = trainer.best_val_mae
    except RuntimeError as e:
        # 捕获OOM等运行时错误，并返回一个非常大的值
        if "CUDA out of memory" in str(e):
            print(f"Trial {trial.number} OOM error, returning penalty value")
            best_val_mae = float("inf")
        else:
            raise e
    finally:
        # 清理内存
        torch.cuda.empty_cache()
        # 显式删除大型数据对象
        del train_loader, val_loader, scaler, model, optimizer, scheduler
        gc.collect()  # 强制垃圾回收

        # 如果使用wandb，结束当前run
        if args.use_wandb and wandb.run is not None:
            wandb.finish()

    # 恢复原始设置
    args.use_ddp = original_use_ddp
    args.batch_size = original_batch_size
    args.epochs = original_epochs

    # 返回最佳验证MAE作为优化指标
    return best_val_mae


def run_optuna_optimization(args):
    """
    使用Optuna进行超参数优化
    """
    print(f"Starting Optuna hyperparameter optimization with {args.optuna_trials} trials...")
    print(f"Study name: {args.optuna_study_name}")

    # 创建catch预设，指定允许捕获的异常
    catch = (RuntimeError, ValueError, torch.cuda.OutOfMemoryError)

    # 创建或加载study
    try:
        if args.optuna_storage:
            # 使用持久化存储
            study = optuna.create_study(
                study_name=args.optuna_study_name,
                storage=args.optuna_storage,
                load_if_exists=True,
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=args.seed),  # 使用TPE采样器并设置种子
            )
        else:
            # 内存中的study
            study = optuna.create_study(
                study_name=args.optuna_study_name, direction="minimize", sampler=optuna.samplers.TPESampler(seed=args.seed)  # 使用TPE采样器并设置种子
            )

        # 设置pruner以提前中止表现不佳的trials
        study.sampler = optuna.samplers.TPESampler(seed=args.seed)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=1)
        study.pruner = pruner

        # 运行优化，设置最大超时
        study.optimize(lambda trial: objective(trial, args), n_trials=args.optuna_trials, catch=catch, timeout=86400)  # 最长运行24小时

        # 打印结果
        if study.best_trial:
            print("\n=== Optuna Optimization Results ===")
            print(f"Best trial: #{study.best_trial.number}")
            print(f"Best value (validation MAE): {study.best_trial.value:.6f}")
            print("Best hyperparameters:")
            for key, value in study.best_trial.params.items():
                print(f"  {key}: {value}")

            # 保存优化结果
            results_dir = os.path.join(args.save_dir, "optuna_results")
            os.makedirs(results_dir, exist_ok=True)

            # 保存最佳超参数
            with open(os.path.join(results_dir, "best_params.json"), "w") as f:
                json.dump(study.best_trial.params, f, indent=2)

            # 保存所有trials信息
            trials_df = study.trials_dataframe()
            trials_df.to_csv(os.path.join(results_dir, "all_trials.csv"), index=False)

            # 生成超参数重要性分析
            try:
                if len(study.trials) >= 10:  # 至少需要10个trials才能进行分析
                    importance = optuna.importance.get_param_importances(study)
                    with open(os.path.join(results_dir, "param_importance.txt"), "w") as f:
                        for param, score in importance.items():
                            print(f"{param}: {score:.4f}", file=f)
                            print(f"Parameter importance: {param}: {score:.4f}")
                else:
                    print("Not enough trials to compute parameter importance.")
            except Exception as e:
                print(f"Could not calculate parameter importances: {e}")

            # 返回最佳超参数
            return study.best_trial.params
        else:
            print("Warning: No successful trials found. Using default parameters.")
            return {}  # 返回空字典，将使用默认参数

    except Exception as e:
        print(f"Optuna optimization failed: {e}")
        print("Falling back to default parameters.")
        return {}  # 返回空字典，将使用默认参数


def main():
    args = parse_args()

    # 检查是否使用Optuna优化超参数
    if args.use_optuna:
        print("使用Optuna进行超参数优化...")
        base_save_dir = args.save_dir  # 保存原始保存路径作为基准目录

        # 暂时禁用DDP以进行Optuna优化
        original_use_ddp = args.use_ddp
        args.use_ddp = False  # 在Optuna期间禁用DDP

        # 运行Optuna优化
        best_params = run_optuna_optimization(args)

        # 使用最佳参数更新args，进行最终训练
        if best_params:  # 检查是否有有效的参数
            for param, value in best_params.items():
                setattr(args, param, value)

            # 恢复原始保存目录和DDP设置，并在其中创建final_model子目录
            args.save_dir = os.path.join(base_save_dir, "final_model")
            args.use_ddp = original_use_ddp
            os.makedirs(args.save_dir, exist_ok=True)

            print("\n=== Training final model with best hyperparameters ===")
            print("Best hyperparameters:")
            for param, value in best_params.items():
                print(f"  {param}: {value}")
        else:
            print("\n=== No valid hyperparameters found, using default settings ===")
            args.save_dir = os.path.join(base_save_dir, "default_model")
            args.use_ddp = original_use_ddp
            os.makedirs(args.save_dir, exist_ok=True)

    # 检查是否使用wandb
    if args.use_wandb and not args.use_ddp:
        # 初始化wandb (单GPU模式)
        wandb_run_name = f"tecGPT_{time.strftime('%Y%m%d-%H%M%S')}"
        if args.use_optuna:
            wandb_run_name += "_final_model"

        config = vars(args).copy()
        config.update(get_model_config(args))
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=wandb_run_name, group=args.wandb_group, config=config)

    # 检查是否使用分布式训练
    if args.use_ddp or "--use_ddp" in sys.argv:
        args.use_ddp = True
        args.world_size = 2  # 双3090环境
        args.use_amp = True  # 启用混合精度训练

        # 使用分布式训练
        try:
            mp.spawn(train_ddp, args=(args.world_size, args), nprocs=args.world_size, join=True)
        except Exception as e:
            print(f"分布式训练失败: {e}")
            print("将回退到单GPU训练...")
            args.use_ddp = False
            # 如果DDP失败，尝试普通单GPU训练
            run_single_gpu_training(args)
    else:
        # 使用原有的单GPU训练逻辑
        run_single_gpu_training(args)


def run_single_gpu_training(args):
    """提取单GPU训练逻辑为单独的函数，便于复用和错误处理"""
    seed_environment(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Logs and models will be saved to: {args.save_dir}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 启用内存优化选项
    torch.backends.cudnn.benchmark = True  # 可能提高性能
    args.use_amp = True  # 启用混合精度训练

    # 根据GPU数量自动调整批量大小
    if torch.cuda.device_count() > 1:
        args.batch_size = 16  # 使用较大的批量大小，因为我们现在有两个GPU

    # 确保批量大小不会导致OOM
    if hasattr(args, "llm_layers_to_use") and args.llm_layers_to_use > 6:
        # 对于大模型，限制批量大小
        args.batch_size = min(args.batch_size, 8)

    print("Loading dataset...")
    try:
        dataset_loaders = util.load_dataset(
            dataset_dir=args.data_dir, scaler_path=args.scaler_path, batch_size=args.batch_size, target_device=str(device)
        )
        train_loader = dataset_loaders["train_loader"]
        val_loader = dataset_loaders["val_loader"]
        test_loader = dataset_loaders["test_loader"]
        scaler = dataset_loaders["scaler"]
        print("Dataset loaded.")
    except Exception as e:
        print(f"数据加载失败: {e}")
        return

    print("Instantiating tecGPT model...")
    # 启用梯度检查点以减少GPU内存使用
    args.enable_gradient_checkpointing_llm = True
    print("Gradient checkpointing enabled to reduce memory usage")

    try:
        model = create_model(args)
        print(f"Model instantiated. Trainable parameters: {model.param_num(trainable_only=True):,}")
        print(f"Model instantiated. Total parameters: {model.param_num(trainable_only=False):,}")
    except Exception as e:
        print(f"模型初始化失败: {e}")
        return

    if args.optimizer_type == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer_type == "Ranger":
        optimizer = Ranger(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer_type: {args.optimizer_type}")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=args.patience // 2, verbose=True)
    loss_fn = util.MAE_torch

    trainer = SimpleTrainer(model, scaler, optimizer, scheduler, loss_fn, args, device)
    training_history = trainer.train(train_loader, val_loader)

    # --- Final Evaluation on Test Set ---
    print("\nLoading best model for final evaluation on test set...")
    try:
        model.load_state_dict(torch.load(os.path.join(args.save_dir, "best_model.pth"), map_location=device))
        print("Best model loaded for testing.")
    except FileNotFoundError:
        print("Warning: best_model.pth not found. Evaluating with the last model state (which might not be the best).")

    model.eval()  # 确保模型在评估模式
    test_predictions_all_horizons = [[] for _ in range(args.forecast_len)]
    test_targets_all_horizons = [[] for _ in range(args.forecast_len)]

    with torch.no_grad():
        for batch_x_test, batch_y_raw_test in tqdm(test_loader.get_iterator(), desc="Testing"):
            batch_x_test = torch.FloatTensor(batch_x_test).to(device)
            batch_y_raw_test = torch.FloatTensor(batch_y_raw_test).to(device)

            pred_scaled_test = model(batch_x_test)  # [B, N, S]
            pred_raw_test = scaler.inverse_transform_tec(pred_scaled_test)  # [B, N, S]
            target_raw_test = batch_y_raw_test.squeeze(-1).permute(0, 2, 1)  # [B, N, S]

            for s_idx in range(args.forecast_len):
                test_predictions_all_horizons[s_idx].append(pred_raw_test[:, :, s_idx].cpu())
                test_targets_all_horizons[s_idx].append(target_raw_test[:, :, s_idx].cpu())

    test_results_per_horizon = []
    print("\n--- Final Test Results (Per Horizon) ---")
    for s_idx in range(args.forecast_len):
        preds_h = torch.cat(test_predictions_all_horizons[s_idx], dim=0)
        targets_h = torch.cat(test_targets_all_horizons[s_idx], dim=0)

        m_test = util.metric(preds_h, targets_h)  # metric 函数现在处理 Tensor
        horizon_results = {"horizon": s_idx + 1, "mae": m_test[0], "mape": m_test[1], "rmse": m_test[2], "wmape": m_test[3]}
        test_results_per_horizon.append(horizon_results)
        print(f"Horizon {s_idx+1:02d} - MAE: {m_test[0]:.4f}, RMSE: {m_test[2]:.4f}, MAPE: {m_test[1]:.2f}%, WMAPE: {m_test[3]:.2f}%")

        # 记录测试结果到wandb
        if args.use_wandb:
            wandb.log(
                {
                    f"test_mae_h{s_idx+1}": m_test[0],
                    f"test_rmse_h{s_idx+1}": m_test[2],
                    f"test_mape_h{s_idx+1}": m_test[1],
                    f"test_wmape_h{s_idx+1}": m_test[3],
                    "horizon": s_idx + 1,
                }
            )

    pd.DataFrame(training_history).to_csv(os.path.join(args.save_dir, "training_log.csv"), index=False)
    df_test_results = pd.DataFrame(test_results_per_horizon)
    df_test_results.to_csv(os.path.join(args.save_dir, "test_results_per_horizon.csv"), index=False)

    avg_test_metrics = df_test_results.drop(columns=["horizon"]).mean()  # 排除 horizon 列再计算平均值
    print("\n--- Average Test Results (All Horizons) ---")
    print(f"Avg MAE:   {avg_test_metrics['mae']:.4f}")
    print(f"Avg RMSE:  {avg_test_metrics['rmse']:.4f}")
    print(f"Avg MAPE:  {avg_test_metrics['mape']:.2f}%")
    print(f"Avg WMAPE: {avg_test_metrics['wmape']:.2f}%")

    # 记录平均测试指标到wandb
    if args.use_wandb:
        wandb.log(
            {
                "test_avg_mae": avg_test_metrics["mae"],
                "test_avg_rmse": avg_test_metrics["rmse"],
                "test_avg_mape": avg_test_metrics["mape"],
                "test_avg_wmape": avg_test_metrics["wmape"],
            }
        )
        wandb.finish()

    print(f"\nBest validation MAE achieved during training: {trainer.best_val_mae:.4f}")
    print(f"Full results saved to: {args.save_dir}")


def create_model(args):
    """模型工厂函数：根据参数创建相应的模型"""
    model_class = get_model(args.model_name)

    # 根据模型类型创建不同的参数配置
    if args.model_name == "tecGPT":
        model = model_class(
            input_len=args.history_len,
            output_len=args.forecast_len,
            num_nodes=args.num_nodes,
            n_lat=args.n_lat,
            n_lon=args.n_lon,
            tec_feat_dim=args.tec_feat_dim,
            sw_feat_dim=args.sw_feat_dim,
            time_feat_dim=args.time_feat_dim,
            d_embed=args.d_embed,
            d_llm=args.d_llm,
            llm_model_local_path=args.local_gpt2_path,
            llm_layers_to_use=args.llm_layers_to_use,
            U_unfrozen_mha=args.U_unfrozen_mha,
            dropout_embed=args.dropout_embed,
            dropout_llm_out=args.dropout_llm_out,
            enable_gradient_checkpointing_llm=args.enable_gradient_checkpointing_llm,
            device=str(args.device),
        )
    else:
        # 为未来的模型预留接口
        raise NotImplementedError(f"Model {args.model_name} not yet implemented")

    return model


def get_model_config(args):
    """获取模型配置信息，用于wandb和optuna记录"""
    config = {
        "model_name": args.model_name,
        "model_type": args.model_name,
        "num_nodes": args.num_nodes,
        "history_len": args.history_len,
        "forecast_len": args.forecast_len,
    }

    # 根据模型类型添加特定配置
    if args.model_name == "tecGPT":
        config.update(
            {
                "d_embed": args.d_embed,
                "d_llm": args.d_llm,
                "llm_layers_to_use": args.llm_layers_to_use,
                "U_unfrozen_mha": args.U_unfrozen_mha,
                "dropout_embed": args.dropout_embed,
                "dropout_llm_out": args.dropout_llm_out,
                "enable_gradient_checkpointing_llm": args.enable_gradient_checkpointing_llm,
            }
        )

    return config


if __name__ == "__main__":
    torch.cuda.empty_cache()
    script_start_time = time.time()
    main()
    print(f"\nTotal script execution time: {(time.time() - script_start_time)/60:.2f} minutes")
