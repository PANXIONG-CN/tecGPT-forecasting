# src/tec_train.py

import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import argparse
import time
import os
import warnings
import random
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler  # 添加混合精度训练支持

# 假设这些模块在你的PYTHONPATH中，或者tec_train.py与它们在合适的目录结构下
# 例如，如果tec_train.py在src/下，而models和utils是src/的子目录
from utils import util  # 或者 .utils import util
from models.tec_gpt import ST_LLM  # 或者 .models.tec_gpt import ST_LLM

# from ranger21 import Ranger # 如果ranger21.py在src或PYTHONPATH

# 为了简单，如果ranger21.py在项目根目录，你可能需要修改PYTHONPATH或将它复制到src
# 或者，如果 ranger21.py 就在 src/ 目录下:
# from ranger21 import Ranger
# 假设Ranger在util.py的同级目录或更上层可以被导入
try:
    from ranger21 import Ranger
except ImportError:
    # 如果ranger21.py在项目根目录，而此脚本在src/下运行
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))  # 添加项目根到路径
    from ranger21 import Ranger


# 增加CUDA内存配置，尝试避免内存碎片化
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1" # 仅用于调试，正常训练时注释掉

torch.cuda.empty_cache()


def parse_args():
    """参数解析配置"""
    parser = argparse.ArgumentParser(description="tecGPT Training Script")
    # --- 路径和设备 ---
    parser.add_argument("--device", type=str, default="cuda:0", help="Device (cuda:0/cpu)")
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

    # --- 模型超参数 (tecGPT) ---
    parser.add_argument("--d_embed", type=int, default=128, help="Dimension for sub-embeddings")
    parser.add_argument("--d_llm", type=int, default=768, help="LLM hidden dimension (n_embd for GPT2)")
    parser.add_argument(
        "--local_gpt2_path",
        type=str,
        default="/home/panxiong/tecGPT-forecasting/src/models/TEC-LLM/gpt2",  # 请替换为你的实际路径
        help="Path to local GPT-2 model files directory (config.json, pytorch_model.bin)",
    )
    parser.add_argument("--llm_layers_to_use", type=int, default=6, help="Number of GPT layers to USE from pretrained")
    parser.add_argument("--U_unfrozen_mha", type=int, default=2, help="Number of UNfrozen MHA/LN layers in PFA (from end)")
    parser.add_argument("--dropout_embed", type=float, default=0.1, help="Dropout for embedding layers")
    parser.add_argument("--dropout_llm_out", type=float, default=0.1, help="Dropout after LLM and before prediction head")
    parser.add_argument("--enable_gradient_checkpointing_llm", action="store_true", help="Enable gradient checkpointing in LLM")

    # --- 训练器参数 ---
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
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

    return parser.parse_args()


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

    def __init__(self, model, scaler, optimizer, scheduler, loss_fn, args, device):
        self.model = model.to(device)
        self.scaler = scaler
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.args = args
        self.device = device
        self.best_val_mae = float("inf")
        self.epochs_no_improve = 0

        # 如果启用混合精度训练，初始化GradScaler
        self.use_amp = args.use_amp
        if self.use_amp:
            self.grad_scaler = GradScaler()
            print("Using Automatic Mixed Precision (AMP) training")

    def _run_epoch(self, dataloader, is_training=True):
        if is_training:
            self.model.train()
        else:
            self.model.eval()

        epoch_losses = []
        epoch_metrics = {"mape": [], "rmse": [], "wmape": []}

        desc = "Train" if is_training else "Val"
        for batch_x, batch_y_raw in tqdm(dataloader.get_iterator(), desc=f"Epoch {self.current_epoch} {desc}", leave=False):
            batch_x = torch.FloatTensor(batch_x).to(self.device)  # [B, P, N, C_in]
            batch_y_raw = torch.FloatTensor(batch_y_raw).to(self.device)  # [B, S, N, 1]

            if is_training:
                self.optimizer.zero_grad()

            # 使用混合精度训练
            with torch.set_grad_enabled(is_training), autocast(enabled=self.use_amp and is_training):  # 控制梯度计算和混合精度
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
        return avg_loss, avg_metrics

    def train(self, train_loader, val_loader):
        training_history = []
        print(f"Starting training for {self.args.epochs} epochs...")

        for epoch in range(1, self.args.epochs + 1):
            self.current_epoch = epoch  # 用于tqdm描述
            epoch_start_time = time.time()

            train_loss, train_metrics = self._run_epoch(train_loader, is_training=True)
            val_loss, val_metrics = self._run_epoch(val_loader, is_training=False)

            epoch_duration = time.time() - epoch_start_time

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
                torch.save(self.model.state_dict(), os.path.join(self.args.save_dir, "best_model.pth"))
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1
                print(f"No improvement in validation MAE for {self.epochs_no_improve} epochs.")
                if self.epochs_no_improve >= self.args.patience:
                    print(f"Early stopping triggered at epoch {epoch}.")
                    break

            torch.cuda.empty_cache()

        print(f"\nTraining finished. Best validation MAE: {self.best_val_mae:.4f}")
        return training_history


def main():
    args = parse_args()
    seed_environment(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Logs and models will be saved to: {args.save_dir}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 启用内存优化选项
    torch.backends.cudnn.benchmark = True  # 可能提高性能
    args.use_amp = True  # 启用混合精度训练
    args.batch_size = 8  # 确保使用较小的批量大小

    print("Loading dataset...")
    dataset_loaders = util.load_dataset(
        dataset_dir=args.data_dir, scaler_path=args.scaler_path, batch_size=args.batch_size, target_device=str(device)
    )
    train_loader = dataset_loaders["train_loader"]
    val_loader = dataset_loaders["val_loader"]
    test_loader = dataset_loaders["test_loader"]
    scaler = dataset_loaders["scaler"]
    print("Dataset loaded.")

    # 验证从 NPZ 加载的数据形状
    # x_sample_npz, y_sample_npz = next(train_loader.get_iterator())
    # print(f"Sample X shape from NPZ (used by DataLoader): {x_sample_npz.shape} -> Expected [B, P, N, C_in]")
    # print(f"Sample Y shape from NPZ (used by DataLoader): {y_sample_npz.shape} -> Expected [B, S, N, 1]")

    print("Instantiating tecGPT model...")
    # 启用梯度检查点以减少GPU内存使用
    args.enable_gradient_checkpointing_llm = True
    print("Gradient checkpointing enabled to reduce memory usage")

    model = ST_LLM(
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
        device=str(device),
    )  # .to(device) 已在 Trainer 中处理
    print(f"Model instantiated. Trainable parameters: {model.param_num(trainable_only=True):,}")
    print(f"Model instantiated. Total parameters: {model.param_num(trainable_only=False):,}")

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

    pd.DataFrame(training_history).to_csv(os.path.join(args.save_dir, "training_log.csv"), index=False)
    df_test_results = pd.DataFrame(test_results_per_horizon)
    df_test_results.to_csv(os.path.join(args.save_dir, "test_results_per_horizon.csv"), index=False)

    avg_test_metrics = df_test_results.drop(columns=["horizon"]).mean()  # 排除 horizon 列再计算平均值
    print("\n--- Average Test Results (All Horizons) ---")
    print(f"Avg MAE:   {avg_test_metrics['mae']:.4f}")
    print(f"Avg RMSE:  {avg_test_metrics['rmse']:.4f}")
    print(f"Avg MAPE:  {avg_test_metrics['mape']:.2f}%")
    print(f"Avg WMAPE: {avg_test_metrics['wmape']:.2f}%")

    print(f"\nBest validation MAE achieved during training: {trainer.best_val_mae:.4f}")
    print(f"Full results saved to: {args.save_dir}")


if __name__ == "__main__":
    torch.cuda.empty_cache()
    script_start_time = time.time()
    main()
    print(f"\nTotal script execution time: {(time.time() - script_start_time)/60:.2f} minutes")
