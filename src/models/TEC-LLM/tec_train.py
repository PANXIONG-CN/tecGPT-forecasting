import torch
import numpy as np
import pandas as pd
import argparse
import time
import util
import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "0"
warnings.filterwarnings("ignore")
from util import *
import random
from ST_LLM import ST_LLM
from ranger21 import Ranger

# 增加CUDA内存配置，尝试避免内存碎片化
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# 启用梯度检查点以减少内存使用
torch.cuda.empty_cache()


def parse_args():
    """参数解析配置"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0", help="Device (cuda:0/cpu)")
    parser.add_argument(
        "--data_dir", type=str, default="processed_tec_data", help="Path to the root directory of processed NPZ files (e.g., processed_tec_data)"
    )
    parser.add_argument("--input_dim", type=int, default=5, help="Input features (TEC + 4 aux)")
    parser.add_argument("--num_nodes", type=int, default=2911, help="Graph nodes")
    parser.add_argument("--input_len", type=int, default=12, help="Input sequence length (P)")
    parser.add_argument("--output_len", type=int, default=12, help="Prediction length (S)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")  # 将默认批量大小从64减小到8
    parser.add_argument("--lrate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--llm_layer", type=int, default=1, help="GPT layers")
    parser.add_argument("--U", type=int, default=2, help="Unfrozen attention layers")
    parser.add_argument("--epochs", type=int, default=300, help="Training epochs")
    parser.add_argument("--print_every", type=int, default=50, help="Print interval")
    parser.add_argument("--wdecay", type=float, default=0.0001, help="Weight decay")
    parser.add_argument("--save", type=str, default=f"./logs/{time.strftime('%Y-%m-%d-%H-%M-%S')}-", help="Save path")
    parser.add_argument("--es_patience", type=int, default=100, help="Early stopping patience")
    return parser.parse_args()


def configure_dataset(data_root_dir, num_nodes_arg):
    """
    数据集路径配置.
    现在直接使用 data_root_dir 作为 NPZ 文件所在的目录。
    num_nodes_arg 是从命令行传入的 num_nodes，用于校验或作为默认值。
    """
    # 确保 data_root_dir 是绝对路径或相对于 tecGPT-forecasting 项目根目录的正确路径
    # util.load_dataset 会在 data_root_dir 下寻找 train.npz, val.npz, test.npz
    print(f"Configuring dataset. NPZ files expected in: {os.path.abspath(data_root_dir)}")

    # 我们可以尝试从一个样本 NPZ 文件中读取 num_nodes 来验证或设定
    # 这里假设 train.npz 存在且结构正确
    try:
        sample_npz_path = os.path.join(data_root_dir, "train.npz")
        if os.path.exists(sample_npz_path):
            with np.load(sample_npz_path) as sample_data:
                if "x" in sample_data:
                    # 原始x的形状是 [Num_Samples, P, N_nodes, C_in]
                    # 或者 transpose 后 [Num_Samples, C_in, N_nodes, P]
                    # load_dataset 中 x 的形状是 [Num_Samples, P, N_nodes, C_in] 然后 transpose
                    # 所以 cat_data['x'].shape[2] 应该是 N_nodes
                    actual_num_nodes = sample_data["x"].shape[2]
                    print(f"Detected num_nodes from {sample_npz_path}: {actual_num_nodes}")
                    if num_nodes_arg != actual_num_nodes:
                        print(
                            f"Warning: Command line num_nodes ({num_nodes_arg}) does not match num_nodes in data ({actual_num_nodes}). Using value from data: {actual_num_nodes}."
                        )
                    return os.path.normpath(data_root_dir), actual_num_nodes
                else:
                    print(f"Warning: 'x' key not found in {sample_npz_path}. Using command line num_nodes: {num_nodes_arg}")
                    return os.path.normpath(data_root_dir), num_nodes_arg
        else:
            print(f"Warning: Sample file {sample_npz_path} not found. Using command line num_nodes: {num_nodes_arg}")
            return os.path.normpath(data_root_dir), num_nodes_arg
    except Exception as e:
        print(f"Error reading num_nodes from NPZ file: {e}. Using command line num_nodes: {num_nodes_arg}")
        return os.path.normpath(data_root_dir), num_nodes_arg


class STLLMTrainer:
    """完整的训练器类"""

    def __init__(self, scaler, config, device):
        self.config = config
        self.device = device
        self.scaler = scaler

        self.model = ST_LLM(
            input_dim=config.input_dim,
            num_nodes=config.num_nodes,
            input_len=config.input_len,
            output_len=config.output_len,
            llm_layer=config.llm_layer,
            U=config.U,
            device=device,
        ).to(device)

        self.optimizer = Ranger(params=self.model.parameters(), lr=config.lrate, weight_decay=config.wdecay, gc_conv_only=False)

        self.loss_fn = util.MAE_torch
        self.clip_grad = 5
        self.best_metric = float("inf")

        print(f"\nModel initialized on {device}")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def train_step(self, x, y):
        """单批次训练"""
        self.model.train()
        self.optimizer.zero_grad()

        if x.dim() != 4 or x.shape[1] != self.config.input_dim or x.shape[2] != self.config.num_nodes:
            raise ValueError(f"输入数据形状错误！应为[B,5,N,T]，实际得到{x.shape}")

        x = x.to(self.device)
        y = y.to(self.device)

        output = self.model(x)

        if output.dim() != 3 or output.shape[1] != self.config.num_nodes or output.shape[2] != self.config.output_len:
            raise ValueError(f"模型输出形状错误！应为[B,N,T]，实际得到{output.shape}")

        real = y[:, 0, :, :]  # y的形状是[B,1,N,T]，取第0个特征（TEC）

        if output.shape != real.shape:
            raise RuntimeError(
                f"最终维度不匹配！\n" f"预测形状: {output.shape}\n" f"真实形状: {real.shape}\n" f"输入形状: {x.shape}\n" f"模型输出形状应为[B,N,T]"
            )

        predict = self.scaler.inverse_transform(output)
        loss = self.loss_fn(predict, real, 0.0)

        loss.backward()
        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
        self.optimizer.step()

        with torch.no_grad():
            metrics = {
                "mape": util.MAPE_torch(predict, real, 0.0).item(),
                "rmse": util.RMSE_torch(predict, real, 0.0).item(),
                "wmape": util.WMAPE_torch(predict, real, 0.0).item(),
            }
        return loss.item(), metrics

    def eval_step(self, x, y):
        """单批次验证"""
        self.model.eval()
        with torch.no_grad():
            if x.dim() != 4 or x.shape[1] != self.config.input_dim:
                raise ValueError(f"验证输入数据形状错误！应为[B,5,N,T]，实际得到{x.shape}")

            x = x.to(self.device)
            y = y.to(self.device)

            output = self.model(x)
            real = y[:, 0, :, :]  # y的形状是[B,1,N,T]

            if output.shape != real.shape:
                raise RuntimeError(f"验证阶段维度不匹配！\n" f"预测形状: {output.shape}\n" f"真实形状: {real.shape}\n" f"输入形状: {x.shape}")

            predict = self.scaler.inverse_transform(output)
            loss = self.loss_fn(predict, real, 0.0).item()
            metrics = {
                "mape": util.MAPE_torch(predict, real, 0.0).item(),
                "rmse": util.RMSE_torch(predict, real, 0.0).item(),
                "wmape": util.WMAPE_torch(predict, real, 0.0).item(),
            }
            return loss, metrics


def seed_environment(seed=42):
    """固定随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = parse_args()
    seed_environment()

    # 优化CUDA内存管理
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    data_path, num_nodes_from_data = configure_dataset(args.data_dir, args.num_nodes)
    args.num_nodes = num_nodes_from_data

    print(f"\nLoading dataset from {data_path}...")
    dataset = util.load_dataset(data_path, args.batch_size, args.batch_size, args.batch_size)

    # 数据形状验证
    try:
        sample_x, sample_y = next(iter(dataset["train_loader"].get_iterator()))
    except StopIteration:
        print("Error: Train loader is empty. Check your data files and paths.")
        return

    print("\n=== 数据形状验证 ===")
    print(f"Input data shape from loader: {sample_x.shape} (Expected: [B, {args.input_dim}, {args.num_nodes}, {args.input_len}])")
    print(f"Output data shape from loader: {sample_y.shape} (Expected: [B, 1, {args.num_nodes}, {args.output_len}])")

    assert (
        sample_x.shape[1] == args.input_dim
    ), f"Input feature dimension (C_in) mismatch. Expected {args.input_dim}, got {sample_x.shape[1]} from data loader."
    assert (
        sample_x.shape[2] == args.num_nodes
    ), f"Number of nodes (N_nodes) mismatch. Expected {args.num_nodes}, got {sample_x.shape[2]} from data loader."
    assert (
        sample_x.shape[3] == args.input_len
    ), f"Input sequence length (P_history) mismatch. Expected {args.input_len}, got {sample_x.shape[3]} from data loader."
    assert sample_y.shape[1] == 1, f"Output feature dimension should be 1, got {sample_y.shape[1]} from data loader."
    assert (
        sample_y.shape[2] == args.num_nodes
    ), f"Number of nodes (N_nodes) in output mismatch. Expected {args.num_nodes}, got {sample_y.shape[2]} from data loader."
    assert (
        sample_y.shape[3] == args.output_len
    ), f"Output sequence length (S_forecast) mismatch. Expected {args.output_len}, got {sample_y.shape[3]} from data loader."

    os.makedirs(args.save, exist_ok=True)

    trainer = STLLMTrainer(scaler=dataset["scaler"], config=args, device=device)

    print(f"\nStarting training for {args.epochs} epochs...")
    history = []
    early_stop_counter = 0

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss = []
        train_metrics = {"mape": [], "rmse": [], "wmape": []}

        for batch_idx, (x, y) in enumerate(dataset["train_loader"].get_iterator()):
            try:
                x_tensor = torch.FloatTensor(x).to(device)
                y_tensor = torch.FloatTensor(y).to(device)

                if epoch == 1 and batch_idx == 0:
                    print("\n=== 第一批训练数据 ===")
                    print(f"Input tensor shape to model: {x_tensor.shape} (Expected [B, {args.input_dim}, {args.num_nodes}, {args.input_len}])")
                    print(f"Target tensor shape: {y_tensor.shape} (Expected [B, 1, {args.num_nodes}, {args.output_len}])")

                loss, metrics = trainer.train_step(x_tensor, y_tensor)
                train_loss.append(loss)
                for k in metrics:
                    train_metrics[k].append(metrics[k])

            except Exception as e:
                print(f"\n训练错误 (批次 {batch_idx}): {str(e)}")
                print(f"当前输入形状: {x_tensor.shape}")
                print(f"当前输出形状: {y_tensor.shape}")
                raise

        val_loss = []
        val_metrics = {"mape": [], "rmse": [], "wmape": []}
        with torch.no_grad():
            for x, y in dataset["val_loader"].get_iterator():
                try:
                    x_tensor = torch.FloatTensor(x).to(device)
                    y_tensor = torch.FloatTensor(y).to(device)

                    loss, metrics = trainer.eval_step(x_tensor, y_tensor)
                    val_loss.append(loss)
                    for k in metrics:
                        val_metrics[k].append(metrics[k])

                except Exception as e:
                    print(f"\n验证错误: {str(e)}")
                    raise

        avg_train_loss = np.mean(train_loss)
        avg_val_loss = np.mean(val_loss)
        avg_train_metrics = {k: np.mean(v) for k, v in train_metrics.items()}
        avg_val_metrics = {k: np.mean(v) for k, v in val_metrics.items()}

        history.append(
            {
                "epoch": epoch,
                "time": time.time() - epoch_start,
                "train_loss": avg_train_loss,
                **{f"train_{k}": v for k, v in avg_train_metrics.items()},
                "val_loss": avg_val_loss,
                **{f"val_{k}": v for k, v in avg_val_metrics.items()},
            }
        )

        if epoch % args.print_every == 0 or epoch == 1:
            print(f"\nEpoch {epoch}/{args.epochs} ({history[-1]['time']:.1f}s)")
            print(f"Train Loss: {avg_train_loss:.4f} | MAE: {avg_train_loss:.4f}")
            print(f"Val Loss:   {avg_val_loss:.4f} | MAE: {avg_val_loss:.4f}")

        if avg_val_loss < trainer.best_metric:
            trainer.best_metric = avg_val_loss
            early_stop_counter = 0
            torch.save(trainer.model.state_dict(), os.path.join(args.save, "best_model.pth"))
        else:
            early_stop_counter += 1
            if early_stop_counter >= args.es_patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    print("\nRunning final evaluation...")
    trainer.model.load_state_dict(torch.load(os.path.join(args.save, "best_model.pth")))
    test_results = []

    with torch.no_grad():
        for x, y in dataset["test_loader"].get_iterator():
            x_tensor = torch.FloatTensor(x).to(device)
            y_tensor = torch.FloatTensor(y).to(device)

            output = trainer.model(x_tensor)
            real = y_tensor[:, 0, :, :]
            predict = dataset["scaler"].inverse_transform(output)

            for t in range(args.output_len):
                metrics = util.metric(predict[..., t], real[..., t])
                test_results.append({"horizon": t + 1, "mae": metrics[0], "mape": metrics[1], "rmse": metrics[2], "wmape": metrics[3]})

    pd.DataFrame(history).to_csv(os.path.join(args.save, "training_log.csv"), index=False)
    pd.DataFrame(test_results).to_csv(os.path.join(args.save, "test_results.csv"), index=False)

    print("\nTraining completed!")
    print(f"Best validation loss: {trainer.best_metric:.4f}")
    print(f"Results saved to: {args.save}")
    print(f"Test MAE by horizon:\n{pd.DataFrame(test_results).groupby('horizon')['mae'].mean()}")
    print(f"\nTest RMSE by horizon:\n{pd.DataFrame(test_results).groupby('horizon')['rmse'].mean()}")


if __name__ == "__main__":
    torch.cuda.empty_cache()
    start_time = time.time()
    main()
    print(f"\nTotal execution time: {(time.time() - start_time)/60:.1f} minutes")
