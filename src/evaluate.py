#!/usr/bin/env python3
"""
模型评估脚本
"""

import torch
import argparse
import os
import pandas as pd
from tqdm import tqdm

from utils import util
from models.tecGPT.tec_gpt import tecGPT


def parse_args():
    parser = argparse.ArgumentParser(description="Model Evaluation")
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved model")
    parser.add_argument("--data_dir", type=str, default="./processed_tec_data", help="Data directory")
    parser.add_argument("--scaler_path", type=str, default="./processed_tec_data/scaler.pkl", help="Scaler path")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--output_dir", type=str, default="./eval_results", help="Output directory for results")

    # Model parameters (should match training)
    parser.add_argument("--num_nodes", type=int, default=2911)
    parser.add_argument("--history_len", type=int, default=12)
    parser.add_argument("--forecast_len", type=int, default=12)
    parser.add_argument("--tec_feat_dim", type=int, default=1)
    parser.add_argument("--sw_feat_dim", type=int, default=5)
    parser.add_argument("--time_feat_dim", type=int, default=6)
    parser.add_argument("--d_embed", type=int, default=128)
    parser.add_argument("--d_llm", type=int, default=768)
    parser.add_argument("--local_gpt2_path", type=str, default="/home/panxiong/tecGPT-forecasting/src/models/tecGPT/gpt2")
    parser.add_argument("--llm_layers_to_use", type=int, default=6)
    parser.add_argument("--U_unfrozen_mha", type=int, default=2)

    return parser.parse_args()


def main():
    args = parse_args()

    print("=== 模型评估 ===")
    print(f"模型路径: {args.model_path}")
    print(f"数据目录: {args.data_dir}")

    # 设备配置
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载数据
    print("加载数据集...")
    dataset_loaders = util.load_dataset(
        dataset_dir=args.data_dir, scaler_path=args.scaler_path, batch_size=args.batch_size, target_device=str(device)
    )

    test_loader = dataset_loaders["test_loader"]
    scaler = dataset_loaders["scaler"]
    print(f"测试集批次数: {test_loader.num_batch}")

    # 创建模型
    print("创建模型...")
    model = tecGPT(
        input_len=args.history_len,
        output_len=args.forecast_len,
        num_nodes=args.num_nodes,
        n_lat=41,
        n_lon=71,
        tec_feat_dim=args.tec_feat_dim,
        sw_feat_dim=args.sw_feat_dim,
        time_feat_dim=args.time_feat_dim,
        d_embed=args.d_embed,
        d_llm=args.d_llm,
        llm_model_local_path=args.local_gpt2_path,
        llm_layers_to_use=args.llm_layers_to_use,
        U_unfrozen_mha=args.U_unfrozen_mha,
        dropout_embed=0.1,
        dropout_llm_out=0.1,
        enable_gradient_checkpointing_llm=False,
        device=str(device),
    ).to(device)

    # 加载模型权重
    print(f"加载模型权重: {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # 评估
    print("开始评估...")
    test_predictions_all_horizons = [[] for _ in range(args.forecast_len)]
    test_targets_all_horizons = [[] for _ in range(args.forecast_len)]

    with torch.no_grad():
        for batch_x, batch_y_raw in tqdm(test_loader.get_iterator(), desc="评估"):
            batch_x = torch.FloatTensor(batch_x).to(device)
            batch_y_raw = torch.FloatTensor(batch_y_raw).to(device)

            pred_scaled = model(batch_x)  # [B, N, S]
            pred_raw = scaler.inverse_transform_tec(pred_scaled)  # [B, N, S]
            target_raw = batch_y_raw.squeeze(-1).permute(0, 2, 1)  # [B, N, S]

            for s_idx in range(args.forecast_len):
                test_predictions_all_horizons[s_idx].append(pred_raw[:, :, s_idx].cpu())
                test_targets_all_horizons[s_idx].append(target_raw[:, :, s_idx].cpu())

    # 计算指标
    print("\n=== 评估结果 ===")
    test_results_per_horizon = []

    for s_idx in range(args.forecast_len):
        preds_h = torch.cat(test_predictions_all_horizons[s_idx], dim=0)
        targets_h = torch.cat(test_targets_all_horizons[s_idx], dim=0)

        m_test = util.metric(preds_h, targets_h)
        horizon_results = {"horizon": s_idx + 1, "mae": m_test[0], "mape": m_test[1], "rmse": m_test[2], "wmape": m_test[3]}
        test_results_per_horizon.append(horizon_results)
        print(f"Horizon {s_idx+1:02d} - MAE: {m_test[0]:.4f}, RMSE: {m_test[2]:.4f}, MAPE: {m_test[1]:.2f}%, WMAPE: {m_test[3]:.2f}%")

    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)

    df_results = pd.DataFrame(test_results_per_horizon)
    result_path = os.path.join(args.output_dir, "evaluation_results.csv")
    df_results.to_csv(result_path, index=False)
    print(f"\n详细结果已保存到: {result_path}")

    # 总结
    avg_metrics = df_results.drop(columns=["horizon"]).mean()
    print("\n=== 平均指标 ===")
    print(f"平均 MAE:   {avg_metrics['mae']:.4f}")
    print(f"平均 RMSE:  {avg_metrics['rmse']:.4f}")
    print(f"平均 MAPE:  {avg_metrics['mape']:.2f}%")
    print(f"平均 WMAPE: {avg_metrics['wmape']:.2f}%")


if __name__ == "__main__":
    main()
