# src/models/tec_gpt.py
import torch
import torch.nn as nn
from .embeddings import TecHistoryEmbedding, TimeFeatureEmbedding, SpatialEmbedding, SpaceWeatherEmbedding, FusionLayer
from .pfa_llm import PFA_GPT2


class ST_LLM(nn.Module):
    def __init__(
        self,
        input_len: int,
        output_len: int,
        num_nodes: int,
        n_lat: int = 41,
        n_lon: int = 71,
        tec_feat_dim: int = 1,
        sw_feat_dim: int = 5,
        time_feat_dim: int = 6,
        d_embed: int = 128,
        d_llm: int = 768,
        llm_model_local_path: str = "/home/panxiong/tecGPT-forecasting/src/models/tecGPT/gpt2",  # 更新默认路径
        llm_layers_to_use: int = 6,
        U_unfrozen_mha: int = 2,
        dropout_embed: float = 0.1,
        dropout_llm_out: float = 0.1,
        enable_gradient_checkpointing_llm: bool = False,
        device: str = "cuda:0",
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.output_len = output_len
        self.device_str = device  # 保存设备字符串
        self.sw_feat_dim = sw_feat_dim
        self.time_feat_dim = time_feat_dim

        # 实例化嵌入模块
        self.tec_hist_embed = TecHistoryEmbedding(
            history_len=input_len, tec_feat_dim=tec_feat_dim, embed_dim=d_embed, num_nodes=num_nodes, dropout=dropout_embed
        )
        self.time_feat_embed = TimeFeatureEmbedding(num_time_features=time_feat_dim, embed_dim=d_embed, num_nodes=num_nodes, dropout=dropout_embed)
        self.spatial_embed = SpatialEmbedding(n_lat, n_lon, d_embed, num_nodes, dropout=dropout_embed)
        self.sw_embed = SpaceWeatherEmbedding(
            history_len=input_len, num_sw_indices=sw_feat_dim, embed_dim=d_embed, num_nodes=num_nodes, dropout=dropout_embed
        )

        self.fusion = FusionLayer(d_embed_total=d_embed * 4, d_llm=d_llm, dropout=dropout_embed)

        self.llm = PFA_GPT2(
            local_gpt2_model_path=llm_model_local_path,
            gpt_layers_to_use=llm_layers_to_use,
            U_unfrozen_mha=U_unfrozen_mha,
            num_spatial_nodes=num_nodes,
            d_llm_embedding=d_llm,
            enable_gradient_checkpointing=enable_gradient_checkpointing_llm,
        )

        self.prediction_head = nn.Sequential(
            nn.Linear(d_llm, d_llm // 2), nn.GELU(), nn.LayerNorm(d_llm // 2), nn.Dropout(dropout_llm_out), nn.Linear(d_llm // 2, output_len)
        )
        self._reset_non_llm_parameters()

    def _reset_non_llm_parameters(self):
        print("Initializing non-LLM parameters...")
        for name, param in self.named_parameters():
            if "llm.gpt2." not in name:  # 不重新初始化已加载的LLM权重或PFA_GPT2内部已初始化的wpe
                if param.dim() > 1:
                    if "weight" in name and "embed" not in name:  # Xavier for Linear/Conv weights
                        nn.init.xavier_uniform_(param)
                    elif "embed" in name and "weight" in name:  # Normal for Embedding weights
                        nn.init.normal_(param, std=0.02)
                elif "bias" in name:
                    nn.init.zeros_(param)

    def forward(self, x_input_tensor):
        if x_input_tensor.dim() != 4:
            raise ValueError(f"tecGPT input must be 4D [B, P, N, C_in], got {x_input_tensor.dim()}D {x_input_tensor.shape}")

        B, P, N, C_in_actual = x_input_tensor.shape

        # --- 动态确定切片索引 ---
        idx_tec_end = self.tec_hist_embed.mlp[0].in_features // P if hasattr(self.tec_hist_embed, "mlp") else 1  # 推断 tec_feat_dim
        idx_sw_start = idx_tec_end
        idx_sw_end = idx_sw_start + self.sw_feat_dim
        idx_time_start = idx_sw_end
        idx_time_end = idx_time_start + self.time_feat_dim

        if C_in_actual != idx_time_end:
            raise ValueError(
                f"Input feature dimension C_in ({C_in_actual}) does not match sum of "
                f"tec_feat_dim ({idx_tec_end}), sw_feat_dim ({self.sw_feat_dim}), "
                f"and time_feat_dim ({self.time_feat_dim}). Expected sum: {idx_time_end}"
            )

        tec_hist_scaled = x_input_tensor[..., :idx_tec_end]  # [B, P, N, tec_feat_dim]
        sw_hist_proc = x_input_tensor[..., idx_sw_start:idx_sw_end]  # [B, P, N, sw_feat_dim]
        time_features_hist = x_input_tensor[..., idx_time_start:idx_time_end]  # [B, P, N, time_feat_dim]

        # 对于共享特征（SW, Time），它们在N维度上是重复的，取第一个节点即可
        # permute and squeeze tec_hist
        tec_hist_input = tec_hist_scaled.squeeze(-1)  # if tec_feat_dim is 1 -> [B, P, N]

        sw_hist_input = sw_hist_proc[:, :, 0, :]  # -> [B, P, sw_feat_dim]
        time_feat_input = time_features_hist[:, :, 0, :]  # -> [B, P, time_feat_dim]

        current_device = x_input_tensor.device  # 确保所有操作在同一设备

        ep = self.tec_hist_embed(tec_hist_input.to(current_device))
        et = self.time_feat_embed(time_feat_input.to(current_device))
        es = self.spatial_embed(current_device)  # SpatialEmbedding.forward 只接收 device
        esw = self.sw_embed(sw_hist_input.to(current_device))

        h_0 = self.fusion(ep, et, es, esw)
        h_l = self.llm(inputs_embeds=h_0)
        y_pred_scaled = self.prediction_head(h_l)

        return y_pred_scaled

    def param_num(self, trainable_only=True):
        """计算模型参数数量"""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())
