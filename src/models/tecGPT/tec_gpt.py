# src/models/tec_gpt.py
import torch
import torch.nn as nn
from .embeddings import TecHistoryEmbedding, TimeFeatureEmbedding, SpatialEmbedding, SpaceWeatherEmbedding, FusionLayer
from .pfa_llm import PFA_GPT2


class tecGPT(nn.Module):
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
        use_time_features: bool = True,  # 新增参数
        d_embed: int = 128,
        d_llm: int = 768,
        llm_model_local_path: str = "/home/panxiong/tecGPT-forecasting/src/models/tecGPT/gpt2",
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
        self.device_str = device
        self.tec_feat_dim_model = tec_feat_dim
        self.sw_feat_dim_model = sw_feat_dim
        self.use_time_features_actual = use_time_features
        self.time_feat_dim_actual_if_used = time_feat_dim

        # 实例化嵌入模块
        self.tec_hist_embed = TecHistoryEmbedding(
            history_len=input_len, tec_feat_dim=tec_feat_dim, embed_dim=d_embed, num_nodes=num_nodes, dropout=dropout_embed
        )
        self.spatial_embed = SpatialEmbedding(n_lat, n_lon, d_embed, num_nodes, dropout=dropout_embed)
        self.sw_embed = SpaceWeatherEmbedding(
            history_len=input_len, num_sw_indices=sw_feat_dim, embed_dim=d_embed, num_nodes=num_nodes, dropout=dropout_embed
        )

        # 有条件地实例化时间特征嵌入
        fusion_input_components = 3  # tec, spatial, sw
        if self.use_time_features_actual and self.time_feat_dim_actual_if_used > 0:
            self.time_feat_embed = TimeFeatureEmbedding(
                num_time_features=self.time_feat_dim_actual_if_used, embed_dim=d_embed, num_nodes=num_nodes, dropout=dropout_embed
            )
            fusion_input_components += 1
        else:
            self.time_feat_embed = None

        self.fusion = FusionLayer(d_embed_total=d_embed * fusion_input_components, d_llm=d_llm, dropout=dropout_embed)

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

        # --- 动态特征提取 ---
        current_idx = 0
        tec_hist_scaled = x_input_tensor[..., current_idx : current_idx + self.tec_feat_dim_model]
        current_idx += self.tec_feat_dim_model

        sw_hist_proc = x_input_tensor[..., current_idx : current_idx + self.sw_feat_dim_model]
        current_idx += self.sw_feat_dim_model

        time_features_hist_input_for_embedding = None
        if self.use_time_features_actual and self.time_feat_dim_actual_if_used > 0:
            if C_in_actual < current_idx + self.time_feat_dim_actual_if_used:
                raise ValueError(f"输入张量 C_in ({C_in_actual}) 对于配置的时间特征太小。至少期望 {current_idx + self.time_feat_dim_actual_if_used}")
            time_features_hist_input_for_embedding = x_input_tensor[..., current_idx : current_idx + self.time_feat_dim_actual_if_used]
            current_idx += self.time_feat_dim_actual_if_used

        if C_in_actual != current_idx:
            raise ValueError(f"输入特征维度 C_in ({C_in_actual}) 与基于模型配置的期望总特征 ({current_idx}) 不匹配。")

        # 为嵌入准备输入
        tec_hist_input = tec_hist_scaled.squeeze(-1)  # 假设 tec_feat_dim_model 为 1
        sw_hist_input = sw_hist_proc[:, :, 0, :]  # [B, P, sw_feat_dim]

        current_device = x_input_tensor.device

        # 生成嵌入
        ep = self.tec_hist_embed(tec_hist_input.to(current_device))
        es = self.spatial_embed(current_device)
        esw = self.sw_embed(sw_hist_input.to(current_device))

        embeddings_to_fuse = [ep, es, esw]

        if self.time_feat_embed is not None and time_features_hist_input_for_embedding is not None:
            time_feat_input_for_embed_sliced = time_features_hist_input_for_embedding[:, :, 0, :]  # 使用第一个节点的数据
            et = self.time_feat_embed(time_feat_input_for_embed_sliced.to(current_device))
            embeddings_to_fuse.append(et)

        # 融合
        h_0 = self.fusion(*embeddings_to_fuse)
        h_l = self.llm(inputs_embeds=h_0)
        y_pred_scaled = self.prediction_head(h_l)

        return y_pred_scaled

    def param_num(self, trainable_only=True):
        """计算模型参数数量"""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())
