import torch
import torch.nn as nn
import math


class TecHistoryEmbedding(nn.Module):
    def __init__(self, history_len, tec_feat_dim, embed_dim, num_nodes, dropout=0.1):
        super().__init__()
        # 输入: [B, P, N] (tec_feat_dim=1 已被整合到 P)
        # 输出: [B, N, D_embed]
        # 假设 tec_feat_dim 已经是1了，所以输入是 P
        self.mlp = nn.Sequential(
            nn.Linear(history_len, embed_dim * 2),  # P -> 2*D
            nn.GELU(),
            nn.LayerNorm(embed_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),  # 2*D -> D
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.activation = nn.GELU()

    def forward(self, x_tec_hist_scaled):
        # x_tec_hist_scaled shape: [B, P, N]
        B, P, N = x_tec_hist_scaled.shape
        x = x_tec_hist_scaled.permute(0, 2, 1)  # -> [B, N, P]
        x_flat = x.reshape(B * N, P)  # -> [B*N, P]
        embed = self.mlp(x_flat)  # -> [B*N, D_embed]
        embed_reshaped = embed.reshape(B, N, -1)  # -> [B, N, D_embed]
        return self.activation(self.norm(embed_reshaped))


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, num_time_features, embed_dim, num_nodes, dropout=0.1):
        super().__init__()
        self.num_nodes = num_nodes
        # 输入: [B, P, N_time_features=6]
        # 输出: [B, N, D_embed_t]
        self.mlp = nn.Sequential(
            nn.Linear(num_time_features, embed_dim * 2),
            nn.GELU(),
            nn.LayerNorm(embed_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.activation = nn.GELU()

    def forward(self, time_features_hist):
        # time_features_hist shape: [B, P, N_time_features=6]
        last_time_feat = time_features_hist[:, -1, :]  # [B, N_time_features]
        embed = self.mlp(last_time_feat)  # [B, D_embed_t]
        embed_norm = self.norm(embed)
        # 广播到所有节点
        return self.activation(embed_norm.unsqueeze(1).expand(-1, self.num_nodes, -1))


class SpatialEmbedding(nn.Module):
    def __init__(self, n_lat, n_lon, embed_dim, num_nodes, dropout=0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.lat_embed = nn.Embedding(n_lat, embed_dim // 2)
        self.lon_embed = nn.Embedding(n_lon, embed_dim // 2)

        coords = []
        for i in range(n_lat):
            for j in range(n_lon):
                coords.append((i, j))
        self.register_buffer("node_coords", torch.tensor(coords, dtype=torch.long))  # [N_nodes, 2]

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2), nn.GELU(), nn.LayerNorm(embed_dim * 2), nn.Dropout(dropout), nn.Linear(embed_dim * 2, embed_dim)
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.activation = nn.GELU()

    def forward(self, device):
        lat_indices = self.node_coords[:, 0].to(device)
        lon_indices = self.node_coords[:, 1].to(device)

        lat_emb = self.lat_embed(lat_indices)  # [N_nodes, D/2]
        lon_emb = self.lon_embed(lon_indices)  # [N_nodes, D/2]

        coord_emb = torch.cat([lat_emb, lon_emb], dim=-1)  # [N_nodes, D]
        embed = self.mlp(coord_emb)  # [N_nodes, D]
        embed_norm = self.norm(embed)
        return self.activation(embed_norm.unsqueeze(0))  # [1, N_nodes, D]


class SpaceWeatherEmbedding(nn.Module):
    def __init__(self, history_len, num_sw_indices, embed_dim, num_nodes, dropout=0.1):
        super().__init__()
        self.num_nodes = num_nodes
        # 输入: [B, P, N_sw_indices=5]
        # 输出: [B, N, D_embed_sw]
        self.mlp = nn.Sequential(
            nn.Linear(history_len * num_sw_indices, embed_dim * 2),
            nn.GELU(),
            nn.LayerNorm(embed_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.activation = nn.GELU()

    def forward(self, sw_hist):
        # sw_hist shape: [B, P, N_sw_indices=5]
        B = sw_hist.size(0)
        sw_flat = sw_hist.reshape(B, -1)  # [B, P * N_sw_indices]
        embed = self.mlp(sw_flat)  # [B, D_embed_sw]
        embed_norm = self.norm(embed)
        return self.activation(embed_norm.unsqueeze(1).expand(-1, self.num_nodes, -1))


class FusionLayer(nn.Module):
    def __init__(self, d_embed_total, d_llm, dropout=0.1):
        super().__init__()
        self.fusion_mlp = nn.Sequential(
            nn.Linear(d_embed_total, d_llm * 2), nn.GELU(), nn.LayerNorm(d_llm * 2), nn.Dropout(dropout), nn.Linear(d_llm * 2, d_llm)
        )

    def forward(self, *embeddings):
        # embeddings: 可变数量的嵌入张量，每个都是 [B, N, D_embed] 或 [1, N, D_embed]
        if len(embeddings) == 0:
            raise ValueError("At least one embedding is required")

        B = embeddings[0].size(0)

        # 广播所有嵌入到相同的批次大小
        broadcast_embeddings = []
        for emb in embeddings:
            if emb.size(0) == 1:  # 如果是 [1, N, D_embed]，广播到批次大小
                broadcast_embeddings.append(emb.expand(B, -1, -1))
            else:
                broadcast_embeddings.append(emb)

        # 连接所有嵌入
        combined_embed = torch.cat(broadcast_embeddings, dim=-1)
        fused_embed = self.fusion_mlp(combined_embed)
        return fused_embed
