import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Config
import os


class TemporalEmbedding(nn.Module):
    def __init__(self, time_dim=288, embed_dim=256):
        """时间嵌入层"""
        super().__init__()
        self.time_dim = time_dim
        self.embed_dim = embed_dim

        self.day_embed = nn.Embedding(7, embed_dim)
        self.time_embed = nn.Embedding(time_dim, embed_dim)
        self.pos_encoder = nn.Parameter(torch.randn(1, embed_dim, 1, 1))

        nn.init.xavier_uniform_(self.day_embed.weight)
        nn.init.xavier_uniform_(self.time_embed.weight)
        nn.init.normal_(self.pos_encoder, mean=0, std=0.02)

    def forward(self, x):
        """
        输入: [B, C, N, T]
        输出: [B, embed_dim, N, T]
        """
        B, _, N, T = x.shape
        device = x.device

        time_steps = torch.arange(T, device=device, dtype=torch.float) / T

        total_days = time_steps * 365
        total_minutes = time_steps * 1440

        day_of_week = (total_days.long() % 7).clamp(0, 6)
        time_of_day = ((total_minutes // 5).long() % 288).clamp(0, 287)

        day_emb = self.day_embed(day_of_week)  # [T, embed_dim]
        time_emb = self.time_embed(time_of_day)  # [T, embed_dim]

        time_features = (day_emb + time_emb).T  # [embed_dim, T]
        time_features = time_features.unsqueeze(0).unsqueeze(2)  # [1, embed_dim, 1, T]

        time_features = time_features.expand(B, -1, N, -1)  # [B, embed_dim, N, T]

        return time_features + self.pos_encoder


class ST_LLM(nn.Module):
    def __init__(self, input_dim, num_nodes, input_len, output_len, llm_layer=6, U=2, device="cuda:0"):
        """
        完整时空大语言模型
        输入: [B,5,N,T] (5个特征)
        输出: [B,N,output_len] (TEC预测值)
        """
        super().__init__()
        self.input_dim = input_dim  # 5
        self.num_nodes = num_nodes  # 2911
        self.input_len = input_len  # 12
        self.output_len = output_len  # 12
        self.device = device

        self.temporal_embed = TemporalEmbedding()
        self.node_embed = nn.Parameter(torch.empty(num_nodes, 256))
        nn.init.xavier_uniform_(self.node_embed)

        self.start_conv = nn.Conv2d(in_channels=input_len * input_dim, out_channels=256, kernel_size=(1, 1))
        nn.init.kaiming_normal_(self.start_conv.weight)

        self.feature_fusion = nn.Sequential(
            nn.Conv2d(256 + 256 + 256, 768, kernel_size=(1, 1)), nn.LayerNorm([768, num_nodes, 1]), nn.GELU(), nn.Dropout(0.1)
        )

        # 修改GPT2模型加载方式，创建自定义配置支持更多节点
        config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "gpt2", "config.json"))
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "gpt2"))

        # 创建自定义GPT2配置，设置更大的上下文长度和位置嵌入
        custom_config = GPT2Config(
            n_positions=max(num_nodes, 1024),  # 至少支持所有节点数
            n_ctx=max(num_nodes, 1024),  # 至少支持所有节点数
            n_embd=768,  # 保持原有嵌入维度
            n_layer=llm_layer,  # 使用命令行参数指定的层数
            n_head=12,  # 保持原有注意力头数
        )

        print(f"创建自定义GPT2配置，支持 {num_nodes} 个节点, {llm_layer} 层")
        # 使用自定义配置初始化模型
        self.gpt = GPT2Model(custom_config)

        # 初始化完成后打印确认信息
        print(f"GPT模型配置: n_positions={self.gpt.config.n_positions}, n_ctx={self.gpt.config.n_ctx}")

        self.gpt.h = self.gpt.h[:llm_layer]
        self.gpt.config.n_layer = llm_layer

        for layer_idx, layer in enumerate(self.gpt.h):
            for name, param in layer.named_parameters():
                if layer_idx < llm_layer - U:
                    if "ln" in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                else:
                    param.requires_grad = True

        self.gpt.wpe = nn.Embedding(num_nodes, 768)
        self.gpt.wpe.weight = nn.Parameter(torch.empty(num_nodes, 768).normal_(mean=0.0, std=0.02))

        self.regression_layer = nn.Sequential(
            nn.Conv2d(768, 512, kernel_size=(1, 1)),
            nn.LayerNorm([512, num_nodes, 1]),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv2d(512, output_len, kernel_size=(1, 1)),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        """Xavier初始化所有可学习参数"""
        for name, param in self.named_parameters():
            if param.dim() > 1 and "gpt" not in name:
                if "weight" in name:
                    nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)

    def forward(self, history_data):
        if history_data.dim() != 4:
            raise ValueError(f"输入必须是4D张量[B,C,N,T]，实际得到{history_data.dim()}D")

        B, C, N, T = history_data.shape
        if C != self.input_dim:
            if history_data.size(3) == self.input_dim:
                history_data = history_data.permute(0, 3, 2, 1)  # [B,C,N,T]
                C = history_data.size(1)
            else:
                raise ValueError(f"输入特征维度不匹配！期望{self.input_dim}，实际{C}")

        input_flat = history_data.permute(0, 3, 2, 1)  # [B,T,N,C]
        input_flat = input_flat.reshape(B, -1, N, 1)  # [B,T*C,N,1] = [B,60,N,1]
        input_feat = self.start_conv(input_flat)  # [B,256,N,1]

        tem_emb = self.temporal_embed(history_data)  # [B,256,N,T]
        tem_emb = tem_emb.mean(dim=-1, keepdim=True)  # [B,256,N,1]

        node_emb = self.node_embed.unsqueeze(0)  # [1,N,256]
        node_emb = node_emb.expand(B, -1, -1)  # [B,N,256]
        node_emb = node_emb.permute(0, 2, 1).unsqueeze(-1)  # [B,256,N,1]

        fused_feat = torch.cat([input_feat, tem_emb, node_emb], dim=1)  # [B,768,N,1]
        fused_feat = self.feature_fusion(fused_feat)  # [B,768,N,1]

        gpt_input = fused_feat.squeeze(-1).permute(0, 2, 1)  # [B,N,768]
        position_ids = torch.arange(N, device=self.device).expand(B, -1)  # [B,N]

        gpt_output = self.gpt(
            inputs_embeds=gpt_input, position_ids=position_ids, attention_mask=torch.ones((B, N), device=self.device)
        ).last_hidden_state  # [B,N,768]

        gpt_output = gpt_output.permute(0, 2, 1).unsqueeze(-1)  # [B,768,N,1]
        prediction = self.regression_layer(gpt_output)  # [B,12,N,1]

        return prediction.squeeze(-1).permute(0, 2, 1)  # [B,N,12]

    def get_optimizer_groups(self, weight_decay=0.01):
        """优化器参数分组（带权重衰减）"""
        decay = set()
        no_decay = set()
        whitelist = (nn.Linear, nn.Conv1d, nn.Conv2d)
        blacklist = (nn.LayerNorm, nn.Embedding)

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn

                if not p.requires_grad:
                    continue

                if "bias" in pn:
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist):
                    no_decay.add(fpn)
                elif "embed" in pn:
                    no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"参数分组冲突: {inter_params}"
        assert len(param_dict.keys() - union_params) == 0, f"未分组参数: {param_dict.keys() - union_params}"

        return [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

    def param_num(self):
        """返回可训练参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
