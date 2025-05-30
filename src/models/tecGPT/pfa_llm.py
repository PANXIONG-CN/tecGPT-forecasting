# src/models/pfa_llm.py
import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config, modeling_utils
import os
import copy


# 临时修复GPT-2模型的最大位置问题
class PatchedGPT2Model(GPT2Model):
    def forward(self, *args, **kwargs):
        """修改版GPT-2前向传播，支持更长序列"""

        # 检查是否有 inputs_embeds
        if "inputs_embeds" in kwargs:
            inputs_embeds = kwargs["inputs_embeds"]
            # 获取实际序列长度
            seq_length = inputs_embeds.size(1)

            # 检查序列长度是否超过模型当前的n_positions
            if seq_length > self.config.n_positions:
                # 临时修改配置以匹配我们的序列长度
                old_n_positions = self.config.n_positions
                old_max_position_embeddings = self.config.max_position_embeddings

                self.config.n_positions = seq_length
                self.config.max_position_embeddings = seq_length

                # 调用父类的forward方法
                try:
                    result = super().forward(*args, **kwargs)
                    return result
                finally:
                    # 还原原始配置
                    self.config.n_positions = old_n_positions
                    self.config.max_position_embeddings = old_max_position_embeddings

        # 如果序列长度没有超过n_positions或者没有inputs_embeds，直接调用父类的forward
        return super().forward(*args, **kwargs)


class PFA_GPT2(nn.Module):
    def __init__(
        self,
        local_gpt2_model_path: str,  # 明确参数名
        gpt_layers_to_use: int = 6,  # 要实际使用的层数
        U_unfrozen_mha: int = 2,
        num_spatial_nodes: int = 2911,
        d_llm_embedding: int = 768,
        enable_gradient_checkpointing: bool = False,
    ):
        super().__init__()

        print(f"PFA_GPT2: Initializing...")
        print(f"  Attempting to load GPT-2 from local path: {local_gpt2_model_path}")

        if not os.path.isdir(local_gpt2_model_path):
            error_msg = f"CRITICAL ERROR: Provided local_gpt2_model_path '{local_gpt2_model_path}' is not a valid directory."
            print(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            # 1. 从本地路径加载原始配置，不立即修改层数
            #    这样可以确保 config 与预训练权重文件的结构预期一致
            original_config = GPT2Config.from_pretrained(local_gpt2_model_path)

            # 确保 d_llm_embedding 与预训练模型的 n_embd 一致
            if original_config.n_embd != d_llm_embedding:
                print(
                    f"Warning: Requested d_llm_embedding ({d_llm_embedding}) differs from "
                    f"pretrained model's n_embd ({original_config.n_embd}). "
                    f"Using pretrained model's n_embd: {original_config.n_embd} for loading."
                )
                # d_llm_embedding = original_config.n_embd # 强制使用预训练的，或者抛错

            # 2. 从本地路径加载完整的预训练模型权重，同时通过修改GPT-2类支持更长序列
            gpt2_class = PatchedGPT2Model  # 使用我们的打补丁版本

            # 创建配置副本并修改
            config = copy.deepcopy(original_config)
            config.n_positions = max(config.n_positions, num_spatial_nodes)
            config.max_position_embeddings = max(config.max_position_embeddings, num_spatial_nodes)

            # 使用修改后的配置初始化，忽略尺寸不匹配的参数
            self.gpt2 = gpt2_class.from_pretrained(local_gpt2_model_path, config=config, ignore_mismatched_sizes=True)

            print(f"PFA_GPT2: Successfully loaded PRETRAINED GPT-2 weights and original config from {local_gpt2_model_path}")
            print(f"  Original model had {original_config.n_layer} layers.")

            # 3. 截断/选择要使用的层 (如果 gpt_layers_to_use 与原始不同)
            if len(self.gpt2.h) != gpt_layers_to_use:
                if gpt_layers_to_use > len(self.gpt2.h):
                    print(
                        f"Warning: Requested gpt_layers_to_use ({gpt_layers_to_use}) is more than "
                        f"available layers in pretrained model ({len(self.gpt2.h)}). Using all available layers."
                    )
                    gpt_layers_to_use = len(self.gpt2.h)

                self.gpt2.h = self.gpt2.h[:gpt_layers_to_use]
                # 更新模型配置中的层数以反映实际使用的层数
                self.gpt2.config.n_layer = gpt_layers_to_use
                print(f"PFA_GPT2: Using first {gpt_layers_to_use} layers from the pretrained model.")
            else:
                print(f"PFA_GPT2: Using all {gpt_layers_to_use} layers from the pretrained model (no truncation needed).")

        except Exception as e:
            critical_error_msg = f"CRITICAL ERROR: Failed to load GPT-2 from local path '{local_gpt2_model_path}'. Error: {e}"
            print(critical_error_msg)
            print("  Ensure the path is correct and contains 'config.json' and 'pytorch_model.bin' (or 'model.safetensors').")
            print("  If the model files are for a different architecture (e.g. gpt2-medium), specify that in llm_model_name.")
            raise  # 出现此错误通常意味着模型无法正确加载，应中断执行

        # 4. PFA 冻结逻辑
        actual_layers_in_use = len(self.gpt2.h)
        if U_unfrozen_mha > actual_layers_in_use:
            print(f"Warning: U_unfrozen_mha ({U_unfrozen_mha}) is greater than actual layers in use ({actual_layers_in_use}). Clamping U.")
            U_unfrozen_mha = actual_layers_in_use

        print(f"PFA_GPT2: Applying PFA strategy. Effective GPT layers={actual_layers_in_use}, Keeping last {U_unfrozen_mha} MHA/LN trainable.")
        for layer_idx, layer in enumerate(self.gpt2.h):
            is_unfrozen_block_for_mha_ln = layer_idx >= actual_layers_in_use - U_unfrozen_mha
            for name, param in layer.named_parameters():
                if "mlp" in name:  # MLP (FFN) 总是冻结，除非 U 覆盖了整个模型且有特殊需要
                    param.requires_grad = False
                elif "attn" in name or "ln_1" in name or "ln_2" in name:  # ln_1是MHA前的，ln_2是MLP前的
                    if is_unfrozen_block_for_mha_ln:  # 后 U 层中的 Attention 和 LayerNorm 可训练
                        param.requires_grad = True
                    else:  # 前 F 层中的 Attention 冻结，LayerNorm 可训练
                        param.requires_grad = "ln" in name
                else:  # 其他可能的参数默认冻结
                    param.requires_grad = False

        # 5. 替换位置嵌入层 (wpe) 以匹配 num_spatial_nodes
        # GPT-2 的位置嵌入层名为 'wpe'
        self.gpt2.wpe = nn.Embedding(num_spatial_nodes, d_llm_embedding)
        nn.init.normal_(self.gpt2.wpe.weight, std=0.02)  # 标准初始化
        print(f"PFA_GPT2: Reinitialized GPT2 wpe for {num_spatial_nodes} positions (nodes).")

        # 禁用位置嵌入，让模型完全依赖外部的SpatialEmbedding
        print("PFA_GPT2: Nullifying GPT2 wpe weights and making it non-trainable to rely on external SpatialEmbedding.")
        self.gpt2.wpe.weight.data.zero_()  # 将权重设为0
        self.gpt2.wpe.weight.requires_grad = False  # 设为不可训练

        # 6. 梯度检查点
        if enable_gradient_checkpointing:
            if hasattr(self.gpt2, "gradient_checkpointing_enable") and callable(self.gpt2.gradient_checkpointing_enable):
                try:
                    self.gpt2.gradient_checkpointing_enable()
                    print("PFA_GPT2: Gradient checkpointing enabled.")
                except Exception as e_gc:
                    print(f"PFA_GPT2: Failed to enable gradient checkpointing: {e_gc}")
            else:
                print("PFA_GPT2: Gradient checkpointing requested but not supported or method not found.")
        else:
            print("PFA_GPT2: Gradient checkpointing not enabled.")

    def forward(self, inputs_embeds, attention_mask=None):
        batch_size, seq_length, _ = inputs_embeds.size()
        device = inputs_embeds.device

        # 创建 position_ids，确保它不超过 self.gpt2.wpe 的大小
        max_pos_embed = self.gpt2.wpe.weight.size(0)
        if seq_length > max_pos_embed:
            raise ValueError(
                f"Input sequence length ({seq_length}) exceeds PFA_GPT2 wpe max positions ({max_pos_embed}). "
                f"Ensure num_spatial_nodes in PFA_GPT2 init matches the sequence length (N_nodes) from input."
            )

        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0)  # Shape [1, N_nodes], GPT2Model 会自动扩展

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_length, device=device, dtype=torch.long)

        return self.gpt2(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
        ).last_hidden_state
