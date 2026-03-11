from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    给定线性层 (Linear layer) 的权重，计算批次输入的线性变换。

    参数 (Args):
        d_in (int): 输入维度的尺寸。
        d_out (int): 输出维度的尺寸。
        weights (Float[Tensor, "d_out d_in"]): 要使用的线性层权重矩阵。
        in_features (Float[Tensor, "... d_in"]): 需要应用该变换的输入张量。

    返回 (Returns):
        Float[Tensor, "... d_out"]: 线性模块转换后的输出张量。
    """
    # 待实现：在此处执行矩阵乘法 (如 torch.matmul 或类似操作)
    raise NotImplementedError


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    给定嵌入层 (Embedding layer) 的权重，获取一批 Token ID 的嵌入向量。

    参数 (Args):
        vocab_size (int): 词汇表中的嵌入向量总数。
        d_model (int): 嵌入维度的尺寸。
        weights (Float[Tensor, "vocab_size d_model"]): 用于提取的嵌入向量权重矩阵。
        token_ids (Int[Tensor, "..."]): 需要从嵌入层中提取特征的一组 Token ID。

    返回 (Returns):
        Float[Tensor, "... d_model"]: 嵌入层返回的批次嵌入向量。
    """
    # 待实现：根据 token_ids 索引提取 weights 中的对应行
    raise NotImplementedError


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """给定 SwiGLU 网络的权重，返回基于这些权重计算的输出。
    SwiGLU 结合了 SiLU 激活函数和门控线性单元 (Gated Linear Unit)。

    参数 (Args):
        d_model (int): 前馈网络输入和输出的维度。
        d_ff (int): SwiGLU 内部向上投影 (up-project) 的隐藏层维度。
        w1_weight (Float[Tensor, "d_ff d_model"]): W1 的存储权重。
        w2_weight (Float[Tensor, "d_model d_ff"]): W2 的存储权重。
        w3_weight (Float[Tensor, "d_ff d_model"]): W3 的存储权重。
        in_features (Float[Tensor, "... d_model"]): 输入到前馈层的嵌入张量。

    返回 (Returns):
        Float[Tensor, "... d_model"]: 与输入嵌入形状相同的输出嵌入张量。
    """
    # 示例:
    # 如果你的 state dict 键匹配，你可以使用 `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # 你也可以手动分配权重
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight
    raise NotImplementedError


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    给定查询 (Query, Q)、键 (Key, K) 和值 (Value, V) 张量，
    返回缩放点积注意力 (Scaled Dot-Product Attention, SDPA) 的计算结果。

    参数 (Args):
        Q (Float[Tensor, " ... queries d_k"]): 查询张量。
        K (Float[Tensor, " ... keys d_k"]): 键张量。
        V (Float[Tensor, " ... values d_v"]): 值张量。
        mask (Bool[Tensor, " ... queries keys"] | None): 可选的掩码张量 (True 表示被遮蔽或保留，取决于实现约定，通常用于因果掩码)。
    返回 (Returns):
        Float[Tensor, " ... queries d_v"]: SDPA 的输出结果。
    """
    # 待实现：计算 Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
    raise NotImplementedError


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    给定非批处理多头自注意力 (Multi-Head Attention, MHA) 的朴素实现中的 Q、K、V 投影权重，
    返回优化后的批处理实现的输出。
    此实现应当在单次矩阵乘法中处理所有注意力头 (heads) 的键、查询和值的投影。
    此函数不应使用旋转位置编码 (RoPE)。
    参考文献：Vaswani 等人，2017年《Attention Is All You Need》第 3.2.2 节。

    参数 (Args):
        d_model (int): 前馈网络输入和输出的维度。
        num_heads (int): 多头注意力中使用的头的数量。
        q_proj_weight (Float[Tensor, "d_k d_in"]): Q 投影的权重。
        k_proj_weight (Float[Tensor, "d_k d_in"]): K 投影的权重。
        v_proj_weight (Float[Tensor, "d_k d_in"]): V 投影的权重 (注: 原文 d_k 可能是 typo，逻辑上通常为 d_v)。
        o_proj_weight (Float[Tensor, "d_model d_v"]): 输出投影 (Output projection) 的权重。
        in_features (Float[Tensor, "... sequence_length d_in"]): 运行计算的输入张量。

    返回 (Returns):
        Float[Tensor, " ... sequence_length d_out"]: 使用给定投影权重和输入特征，运行优化后的批处理多头自注意力的输出张量。
    """
    # 待实现：线性投影 -> 拆分注意力头 -> 计算 SDPA -> 拼接特征 -> 输出线性投影
    raise NotImplementedError


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    该版本的多头自注意力机制 (MHA) 包含了旋转位置编码 (Rotary Position Embedding, RoPE)。
    在此情况下，RoPE 的嵌入维度必须等于每个注意力头的嵌入维度 (d_model // num_heads)。

    参数 (Args):
        d_model (int): 前馈网络输入和输出的维度。
        num_heads (int): 多头注意力中使用的头的数量。
        max_seq_len (int): 预缓存的最大序列长度（如果实现采用预缓存策略）。
        theta (float): RoPE 频率计算参数 (通常为 10000.0)。
        q_proj_weight, k_proj_weight, v_proj_weight: Q、K、V 投影权重。
        o_proj_weight: 输出投影权重。
        in_features: 运行计算的输入特征。
        token_positions (Int[Tensor, " ... sequence_length"] | None): 可选张量，指定每个 Token 的位置。

    返回 (Returns):
        Float[Tensor, " ... sequence_length d_out"]: 应用了 RoPE 的批处理 MHA 输出张量。
    """
    # 待实现：在计算 SDPA 之前，对拆分后的 Q 和 K 张量应用 RoPE
    raise NotImplementedError


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    对给定的输入张量应用旋转位置编码 (RoPE)。

    参数 (Args):
        d_k (int): 查询 (Query) 或键 (Key) 张量的特征维度大小 (即单个头的维度)。
        theta (float): RoPE 基数参数。
        max_seq_len (int): 最大序列长度 (用于生成或截断预计算的复数频率张量)。
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): 需要应用 RoPE 的输入张量。
        token_positions (Int[Tensor, "... sequence_length"]): 形状为 (batch_size, sequence_length) 的 Token 位置张量。
    返回 (Returns):
        Float[Tensor, " ... sequence_length d_k"]: 应用 RoPE 旋转后的张量。
    """
    # 待实现：根据 token_positions 计算旋转角度，并作用于输入张量的实部和虚部对
    raise NotImplementedError


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    给定前置归一化 (pre-norm) Transformer 块的权重和输入特征，
    返回该 Transformer 块对输入特征的处理输出。

    此函数必须使用 RoPE。根据具体实现，您可能只需将相关参数传递给您的 TransformerBlock 构造函数，
    或者您需要初始化自己的 RoPE 类并进行传递。

    参数 (Args):
        d_model (int): Transformer 块的输入维度。
        num_heads (int): 多头注意力中头的数量。`d_model` 必须能被 `num_heads` 整除。
        d_ff (int): 前馈网络隐藏层的维度。
        max_seq_len (int): 最大序列长度。
        theta (float): RoPE 参数。
        weights (dict[str, Tensor]):
            参考实现的权重字典 (State dict)。字典的键包含：
            - `attn.q_proj.weight`
                所有 `num_heads` 个注意力头的查询投影权重，形状为 (d_model, d_model)。
                行按 (num_heads, d_k) 的矩阵顺序排列，因此 `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`。
            - `attn.k_proj.weight`
                所有 `num_heads` 个注意力头的键投影权重，形状排列逻辑同上。
            - `attn.v_proj.weight`
                所有 `num_heads` 个注意力头的值投影权重，形状为 (d_model, d_model)，由 (num_heads, d_v) 堆叠而成。
            - `attn.output_proj.weight`
                多头自注意力输出的投影权重，形状为 (d_model, d_model)。
            - `ln1.weight`
                Transformer 块中应用于注意力机制之前的第一个 RMSNorm 的仿射变换权重，形状为 (d_model,)。
            - `ffn.w1.weight`
                前馈网络 (FFN) 第一层线性变换的权重，形状为 (d_model, d_ff)。
            - `ffn.w2.weight`
                FFN 第二层线性变换的权重，形状为 (d_ff, d_model)。
            - `ffn.w3.weight`
                FFN 第三层线性变换的权重，形状为 (d_model, d_ff)。
            - `ln2.weight`
                应用于 FFN 之前的第二个 RMSNorm 的权重，形状为 (d_model,)。
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            运行该计算的输入张量。

    返回 (Returns):
        Float[Tensor, "batch sequence_length d_model"]: 包含 Transformer 块前向传播结果的输出张量。
    """
    # 待实现：RMSNorm -> Attention(with RoPE) -> 残差连接 -> RMSNorm -> FFN -> 残差连接
    raise NotImplementedError


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """给定 Transformer 语言模型 (Language Model) 的权重和输入词索引，
    返回在输入索引上执行前向传播的输出。

    此函数必须使用 RoPE。

    参数 (Args):
        vocab_size (int): 预测输出词表中的唯一项数量。
        context_length (int): 单次处理的最大 Token 数量。
        d_model (int): 模型嵌入和子层输出的维度。
        num_layers (int): 使用的 Transformer 层数。
        num_heads (int): 多头注意力头的数量。`d_model` 必须能被 `num_heads` 整除。
        d_ff (int): 前馈网络隐藏层的维度。
        rope_theta (float): RoPE 频率参数 $\Theta$。
        weights (dict[str, Tensor]):
            模型的权重字典 (State dict)。`{num_layers}` 是介于 `0` 到 `num_layers - 1` 之间的整数 (层索引)。
            字典的键包括：
            - `token_embeddings.weight`: 词嵌入矩阵，形状为 (vocab_size, d_model)。
            - `layers.{num_layers}.attn.q_proj.weight`: 第 N 层注意力的 Q 投影矩阵。
            - `layers.{num_layers}.attn.k_proj.weight`: 第 N 层注意力的 K 投影矩阵。
            - `layers.{num_layers}.attn.v_proj.weight`: 第 N 层注意力的 V 投影矩阵。
            - `layers.{num_layers}.attn.output_proj.weight`: 第 N 层注意力的输出投影矩阵。
            - `layers.{num_layers}.ln1.weight`: 第 N 层的第一个 RMSNorm 权重。
            - `layers.{num_layers}.ffn.w1.weight`: 第 N 层 FFN 的 w1 权重。
            - `layers.{num_layers}.ffn.w2.weight`: 第 N 层 FFN 的 w2 权重。
            - `layers.{num_layers}.ffn.w3.weight`: 第 N 层 FFN 的 w3 权重。
            - `layers.{num_layers}.ln2.weight`: 第 N 层的第二个 RMSNorm 权重。
            - `ln_final.weight`: 最终输出 RMSNorm 的权重。
            - `lm_head.weight`: 语言模型输出头的权重，映射到词汇表。
        in_indices (Int[Tensor, "batch_size sequence_length"]): 作为输入的整数 Token 索引张量。
            最大序列长度不得超过 `context_length`。

    返回 (Returns):
        Float[Tensor, "batch_size sequence_length vocab_size"]: 包含每个 Token 预测的未归一化
        下一个词分布 (Logits) 的张量。
    """
    # 待实现：词嵌入 -> N 层 Transformer Blocks -> 最终 RMSNorm -> LM Head 线性层映射
    raise NotImplementedError


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """给定均方根归一化 (RMSNorm) 仿射变换的权重，
    返回对输入特征运行 RMSNorm 后的输出。

    参数 (Args):
        d_model (int): RMSNorm 的输入维度。
        eps: (float): 添加到分母以保证数值稳定性的微小值。
        weights (Float[Tensor, "d_model"]): RMSNorm 可学习的缩放权重 (gamma)。
        in_features (Float[Tensor, "... d_model"]): 需要应用 RMSNorm 的输入特征，允许包含任意数量的前置维度。

    返回 (Returns):
        Float[Tensor,"... d_model"]: 与 `in_features` 形状相同，且经过 RMSNorm 归一化的输出张量。
    """
    # 待实现：计算特征的均方根并进行缩放: (x / RMS(x)) * weight
    raise NotImplementedError


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """给定输入张量，返回对每个元素应用 SiLU (Sigmoid Linear Unit) 激活函数后的输出。

    参数 (Args):
        in_features(Float[Tensor, "..."]): 需要应用 SiLU 的输入特征。形状任意。

    返回 (Returns):
        Float[Tensor,"..."]: 形状与 `in_features` 相同，每个元素经过 SiLU 计算的结果。
    """
    # 待实现：计算 x * sigmoid(x)
    raise NotImplementedError


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    给定数据集（一维整数 numpy 数组）、期望的批次大小 (batch size) 和上下文长度 (context length)，
    从数据集中随机采样语言建模的输入序列及其对应的标签。

    参数 (Args):
        dataset (np.array): 包含数据集中整数 Token ID 的 1D numpy 数组。
        batch_size (int): 期望采样的批次大小。
        context_length (int): 每个采样样本的期望上下文长度。
        device (str): PyTorch 设备标识符（例如，'cpu' 或 'cuda:0'），指示
            将采样的输入序列和标签放置在哪个设备上。

    返回 (Returns):
        由形状为 (batch_size, context_length) 的 torch.LongTensor 组成的元组。
        第一个张量是采样的输入序列 (X)，第二个张量是对应的语言建模目标标签 (Y，通常是将输入平移1位)。
    """
    # 待实现：随机生成起始索引，截取 X = dataset[i:i+context_length], Y = dataset[i+1:i+1+context_length]
    raise NotImplementedError


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    给定输入张量，返回在指定维度 `dim` 上应用 Softmax 归一化后的结果。

    参数 (Args):
        in_features (Float[Tensor, "..."]): 需要计算 Softmax 的输入特征。形状任意。
        dim (int): 要应用 Softmax 的维度。

    返回 (Returns):
        Float[Tensor, "..."]: 与 `in_features` 形状相同，且在指定维度上完成 Softmax 归一化的输出张量。
    """
    # 待实现：exp(x) / sum(exp(x))，注意需要减去最大值以保持数值稳定
    raise NotImplementedError


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """给定输入张量和目标张量，计算所有样本的平均交叉熵损失 (Cross-Entropy Loss)。

    参数 (Args):
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] 表示
            第 i 个样本在第 j 个类别上的未归一化逻辑值 (logit)。
        targets (Int[Tensor, "batch_size"]): 形状为 (batch_size,) 的张量，包含正确类别的索引。
            每个值必须介于 0 和 `num_classes - 1` 之间。

    返回 (Returns):
        Float[Tensor, ""]: 跨样本求均值的标量交叉熵损失。
    """
    # 待实现：-mean(log(softmax(inputs)[targets]))
    raise NotImplementedError


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """给定一组参数，裁剪其组合梯度 (Gradient Clipping)，确保其全局 L2 范数最大不超过 max_l2_norm。

    参数 (Args):
        parameters (Iterable[torch.nn.Parameter]): 可训练参数的集合。
        max_l2_norm (float): 包含最大 L2 范数限制的正数。

    必须就地 (in-place) 修改参数的梯度 (parameter.grad)。
    """
    # 待实现：计算全局梯度 L2 范数，如果超过 max_l2_norm 则对所有 parameter.grad 进行等比例缩放
    raise NotImplementedError


def get_adamw_cls() -> Any:
    """
    返回实现了 AdamW 算法的 torch.optim.Optimizer 类。
    """
    # 待实现：return torch.optim.AdamW
    raise NotImplementedError


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    给定基于余弦学习率衰减策略（带线性预热）的参数和当前迭代次数，
    返回该调度计划下当前迭代的学习率。

    参数 (Args):
        it (int): 当前请求获取学习率的迭代次数。
        max_learning_rate (float): alpha_max，带预热余弦调度的最大学习率 (峰值)。
        min_learning_rate (float): alpha_min，带预热余弦调度的最小 / 最终学习率。
        warmup_iters (int): T_w，用于线性预热 (Linear Warmup) 的迭代次数。
        cosine_cycle_iters (int): T_c，余弦退火退化的迭代次数周期。

    返回 (Returns):
        当前迭代下计算得出的标量学习率。
    """
    # 待实现：实现预热期(线性增长) 和 衰减期(余弦退火至 min_learning_rate) 的逻辑
    raise NotImplementedError


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    给定模型、优化器和当前的迭代次数，将其序列化并保存到磁盘。

    参数 (Args):
        model (torch.nn.Module): 序列化此模型的状态字典。
        optimizer (torch.optim.Optimizer): 序列化此优化器的状态字典。
        iteration (int): 序列化此数值，代表当前已完成的训练迭代步数。
        out (str | os.PathLike | BinaryIO | IO[bytes]): 保存序列化模型、优化器和迭代次数的目标路径或类文件对象。
    """
    # 待实现：构建保存字典并使用 torch.save 写入 out
    raise NotImplementedError


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    给定序列化后的检查点 (路径或类文件对象)，将序列化状态恢复并加载到模型和优化器中。
    返回先前在检查点中序列化的迭代次数。

    参数 (Args):
        src (str | os.PathLike | BinaryIO | IO[bytes]): 序列化检查点的来源路径或类文件对象。
        model (torch.nn.Module): 恢复该模型的状态字典。
        optimizer (torch.optim.Optimizer): 恢复该优化器的状态字典。
    返回 (Returns):
        int: 先前保存的迭代次数。
    """
    # 待实现：使用 torch.load 读取 src，分别加载模型/优化器状态，并提取迭代次数返回
    raise NotImplementedError


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """根据提供的词汇表、合并规则 (merges) 和特殊 Token 列表，
    返回一个使用这些配置的 BPE 分词器 (BPE Tokenizer)。

    参数 (Args):
        vocab (dict[int, bytes]): 分词器词汇表，为从 int（词表中的 Token ID）到 bytes（Token 字节串）的映射。
        merges (list[tuple[bytes, bytes]]): BPE 合并规则。列表每一项是一个由字节构成的元组 (<token1>, <token2>)，
            代表 <token1> 与 <token2> 被合并的规则。合并规则按创建的先后顺序排列。
        special_tokens (list[str] | None): 分词器的特殊字符串 Token 列表。
            这些字符串将始终被保留为单个独立的 Token，绝不会被拆分成多个 Token。

    返回 (Returns):
        使用指定词汇表、合并规则和特殊 Token 配置的 BPE 分词器实例。
    """
    raise NotImplementedError


import os
import regex as re
from collections import Counter, defaultdict  # 1. 导入 defaultdict

def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """给定输入语料库的路径，运行并训练一个 BPE 分词器，输出其生成的词汇表和合并规则。

    参数 (Args):
        input_path (str | os.PathLike): BPE 分词器训练数据的来源路径。
        vocab_size (int): 词汇表中的项目总数（包含特殊 Token 在内）。
        special_tokens (list[str]): 要添加到词汇表的特殊字符串 Token 列表。
            这些字符串将始终被保留为单一 Token，绝不进行拆分。
            如果这些特殊 Token 出现在 `input_path` 数据中，它们会和其他字符串受到同等对待。

    返回 (Returns):
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab (词汇表):
                训练后生成的词汇表映射，从 int (Token ID) 映射到 bytes (Token 字节串)。
            merges (合并规则):
                BPE 合并规则列表。每一项是由字节组成的元组 (<token1>, <token2>)，
                表示合并记录，排列顺序即为合并执行的顺序。
    """
    # 可合并次数
    sz = vocab_size - len(special_tokens) - 256
    
    # # 1. 切分规则
    # base_pat = r"""'(?i:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    # # 2. 动态拼接特殊 Token
    # if special_tokens:
    #     # 使用 re.escape 转义每个 token，例如把 "[PAD]" 变成 "\[PAD\]"
    #     escaped_tokens = [re.escape(tok) for tok in special_tokens]
    #     # 用 '|' 连接起来，外面包上 (?:...) 表示这是一个非捕获组
    #     special_pat = "(?:" + "|".join(escaped_tokens) + ")"
        
    #     # 把特殊规则放在最前面，加上 '|' 和基础规则拼接
    #     PAT = special_pat + "|" + base_pat
    # else:
    #     PAT = base_pat
    
    # # 3. 强烈建议：在文件循环外预先编译正则，能大幅提升读取大文件时的速度！
    # compiled_pat = re.compile(PAT)
    # cnt = Counter()
    
    base_pat = r"""'(?i:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    base_re = re.compile(base_pat)

    cnt = Counter()

    with open(input_path, "rb") as f:
        raw = f.read()

    text_content = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n").decode("utf-8")
    # text_content = raw.decode('utf-8')
    if special_tokens:
        escaped_tokens = sorted((re.escape(tok) for tok in special_tokens), key=len, reverse=True)
        special_re = re.compile("|".join(escaped_tokens))

        last = 0
        for m in special_re.finditer(text_content):
            # 先处理 special token 前面的普通文本
            if m.start() > last:
                chunk = text_content[last:m.start()]
                for x in base_re.finditer(chunk):
                    cnt[x.group().encode("utf-8")] += 1

            # special token 本身作为一个完整 pretoken
            cnt[m.group().encode("utf-8")] += 1
            last = m.end()

        # 处理最后一个 special token 后面的尾巴
        if last < len(text_content):
            chunk = text_content[last:]
            for x in base_re.finditer(chunk):
                cnt[x.group().encode("utf-8")] += 1
    else:
        for x in base_re.finditer(text_content):
            cnt[x.group().encode("utf-8")] += 1
    mp = {(i,): i for i in range(256)}
    # 3. 初始化词汇表 (vocab) 和特殊 Token 映射
    vocab = {i: bytes([i]) for i in range(256)}
    special_token_map = {}
    # 给特殊 Token 分配固定 ID，紧跟在 255 之后
    next_id = 256
    for st in special_tokens:
        st_bytes = st.encode('utf-8')
        special_token_map[st_bytes] = next_id
        vocab[next_id] = st_bytes
        next_id += 1
    
    # 4. 初始化待合并的 texts
    texts = []
    for c in cnt.keys(): # c 是 bytes
        if c in special_token_map:
            # 特殊 Token：直接转为单元素 ID 列表，它将自然免疫后续的相邻对合并
            texts.append([special_token_map[c]])
        else:
            # 普通词块：打散成基础字节 ID 列表
            texts.append(list(c))
    merges = []
    tc = Counter()
    for c, text in zip(cnt, texts):
        num = cnt[c]
        for x, y in zip(text, text[1:]):
            tc[(x, y)] += num
    while True:
        if not tc:
            break
        # 把 k (ID 对) 映射回 vocab 中的真实 bytes 进行字典序比较
        best_pair = max(tc, key=lambda k: (tc[k], vocab[k[0]], vocab[k[1]]))
        # best_pair = max(tc, key=lambda k: (tc[k], k))  (按照更新后的id进行排序  发生错误)
        if tc[best_pair] < 2 or sz == 0:
            break
        sz -= 1
        p0, p1 = best_pair
        vocab[next_id] = vocab[p0] + vocab[p1]
        merges.append((vocab[p0], vocab[p1]))
        mp[best_pair] = next_id
        tc[best_pair] = 0 # 置 0
        for c, i in zip(cnt, range(len(texts))):
            text = texts[i]
            new_text = []
            if p0 not in text:
                continue
            for j in range(len(text) - 1):
                if (text[j], text[j+1]) != best_pair:
                    tc[(text[j], text[j+1])] -= cnt[c]
            j = 0
            while j < len(text):
                if j + 1 < len(text) and (text[j], text[j+1]) == best_pair:
                    new_text.append(next_id)
                    j += 1
                else:
                    new_text.append(text[j])
                j += 1
            for j in range(len(new_text) - 1):
                tc[(new_text[j], new_text[j+1])] += cnt[c]
            texts[i] = new_text
        del tc[best_pair] # 删除，防止一堆为0的死键
        next_id += 1
    return vocab, merges
