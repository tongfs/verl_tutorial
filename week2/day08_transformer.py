"""
Day 8 手写 Transformer Decoder

对应 day08_attention_transformer.md 学习内容，实现：
- Scaled Dot-Product Attention（含因果掩码）
- Multi-Head Attention
- FFN（前馈网络）
- LayerNorm + Residual
- Decoder Block 与完整 Decoder 堆叠
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Scaled Dot-Product Attention
# ---------------------------------------------------------------------------


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Args:
        Q, K, V: (batch, seq_len, d_k) 或 (batch, num_heads, seq_len, d_k)
        mask: (batch, seq_len, seq_len) 或 (batch, 1, seq_len, seq_len)，
              0 表示要遮蔽的位置（会被置为 -inf）

    Returns:
        output: 与 V 最后一维相同
        attn_weights: 注意力权重
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k**0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)
    return output, attn_weights


def causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """因果掩码：位置 i 只能看 0..i，形状 (1, seq_len, seq_len)。"""
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0)  # (1, seq_len, seq_len)


# ---------------------------------------------------------------------------
# 2. Multi-Head Attention
# ---------------------------------------------------------------------------


class MultiHeadAttention(nn.Module):
    """
    多头自注意力：多组 Q/K/V 并行计算，Concat 后线性变换。
    d_k = d_v = d_model / num_heads。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        use_causal: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: 可选，额外掩码（如 padding mask），1=保留 0=遮蔽
            use_causal: 是否使用因果掩码（Decoder 用 True）
        """
        B, L, _ = x.shape
        Q = self.W_q(x)  # (B, L, d_model)
        K = self.W_k(x)
        V = self.W_v(x)

        # 拆成多头: (B, L, d_model) -> (B, L, h, d_k) -> (B, h, L, d_k)
        Q = Q.view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(B, L, self.num_heads, self.d_k).transpose(1, 2)

        if use_causal:
            causal = causal_mask(L, x.device)  # (1, L, L), 1=保留 0=遮蔽
            if mask is None:
                mask = causal
            else:
                if mask.dim() == 2:
                    # (B, L) -> (B, L, 1)*(B, 1, L) = (B, L, L)，再与 causal 取交
                    mask = mask.unsqueeze(2) * mask.unsqueeze(1) * causal
                else:
                    mask = mask * causal

        if mask is not None and mask.dim() == 3:
            mask = mask.unsqueeze(1)  # (B, 1, L, L)

        out, _ = scaled_dot_product_attention(Q, K, V, mask)
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.W_o(self.dropout(out))


# ---------------------------------------------------------------------------
# 3. FFN 前馈网络
# ---------------------------------------------------------------------------


class FFN(nn.Module):
    """
    前馈网络：Linear -> GELU -> Linear，中间维度通常为 4 * d_model。
    """

    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


# ---------------------------------------------------------------------------
# 4. Decoder Block（MHA + Residual + LN + FFN + Residual + LN）
# ---------------------------------------------------------------------------


class DecoderBlock(nn.Module):
    """
    一个 Transformer Decoder Block：
    x -> MHA + residual -> LayerNorm -> FFN + residual -> LayerNorm -> out
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FFN(d_model, d_ff, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        use_causal: bool = True,
    ) -> torch.Tensor:
        # 子层 1: Self-Attention + Residual + LayerNorm
        attn_out = self.self_attn(x, mask=mask, use_causal=use_causal)
        x = self.ln1(x + self.dropout(attn_out))

        # 子层 2: FFN + Residual + LayerNorm
        x = self.ln2(x + self.dropout(self.ffn(x)))
        return x


# ---------------------------------------------------------------------------
# 5. 位置编码（可学习或 sinusoidal）
# ---------------------------------------------------------------------------


class PositionalEncoding(nn.Module):
    """Sinusoidal 位置编码，或可学习 embedding（由 learnable 控制）。"""

    def __init__(self, d_model: int, max_len: int = 5120, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# 6. 完整 Transformer Decoder（GPT 风格）
# ---------------------------------------------------------------------------


class TransformerDecoder(nn.Module):
    """
    纯 Decoder Transformer：embedding + 位置编码 + N 个 DecoderBlock。
    不包含 LM head，可单独接 vocab 投影做语言模型。
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: Optional[int] = None,
        max_len: int = 5120,
        dropout: float = 0.1,
        padding_idx: Optional[int] = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList(
            [DecoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        use_causal: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len)  token ids
            mask: 可选 padding mask，(batch, seq_len) 或 (batch, L, L)，1=有效 0=遮蔽
            use_causal: 是否使用因果注意力
        Returns:
            (batch, seq_len, d_model)
        """
        x = self.embed(input_ids) * (self.d_model**0.5)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, mask=mask, use_causal=use_causal)
        return self.ln_f(x)


# ---------------------------------------------------------------------------
# 7. 带 LM Head 的语言模型（便于测试）
# ---------------------------------------------------------------------------


class TransformerLM(nn.Module):
    """
    Transformer Decoder + LM Head：输入 token ids，输出每个位置的 logits。
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: Optional[int] = None,
        max_len: int = 5120,
        dropout: float = 0.1,
        padding_idx: Optional[int] = 0,
    ):
        super().__init__()
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            max_len=max_len,
            dropout=dropout,
            padding_idx=padding_idx,
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # 可选：与 embed 共享权重（很多 GPT 实现会共享）
        self.lm_head.weight = self.decoder.embed.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        use_causal: bool = True,
    ) -> torch.Tensor:
        """
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        hidden = self.decoder(input_ids, mask=mask, use_causal=use_causal)
        return self.lm_head(hidden)


# ---------------------------------------------------------------------------
# 简单测试
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    batch, seq_len, vocab_size = 2, 8, 1000
    d_model, num_heads, num_layers = 256, 8, 4
    d_k = d_model // num_heads

    # 1) Scaled Dot-Product Attention
    Q = K = V = torch.randn(batch, seq_len, d_k)
    out, weights = scaled_dot_product_attention(Q, K, V)
    print("Scaled Dot-Product Attention:", out.shape, weights.shape)

    # 2) Causal mask
    cm = causal_mask(seq_len, Q.device)
    out_causal, _ = scaled_dot_product_attention(Q, K, V, mask=cm)
    print("With causal mask:", out_causal.shape)

    # 3) TransformerLM
    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
    )
    ids = torch.randint(0, vocab_size, (batch, seq_len))
    logits = model(ids)
    print("TransformerLM logits:", logits.shape)  # (2, 8, 1000)

    n_params = sum(p.numel() for p in model.parameters())
    print("Total parameters:", f"{n_params:,}")
