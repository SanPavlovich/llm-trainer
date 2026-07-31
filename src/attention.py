import logging
from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.schemas import TransformerConfig

logger = logging.getLogger(__name__)


def build_causal_key_mask(
    attention_mask: Tensor | None,
    L: int,
    B: int,
    device: torch.device,
) -> Tensor:
    causal = torch.ones(L, L, dtype=torch.bool, device=device).tril().view(1, 1, L, L)

    if attention_mask is None:
        return causal.expand(B, 1, L, L)

    key_mask = attention_mask
    if key_mask.dim() > 2:
        key_mask = key_mask[..., 0]
    key_mask = (key_mask > 0).view(B, 1, 1, L)  # True = keep
    return causal & key_mask  # [B, 1, L, L]


def eager_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attention_mask: Tensor | None,
    dropout_p: float,
    scale: float,
    attn_dropout: nn.Module,
    alibi_bias: Tensor | None = None,
) -> Tensor:
    B, H, L, _ = q.shape
    att = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, L, L]

    if alibi_bias is not None:
        i = torch.arange(L, device=q.device).view(1, 1, L, 1)
        j = torch.arange(L, device=q.device).view(1, 1, 1, L)
        att = att + (alibi_bias.to(att.dtype) * (j - i))

    allow = build_causal_key_mask(attention_mask, L, B, q.device)
    att = att.masked_fill(~allow, torch.finfo(att.dtype).min)

    att = F.softmax(att, dim=-1)
    att = attn_dropout(att)
    return torch.matmul(att, v)


def sdpa_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attention_mask: Tensor | None,
    dropout_p: float,
    scale: float,
    attn_dropout: nn.Module | None = None,   # unused; SDPA applies dropout internally
    alibi_bias: Tensor | None = None,        # unused; sdpa is RoPE-only
) -> Tensor:
    B, H, L, _ = q.shape

    if attention_mask is None:
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=True, scale=scale
        )

    allow = build_causal_key_mask(attention_mask, L, B, q.device)  # [B, 1, L, L] bool
    # Additive float mask: 0 where allowed, -inf where masked.
    attn_bias = torch.zeros(B, 1, L, L, dtype=q.dtype, device=q.device)
    attn_bias = attn_bias.masked_fill(~allow, torch.finfo(q.dtype).min)

    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_bias, dropout_p=dropout_p, is_causal=False, scale=scale
    )


ATTENTION_IMPLEMENTATIONS = {
    "eager": eager_attention,
    "sdpa": sdpa_attention
}


class CausalSelfAttentionMLA(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        assert self.config.hidden_dim % self.config.n_head == 0
        assert self.config.n_head % self.config.n_kv_head == 0
        self.head_dim = self.config.hidden_dim // self.config.n_head
        self.scale = self.head_dim ** -0.5
        self.q_per_kv = self.config.n_head // self.config.n_kv_head
        self.attention_interface: Callable = ATTENTION_IMPLEMENTATIONS[config.attn_impl]
        # SDPA can't apply our ALiBi bias, so it is RoPE-only.
        assert not (self.config.attn_impl == "sdpa" and not self.config.use_rope), (
            "attn_impl='sdpa' is not supported with ALiBi (use_rope=False); "
            "use attn_impl='eager' for ALiBi."
        )

        self.q_proj = nn.Linear(self.config.hidden_dim, self.config.n_head * self.head_dim, bias=False)

        self.c_proj   = nn.Linear(self.config.hidden_dim, self.config.latent_dim, bias=False)
        self.k_dec    = nn.Linear(self.config.latent_dim, self.config.n_kv_head * self.head_dim, bias=False)
        self.v_dec    = nn.Linear(self.config.latent_dim, self.config.n_kv_head * self.head_dim, bias=False)

        self.out_proj = nn.Linear(self.config.n_head * self.head_dim, self.config.hidden_dim, bias=False)
        self.attn_dropout = nn.Dropout(self.config.dropout)
        mask = torch.tril(torch.ones(self.config.max_seq_len, self.config.max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, self.config.max_seq_len, self.config.max_seq_len), persistent=False)

        n = self.config.n_head
        if (n & (n - 1)) == 0:
            start = 2 ** (-8.0 / n); slopes = [start ** i for i in range(1, n + 1)]
        else:
            closest = 1 << (n.bit_length() - 1)
            start1 = 2 ** (-8.0 / closest); slopes1 = [start1 ** i for i in range(1, closest + 1)]
            start2 = 2 ** (-8.0 / (2 * closest)); slopes2 = [start2 ** i for i in range(1, 2 * closest + 1)]
            slopes = slopes1 + slopes2[: n - closest]
        self.register_buffer("alibi", torch.tensor(slopes, dtype=torch.float32).view(1, n, 1, 1), persistent=False)

        if self.config.use_rope:
            half = self.head_dim // 2
            inv_freq = 1.0 / (self.config.rope_theta ** (torch.arange(half, dtype=torch.float32) / half))
            self.register_buffer("rope_inv_freq", inv_freq, persistent=False)

    def forward(self, x: Tensor, attention_mask: Tensor = None) -> Tensor:
        B, L, _ = x.shape

        # Q
        q = self.q_proj(x).view(B, L, self.config.n_head, self.head_dim).permute(0, 2, 1, 3)

        c  = self.c_proj(x)
        kd = self.k_dec(c).view(B, L, self.config.n_kv_head, self.head_dim)
        vd = self.v_dec(c).view(B, L, self.config.n_kv_head, self.head_dim)
        k = kd.permute(0, 2, 1, 3)
        v = vd.permute(0, 2, 1, 3)

        if self.q_per_kv > 1:
            k = k.repeat_interleave(self.q_per_kv, dim=1)
            v = v.repeat_interleave(self.q_per_kv, dim=1)

        if self.config.use_rope:
            t = torch.arange(L, device=x.device, dtype=torch.float32)
            freqs = torch.outer(t, self.rope_inv_freq)
            cos = freqs.cos().to(q.dtype).view(1, 1, L, -1)
            sin = freqs.sin().to(q.dtype).view(1, 1, L, -1)
            h = self.head_dim // 2

            q1, q2 = q[..., :h], q[..., h:]
            k1, k2 = k[..., :h], k[..., h:]
            q = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
            k = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)

        # ALiBi bias only for the eager path without RoPE; None otherwise.
        alibi_bias = None if self.config.use_rope else self.alibi
        dropout_p = self.attn_dropout.p if self.training else 0.0

        y = self.attention_interface(
            q,
            k,
            v,
            attention_mask,
            dropout_p=dropout_p,
            scale=self.scale,
            attn_dropout=self.attn_dropout,
            alibi_bias=alibi_bias,
        )

        y = y.permute(0, 2, 1, 3).contiguous().view(B, L, self.config.n_head * self.head_dim)
        y = self.out_proj(y)
        return y
    

class CausalSelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        """Causal Self-Attention with support of
        Grouped-Query Attention and ALiBi for positional encoding
        """
        super().__init__()
        self.config = config
        assert self.config.hidden_dim % self.config.n_head == 0
        assert self.config.n_head % self.config.n_kv_head == 0
        self.head_dim = self.config.hidden_dim // self.config.n_head
        self.scale = self.head_dim**-0.5
        self.q_per_kv = self.config.n_head // self.config.n_kv_head
        self.attention_interface: Callable = ATTENTION_IMPLEMENTATIONS[config.attn_impl]

        # Init projection layers
        self.q_proj  = nn.Linear(self.config.hidden_dim, self.config.n_head * self.head_dim, bias=False)
        self.kv_proj = nn.Linear(self.config.hidden_dim, self.config.n_kv_head * self.head_dim * 2, bias=False)
        self.out_proj = nn.Linear(self.config.n_head * self.head_dim, self.config.hidden_dim, bias=False)

        self.attn_dropout = nn.Dropout(self.config.dropout)

        self.register_buffer("causal_mask", self._create_causal_mask(self.config.max_seq_len))
        self.register_buffer("alibi", self._build_alibi_bias(self.config.n_head))
        # SDPA can't apply our ALiBi bias, so it is RoPE-only.
        assert not (self.config.attn_impl == "sdpa" and not self.config.use_rope), (
            "attn_impl='sdpa' is not supported with ALiBi (use_rope=False); "
            "use attn_impl='eager' for ALiBi."
        )

        # RoPE
        half = self.head_dim // 2
        inv = 1.0 / (self.config.rope_theta ** (torch.arange(half, dtype=torch.float32) / half))
        self.register_buffer("rope_inv_freq", inv, persistent=False)

    def _build_alibi_bias(self, num_heads: int) -> Tensor:
        """Build ALiBi for specified number of heads:

        Returns:
            Tensor with ALiBi biases, shape: [1, num heads, 1, 1]
        """
        if (num_heads & (num_heads - 1)) == 0:
            start = 2 ** (-8.0 / num_heads)
            slopes = [start ** i for i in range(1, num_heads + 1)]
        else:
            closest = 1 << (num_heads.bit_length() - 1)
            start1 = 2 ** (-8.0 / closest)
            slopes1 = [start1 ** i for i in range(1, closest + 1)]
            start2 = 2 ** (-8.0 / (2 * closest))
            slopes2 = [start2 ** i for i in range(1, 2 * closest + 1)]
            slopes = slopes1 + slopes2[: num_heads - closest]
        return torch.tensor(slopes, dtype=torch.float32).view(1, num_heads, 1, 1)

    def _create_causal_mask(self, max_seq_len: int) -> Tensor:
        """Create causal mask with ones where tokens can attend to each other.

        Returns:
            Tensor with causal mask, shape: [1, 1, seq len, seq len]
        """
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        return mask.view(1, 1, max_seq_len, max_seq_len)

    def forward(self, x: Tensor, attention_mask: Tensor = None) -> Tensor:
        """Apply Self-Attention to input data with respect to pad tokens.

        Args:
            x: input tensor, shape [bs, seq len, hidden dim]
            attention_mask: mask with zeros for pad tokens, shape [bs, seq len, hidden dim]
        Returns:
            result tensor, shape [bs, seq len, hidden dim]
        """
        B, L, Z = x.shape

        q = self.q_proj(x).view(B, L, self.config.n_head, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv_proj(x).view(B, L, self.config.n_kv_head, 2, self.head_dim)
        k = kv[:, :, :, 0, :].permute(0, 2, 1, 3)
        v = kv[:, :, :, 1, :].permute(0, 2, 1, 3)

        if self.q_per_kv > 1:
            k = k.repeat_interleave(self.q_per_kv, dim=1)
            v = v.repeat_interleave(self.q_per_kv, dim=1)

        #RoPE
        if self.config.use_rope:
            t = torch.arange(L, device=x.device, dtype=torch.float32)
            freqs = torch.outer(t, self.rope_inv_freq)  # [L, half]
            cos = freqs.cos().to(q.dtype).view(1, 1, L, -1)
            sin = freqs.sin().to(q.dtype).view(1, 1, L, -1)

            half = self.head_dim // 2
            q1, q2 = q[..., :half], q[..., half:]
            k1, k2 = k[..., :half], k[..., half:]
            q = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
            k = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)

        # ALiBi bias only for the eager path without RoPE; None otherwise.
        alibi_bias = None if self.config.use_rope else self.alibi
        dropout_p = self.attn_dropout.p if self.training else 0.0

        y = self.attention_interface(
            q,
            k,
            v,
            attention_mask,
            dropout_p=dropout_p,
            scale=self.scale,
            attn_dropout=self.attn_dropout,
            alibi_bias=alibi_bias,
        )

        y = y.permute(0, 2, 1, 3).contiguous().view(B, L, self.config.n_head * self.head_dim)
        y = self.out_proj(y)
        return y
