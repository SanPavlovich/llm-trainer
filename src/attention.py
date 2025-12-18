import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.schemas import TransformerConfig


class CausalSelfAttentionMLA(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        assert self.config.hidden_dim % self.config.n_head == 0
        assert self.config.n_head % self.config.n_kv_head == 0
        self.head_dim = self.config.hidden_dim // self.config.n_head
        self.scale = self.head_dim ** -0.5
        self.q_per_kv = self.config.n_head // self.config.n_kv_head

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

        att = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B,H,L,L]

        if not self.config.use_rope:
            i = torch.arange(L, device=x.device).view(1, 1, L, 1)
            j = torch.arange(L, device=x.device).view(1, 1, 1, L)
            att = att + (self.alibi.to(att.dtype) * (j - i))

        att = att.masked_fill(~self.causal_mask[:, :, :L, :L], torch.finfo(att.dtype).min)
        if attention_mask is not None:
            key_mask = attention_mask if attention_mask.dim() == 2 else attention_mask[..., 0]
            key_mask = (key_mask > 0).view(B, 1, 1, L)
            att = att.masked_fill(~key_mask, torch.finfo(att.dtype).min)

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = torch.matmul(att, v)
        y = y.permute(0, 2, 1, 3).contiguous().view(B, L, self.config.n_head * self.head_dim)
        y = self.out_proj(y)
        return y
    

class CausalSelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
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

        # Init projection layers
        self.q_proj  = nn.Linear(self.config.hidden_dim, self.config.n_head * self.head_dim, bias=False)
        self.kv_proj = nn.Linear(self.config.hidden_dim, self.config.n_kv_head * self.head_dim * 2, bias=False)
        self.out_proj = nn.Linear(self.config.n_head * self.head_dim, self.config.hidden_dim, bias=False)

        self.attn_dropout = nn.Dropout(self.config.dropout)

        self.register_buffer("causal_mask", self._create_causal_mask(self.config.max_seq_len))
        self.register_buffer("alibi", self._build_alibi_bias(self.config.n_head))

        # тут RoPE
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

        
        att = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if not self.config.use_rope:
            i = torch.arange(L, device=x.device).view(1, 1, L, 1)
            j = torch.arange(L, device=x.device).view(1, 1, 1, L)
            att = att + (self.alibi.to(att.dtype) * (j - i))

        causal = self.causal_mask[:, :, :L, :L]
        att = att.masked_fill(~causal, torch.finfo(att.dtype).min)

        if attention_mask is not None:
            key_mask = attention_mask
            if key_mask.dim() == 3:
                key_mask = key_mask[..., 0]
            key_mask = (key_mask > 0).view(B, 1, 1, L)
            att = att.masked_fill(~key_mask, torch.finfo(att.dtype).min)

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = torch.matmul(att, v)
        y = y.permute(0, 2, 1, 3).contiguous().view(B, L, self.config.n_head * self.head_dim)
        y = self.out_proj(y)
        return y
