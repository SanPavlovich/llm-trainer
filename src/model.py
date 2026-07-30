import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from huggingface_hub import PyTorchModelHubMixin

from src.attention import CausalSelfAttention, CausalSelfAttentionMLA
from src.schemas import TransformerConfig, VisionAdapterConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """Root Mean Square Layer Normalization

        Args:
            dim: Feature dimension
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(rms + self.eps)
        return self.scale * x_norm


class SwiGLU(nn.Module):
    def __init__(self, config: TransformerConfig):
        """Gated Liner Unit with Swish Activation"""
        super().__init__()
        self.config = config
        # Init up- and down- projection layers
        self.fc1 = nn.Linear(config.hidden_dim, 2 * config.intermediate_dim, bias=True)
        self.fc2 = nn.Linear(config.intermediate_dim, config.hidden_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        """Apply SwiGLU to input data.

        Args:
            x: input tensor, shape [bs, seq len, hidden dim]
        Returns:
            result tensor, shape [bs, seq len, hidden dim]
        """
        h = self.fc1(x)
        a, b = h.chunk(2, dim=-1)
        h = F.silu(a) * b
        h = self.fc2(h)
        return h


class Block(nn.Module):
    def __init__(self, config: TransformerConfig):
        """Base Transformer Block
        - Causal Self-Attention and SwiGLU as main elements
        - Pre-normalization via RMSNorm
        - Regularization with dropouts before residuals
        """
        super().__init__()
        self.ln_1 = RMSNorm(config.hidden_dim)
        self.res_dropout_1 = nn.Dropout(config.dropout)
        self.attn = CausalSelfAttentionMLA(config) if config.use_mla else CausalSelfAttention(config)

        self.ln_2 = RMSNorm(config.hidden_dim)
        self.res_dropout_2 = nn.Dropout(config.dropout)
        self.mlp = SwiGLU(config)

    def forward(self, x: Tensor, attention_mask: Tensor = None) -> Tensor:
        """Apply Transformer Block to input data.

        Args:
            x: input tensor, shape [bs, seq len, hidden dim]
            attention_mask: mask with zeros for pad tokens, shape [bs, seq len, hidden dim]
        Returns:
            result tensor, shape [bs, seq len, hidden dim]
        """
        x = x + self.res_dropout_1(self.attn(self.ln_1(x), attention_mask=attention_mask))
        x = x + self.res_dropout_2(self.mlp(self.ln_2(x)))
        return x


class TransformerForCausalLM(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config: TransformerConfig):
        """Transformer model for Language Modeling"""
        super().__init__()
        self.vocab_size = config.vocab_size
        self.max_seq_len = config.max_seq_len
        self.n_layer = config.n_layer
        self.n_head = config.n_head
        self.hidden_dim = config.hidden_dim
        self.dropout = config.dropout

        self.token_emb = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.emb_dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_final = RMSNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    @property
    def n_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.scale)

    def resize_token_embeddings(self, new_vocab_size: int) -> None:
        """Grow (or shrink) the token embedding and lm_head to ``new_vocab_size``.

        New rows are initialised like the rest of the model (normal(0, 0.02)) so
        freshly added ids (e.g. image special tokens) start from the same prior
        as the pretrained ones. Existing rows are copied over unchanged, so a
        text-pretrained checkpoint stays intact.
        """
        old_vocab_size = self.token_emb.num_embeddings
        if new_vocab_size == old_vocab_size:
            return

        device = self.token_emb.weight.device
        dtype = self.token_emb.weight.dtype

        new_emb = nn.Embedding(new_vocab_size, self.hidden_dim).to(device=device, dtype=dtype)
        torch.nn.init.normal_(new_emb.weight, mean=0.0, std=0.02)
        n = min(old_vocab_size, new_vocab_size)
        with torch.no_grad():
            new_emb.weight[:n] = self.token_emb.weight[:n]
        self.token_emb = new_emb

        new_head = nn.Linear(self.hidden_dim, new_vocab_size, bias=False).to(device=device, dtype=dtype)
        torch.nn.init.normal_(new_head.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            new_head.weight[:n] = self.lm_head.weight[:n]
        self.lm_head = new_head

        self.vocab_size = new_vocab_size

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> Tensor:
        """Calculate logits for given input ids.

        Args:
            x: input tensor, shape [bs, seq len, hidden dim]
            attention_mask: mask with zeros for pad tokens, shape [bs, seq len, hidden dim]
        Returns:
            logits, shape [bs, seq len, hidden dim]
        """
        B, L = input_ids.shape
        x = self.token_emb(input_ids)
        x = self.emb_dropout(x)
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits

    @torch.inference_mode()
    def generate(
        self, idx: Tensor, max_new_tokens, eos_token_id, temperature=1.0, do_sample=False, top_k=None
    ) -> Tensor:
        """Take a conditioning sequence of indices and complete the sequence max_new_tokens times,
        feeding the predictions back into the model each time.

        Args:
            idx: tensor with conditional tokens, shape [seq len]
            max_new_tokens: maximum number of new tokens
            eos_token_id: index of EOS token to stop generation
            temperature, do_sample, top_k: generation parameters
        Return:
            tensor with generated indexes
        """
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.shape[1] <= self.max_seq_len else idx[:, -self.max_seq_len :]
            logits = self(idx_cond)

            # 1. Pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / max(temperature, 1e-8)

            # 2. Optionally crop the logits to only the top k options
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                topk_vals, _ = torch.topk(logits, k, dim=-1)
                cutoff = topk_vals[:, [-1]]
                logits = logits.masked_fill(logits < cutoff, float("-inf"))

            # 3. apply softmax to convert logits to probabilities
            probs = F.softmax(logits, dim=-1)

            # 4. Either sample from the distribution or take the most likely element
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)  
            else:
                idx_next = torch.argmax(probs, dim=-1, keepdim=True)

            # 5. Append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
            if idx_next == eos_token_id:
                break
        return idx


class CLIPVisionAdapter(nn.Module):
    """Frozen CLIP vision tower + a trainable linear projection.

    Maps an image to ``num_image_patches`` embeddings of the LLM hidden size,
    one per vision patch (the CLS token is dropped). The CLIP tower is frozen
    and kept in eval mode; only ``adapter`` is trained.
    """

    def __init__(self, config: VisionAdapterConfig):
        super().__init__()
        from transformers import CLIPVisionModel
        self.vision_model = CLIPVisionModel.from_pretrained(config.vision_model_repo_id)
        for p in self.vision_model.parameters():
            p.requires_grad_(False)
        self.vision_model.eval()

        self.adapter = nn.Linear(config.input_dim, config.output_dim)

    def train(self, mode: bool = True):
        # Keep the frozen CLIP tower in eval mode even when the parent module
        # is switched to train() (so its dropout / LN stay deterministic).
        super().train(mode)
        self.vision_model.eval()
        return self

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """pixel_values: [n_images, 3, H, W] -> [n_images, num_patches, output_dim]."""
        with torch.no_grad():
            out = self.vision_model(pixel_values=pixel_values)
        # drop CLS patch embedding, keep per-patch features
        return self.adapter(out.last_hidden_state[:, 1:, :])


class TransformerForVisualCausalLM(TransformerForCausalLM):
    def __init__(self, transformer_config: TransformerConfig, adapter_config: VisionAdapterConfig):
        super().__init__(transformer_config)
        self.transformer_config = transformer_config
        self.adapter_config = adapter_config
        self.vision_adapter = CLIPVisionAdapter(adapter_config)

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        image_ids_mask: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,   # accepted & ignored; loss is computed in the trainer
    ):
        """
        Args:
            input_ids:      [B, L] token ids. Positions where image_ids_mask is
                            True hold the [IMG] placeholder id (value unused).
            pixel_values:   [N, 3, H, W] the images in this batch (N == number of
                            True entries in image_ids_mask, i.e. B * num_patches
                            when every sample has exactly one image).
            image_ids_mask: [B, L] bool, True at image-patch positions.
            attention_mask: [B, L] bool/0-1 padding mask.
        Returns:
            logits: [B, L, vocab_size]
        """
        x = self._embed_multimodal(input_ids, image_ids_mask, pixel_values=pixel_values)
        x = self.emb_dropout(x)
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits

    def _embed_multimodal(
        self,
        input_ids: torch.Tensor,
        image_ids_mask: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        img_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build input embeddings, replacing [IMG] slots with vision features.

        Provide exactly one of ``pixel_values`` (encoded here through the frozen
        CLIP tower + adapter) or ``img_embeds`` (already [N, P, H], reused across
        decoding steps in generate() so CLIP runs only once).

        Returns x: [B, L, H].
        """
        # Text embeddings for every position (image positions get overwritten).
        x = self.token_emb(input_ids)  # [B, L, H]

        if img_embeds is None:
            assert pixel_values is not None, "pass pixel_values or img_embeds"
            img_embeds = self.vision_adapter(pixel_values)  # [N, P, H]
        img_flat = img_embeds.reshape(-1, img_embeds.size(-1))  # [N*P, H]

        # The number of True positions in image_ids_mask must equal N*P — a
        # mismatch means the image block was truncated by max_seq_len.
        n_slots = int(image_ids_mask.sum().item())
        assert n_slots == img_flat.size(0), (
            f"image_ids_mask has {n_slots} True positions but the vision adapter "
            f"produced {img_flat.size(0)} patch embeddings; increase max_seq_len "
            f"so the image prefix is never truncated."
        )
        x = x.clone()
        x[image_ids_mask] = img_flat.to(x.dtype)
        return x

    @torch.inference_mode()
    def generate(
        self,
        input_ids: Tensor,
        pixel_values: Tensor,
        image_ids_mask: Tensor,
        max_new_tokens: int,
        eos_token_id: int,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_k: int | None = None,
    ) -> Tensor:
        """Autoregressively generate text conditioned on image(s).

        Args:
            input_ids:      [B, L0] prompt ids, including the image prefix
                            ([IMG_START] [IMG]*P [IMG_END]) and any text prompt.
            pixel_values:   [N, 3, H, W] images referenced by the [IMG] slots.
            image_ids_mask: [B, L0] bool, True at the [IMG] patch positions.
            max_new_tokens: number of tokens to sample.
            eos_token_id:   stop when this id is produced (single-sequence path).
        Returns:
            [B, L0 + generated] the prompt followed by the generated ids.
        """
        # Encode the image once; reuse across all decoding steps.
        img_embeds = self.vision_adapter(pixel_values)  # [N, P, H]

        idx = input_ids
        mask = image_ids_mask
        for _ in range(max_new_tokens):
            # The image prefix must stay in the window (its [IMG] slots are what
            # the patch embeddings scatter into). Cropping it would break the
            # scatter, so we stop once the sequence fills the context window.
            if idx.shape[1] >= self.max_seq_len:
                break
            idx_cond = idx
            mask_cond = mask

            x = self._embed_multimodal(idx_cond, mask_cond, img_embeds=img_embeds)
            x = self.emb_dropout(x)
            for layer in self.layers:
                x = layer(x)
            x = self.ln_final(x)
            logits = self.lm_head(x)

            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                topk_vals, _ = torch.topk(logits, k, dim=-1)
                cutoff = topk_vals[:, [-1]]
                logits = logits.masked_fill(logits < cutoff, float("-inf"))

            probs = F.softmax(logits, dim=-1)
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(probs, dim=-1, keepdim=True)

            idx = torch.cat((idx, idx_next), dim=1)
            # Newly generated tokens are always text -> mask is False for them.
            mask = torch.cat(
                (mask, torch.zeros_like(idx_next, dtype=torch.bool)), dim=1
            )
            if idx_next.numel() == 1 and idx_next.item() == eos_token_id:
                break
        return idx