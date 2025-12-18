from pydantic import BaseModel


class TokenizerConfig(BaseModel):
    vocab_size: int = 1024
    special_tokens: list[str] = ["[EOS]"]


class TransformerConfig(BaseModel):
    n_layer: int
    n_head: int
    n_kv_head: int
    hidden_dim: int
    intermediate_dim: int
    dropout: float = 0.1
    vocab_size: int = 1024
    max_seq_len: int = 128
    use_rope: bool = False
    rope_theta: float = 10000.0
    use_mla: bool = False
    latent_dim: int = 128
    

class TrainerConfig(BaseModel):
    max_seq_len: int = 128 
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    clip_grad_norm: float = 1.0
    n_steps: int = 10_000
    val_every_n_steps: int = 1_000
    