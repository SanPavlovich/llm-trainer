from pathlib import Path

import yaml
from pydantic import BaseModel


class TokenizerConfig(BaseModel):
    cache_dir: str | None = None
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
    rmsnorm_eps: float = 1e-6
    attn_impl: str = "eager"
    rmsnorm_impl: str = "custom"


class VisionAdapterConfig(BaseModel):
    vision_model_repo_id: str
    input_dim: int
    output_dim: int
    # Number of patch embeddings the vision tower produces (CLS dropped).
    # openai/clip-vit-base-patch32 @ 224px -> (224/32)^2 = 49 patches.
    num_image_patches: int = 49
    # Special tokens injected around the image patch placeholders.
    image_start_token: str = "[IMG_START]"
    image_token: str = "[IMG]"
    image_end_token: str = "[IMG_END]"


class DDPConfig(BaseModel):
    enabled: bool = False
    dp_size: int = 1


class TrainerConfig(BaseModel):
    save_checkpoint: bool = False
    pretrained_model_path: str | None = None
    pretrained_tokenizer_path: str | None = None
    max_seq_len: int = 128
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    clip_grad_norm: float = 1.0
    n_steps: int = 10_000
    val_every_n_steps: int = 1_000
    val_after_train: bool = False
    num_workers: int = 4
    prefetch_factor: int = 4
    mixed_precision: bool = False
    amp_dtype: str = "bfloat16"             # ['bfloat16', 'float16']; float16 also enables GradScaler
    log_grad_norm_per_layer_every: int = 0
    compile: bool = False                   # torch.compile: JIT-compile the model for faster train/val steps (needs CUDA + triton)
    compile_mode: str = "max-autotune"


class ProfilerConfig(BaseModel):
    """Configuration for the PyTorch profiler.

    The profiler follows a wait -> warmup -> active schedule. To capture the
    N-th training iteration, set wait = N - 1, warmup = 0, active = 1
    (e.g. wait=9, warmup=0, active=1 traces the 10th iteration).
    """

    enabled: bool = False
    wait: int = 9          # steps to skip before profiling starts
    warmup: int = 0        # steps to warm up (traced but discarded)
    active: int = 1        # steps to actually record
    repeat: int = 1        # number of wait/warmup/active cycles (0 = repeat forever)
    record_shapes: bool = True
    profile_memory: bool = True
    with_stack: bool = True

    # CUDA memory snapshot: records the allocation history and dumps a pickle
    # viewable at https://pytorch.org/memory_viz
    memory_snapshot: bool = False
    memory_snapshot_step: int = 9
    memory_snapshot_stop_step: int | None = None
    memory_snapshot_max_entries: int = 100_000


class RunConfig(BaseModel):
    """Top-level configuration for a training run, loaded from config.yaml."""

    run_name: str = "llm_small"
    exp_name: str = "baseline"
    seed: int = 42
    deterministic: bool = True
    shuffle_train: bool = False

    dataset_type: str = "text"
    dataset_path: str = "hf://datasets/IgorVolochay/russian_jokes/dataset.json"
    # Path to the pre-tokenized (n_blocks, block_len) .npy; used when
    # dataset_type == "token_ids".
    tokenized_dataset_path: str | None = None
    test_size: float = 0.1
    valid_texts: list[str] = []

    # Column names for the "vision" dataset_type.
    image_field: str = "image"
    text_field: str = "text"

    tokenizer: TokenizerConfig = TokenizerConfig()
    model: TransformerConfig
    vision_adapter: VisionAdapterConfig | None = None
    trainer: TrainerConfig = TrainerConfig()
    profiler: ProfilerConfig = ProfilerConfig()
    ddp: DDPConfig = DDPConfig()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RunConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)
