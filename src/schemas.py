from pathlib import Path

import yaml
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
    val_after_train: bool = False


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

    # CUDA memory snapshot: records the allocation history of a single iteration
    # and dumps a pickle viewable at https://pytorch.org/memory_viz
    memory_snapshot: bool = False
    memory_snapshot_step: int = 9          # iteration (iter_num, 0-indexed) to snapshot
    memory_snapshot_max_entries: int = 100_000


class RunConfig(BaseModel):
    """Top-level configuration for a training run, loaded from config.yaml."""

    run_name: str = "llm_small"
    exp_name: str = "baseline"
    seed: int = 42
    deterministic: bool = True
    shuffle_train: bool = False

    dataset_path: str = "hf://datasets/IgorVolochay/russian_jokes/dataset.json"
    test_size: float = 0.1
    valid_texts: list[str] = []

    tokenizer: TokenizerConfig = TokenizerConfig()
    model: TransformerConfig
    trainer: TrainerConfig = TrainerConfig()
    profiler: ProfilerConfig = ProfilerConfig()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RunConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)
