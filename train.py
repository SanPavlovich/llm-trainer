import json
import datetime
import os
import argparse
from pathlib import Path

import torch
from datasets import load_dataset

from src.dataset import (
    TextDataset,
    TokenIdsDataset,
    create_dataloader,
    create_token_ids_dataloader,
)
from src.tokenizer.bpe_tokenizer import ByteLevelBPETokenizer, train
from src.tokenizer.bpe_tokenizer_fast import FastByteLevelBPETokenizer
from src.tokenizer.bpe_tokenizer_fast import train as fast_train
from src.model import TransformerForCausalLM
from src.trainer import Trainer
from src.schemas import TokenizerConfig, TransformerConfig, TrainerConfig, RunConfig
from src.utils import set_seed


def save_tokenizer_files(
    path: Path,
    vocab: dict[str, int],
    merges: list[tuple[str, str]]
) -> None:
    with open(path / "vocabulary.json", "w") as f:
        json.dump(vocab, f, indent=2)
    with open(path / "merges.json", "w") as f:
        json.dump({"merges": merges}, f)


def save_config(
    path: Path,
    config: RunConfig
) -> None:
    with open(path / "run_config.json", "w") as f:
        json.dump(config.model_dump(), f, indent=4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the language model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "configs" / "base.yaml",
        help="Path to the run config YAML file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = RunConfig.from_yaml(args.config)

    set_seed(config.seed, deterministic=config.deterministic)

    tokenizer_config = config.tokenizer
    model_config = config.model
    train_config = config.trainer

    time_str_format = datetime.datetime.now().strftime('%m-%d-%Y--%H-%M-%S')
    root_dir = Path(__file__).resolve().parent
    exp_subdir = root_dir/ "runs" / config.run_name / f"{config.exp_name}_{time_str_format}"

    tensorboard_dir = exp_subdir / "tensorboard"
    profiler_dir = exp_subdir / "pytorch_profiler"
    checkpoint_dir = tensorboard_dir  # checkpoints live alongside the tensorboard logs
    tokenizer_cache_dir = root_dir / "tokenizer_cache"
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_cache_dir.mkdir(parents=True, exist_ok=True)

    save_config(exp_subdir, config)

    if config.tokenizer.cache_dir is None:
        raise ValueError("tokenizer cache_dir must not be None!")
    tokenizer_cache_dir_full = tokenizer_cache_dir / config.tokenizer.cache_dir

    if config.dataset_type == "token_ids":
        # Pre-tokenized corpus: fixed-length blocks loaded from a .npy file.
        # The tokenizer is only needed for validation-text generation, so it is
        # loaded from cache (must have been trained/saved beforehand).
        if config.tokenized_dataset_path is None:
            raise ValueError("dataset_type='token_ids' requires tokenized_dataset_path")

        if os.path.exists(tokenizer_cache_dir_full / "vocabulary.json") and os.path.exists(tokenizer_cache_dir_full / "merges.json"):
            tokenizer = ByteLevelBPETokenizer.from_pretrained(tokenizer_cache_dir_full)
        else:
            raise ValueError("tokenizer files for token_ids dataset not found!")

        full_dataset = TokenIdsDataset(
            path=root_dir / "datasets" / config.tokenized_dataset_path, 
            max_seq_len=train_config.max_seq_len
        )
        n_test = int(len(full_dataset) * config.test_size)
        n_train = len(full_dataset) - n_test
        # Sequential split (no shuffle) to keep runs reproducible, matching the
        # text path's seed-based determinism.
        train_dataset = torch.utils.data.Subset(full_dataset, range(n_train))
        test_dataset = torch.utils.data.Subset(full_dataset, range(n_train, len(full_dataset)))

        train_dataloader = create_token_ids_dataloader(
            train_dataset, batch_size=train_config.batch_size,
            drop_last=True, shuffle=config.shuffle_train,
        )
        test_dataloader = create_token_ids_dataloader(
            test_dataset, batch_size=train_config.batch_size,
            drop_last=False, shuffle=False,
        )
    else:
        dataset = load_dataset("json", data_files=config.dataset_path)
        dataset = dataset["train"].train_test_split(test_size=config.test_size, seed=config.seed)

        if os.path.exists(tokenizer_cache_dir_full / "vocabulary.json") and os.path.exists(tokenizer_cache_dir_full / "merges.json"):
            tokenizer = ByteLevelBPETokenizer.from_pretrained(tokenizer_cache_dir_full)
        else:
            vocab, merges = train(data=dataset["train"]["jokes"], **tokenizer_config.model_dump())
            save_tokenizer_files(tokenizer_cache_dir_full, vocab, merges)
            tokenizer = ByteLevelBPETokenizer(vocab, merges)

        train_dataset = TextDataset(dataset["train"]["jokes"], tokenizer)
        train_dataloader = create_dataloader(
            train_dataset, tokenizer.eos_token_id, max_seq_len=train_config.max_seq_len,
            batch_size=train_config.batch_size, drop_last=True, shuffle=config.shuffle_train
        )
        test_dataset = TextDataset(dataset["test"]["jokes"], tokenizer)
        test_dataloader = create_dataloader(
            test_dataset, tokenizer.eos_token_id, max_seq_len=train_config.max_seq_len,
            batch_size=train_config.batch_size, drop_last=False, shuffle=False
        )

    model = TransformerForCausalLM(model_config)

    trainer = Trainer(
        tokenizer=tokenizer,
        model=model,
        train_config=train_config,
        valid_texts=config.valid_texts,
        tensorboard_dir=tensorboard_dir,
        checkpoint_dir=checkpoint_dir,
        profiler_dir=profiler_dir,
        profiler_config=config.profiler,
    )
    trainer.train(train_dataloader, test_dataloader)
