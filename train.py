import json
import datetime
import os
import argparse
from pathlib import Path
from datasets import load_dataset

from src.dataset import TextDataset, create_dataloader
from src.tokenizer import ByteLevelBPETokenizer, train
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
    cache_dir = root_dir / "cache"
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    save_config(exp_subdir, config)

    dataset = load_dataset("json", data_files=config.dataset_path)
    dataset = dataset["train"].train_test_split(test_size=config.test_size, seed=config.seed)

    if os.path.exists(cache_dir / "vocabulary.json") and os.path.exists(cache_dir / "merges.json"):
        tokenizer = ByteLevelBPETokenizer.from_pretrained(cache_dir)
    else:
        vocab, merges = train(data=dataset["train"]["jokes"], **tokenizer_config.model_dump())
        save_tokenizer_files(cache_dir, vocab, merges)
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
