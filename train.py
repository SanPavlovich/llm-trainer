import json
import datetime
import os
import argparse
from pathlib import Path

from tqdm.notebook import tqdm
import numpy as np
import multiprocessing as mp
import torch
from datasets import load_dataset, load_from_disk

from src.dataset import (
    TextDataset,
    TokenIdsDataset,
    create_dataloader,
    create_token_ids_dataloader,
)
from src.tokenizer.bpe_tokenizer import ByteLevelBPETokenizer
from src.tokenizer.bpe_tokenizer import train as tokenizer_train
from src.tokenizer.bpe_tokenizer_fast import FastByteLevelBPETokenizer
from src.tokenizer.bpe_tokenizer_fast import train as tokenizer_fast_train
from src.tokenizer.mp_worker import _init_fast, tokenize_batch_fast
from src.model import TransformerForCausalLM
from src.trainer import Trainer
from src.schemas import TokenizerConfig, TransformerConfig, TrainerConfig, RunConfig
from src.utils import timeit, set_seed


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


@timeit
def tokenize_dataset(
    texts: list[str], 
    max_seq_len: int, 
    out_path: Path,
    tokenizer: ByteLevelBPETokenizer,
    batch_size: int=1024, 
) -> None:
    n_procs = min(16, mp.cpu_count())
    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]

    # Каждый воркер токенизирует свой батч и возвращает плоский uint16-массив id.
    with mp.Pool(processes=n_procs, initializer=_init_fast, initargs=(tokenizer,)) as pool:
        pieces = list(
            tqdm(
                pool.imap(tokenize_batch_fast, batches),
                total=len(batches),
                desc="Tokenizing (mp + numba)",
            )
        )

    # Склеиваем в один поток и режем на блоки max_seq_len (хвост отбрасываем).
    all_ids = np.concatenate(pieces)
    n_blocks = len(all_ids) // max_seq_len
    dropped = len(all_ids) - n_blocks * max_seq_len
    blocks = all_ids[: n_blocks * max_seq_len].reshape(n_blocks, max_seq_len)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, blocks)

    print(f"всего токенов:    {len(all_ids):_}")
    print(f"блоков x{max_seq_len}:  {n_blocks:_}  (отброшено {dropped} токенов хвоста)")
    print(f"shape:            {blocks.shape}, dtype: {blocks.dtype}")
    print(f"сохранено:        {out_path}  ({blocks.nbytes / 1e6:.1f} MB)")
    return


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
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    save_config(exp_subdir, config)

    if config.tokenizer.cache_dir is None:
        raise ValueError("tokenizer cache_dir must not be None!")
    tokenizer_cache_dir = root_dir / "tokenizer_cache"
    tokenizer_cache_dir_full = tokenizer_cache_dir / config.tokenizer.cache_dir
    tokenizer_cache_dir_full.mkdir(parents=True, exist_ok=True)

    # 169K samples - 561_063_303 tokens 
    # 300K samples - 749_678_477 tokens
    if config.dataset_type == "token_ids":
        # Pre-tokenized corpus: fixed-length blocks loaded from a .npy file.
        # The tokenizer is only needed for validation-text generation, so it is
        # loaded from cache (must have been trained/saved beforehand).
        if config.dataset_path is None:
            raise ValueError("dataset_type='token_ids' requires dataset_path!")

        (root_dir / "datasets").mkdir(parents=True, exist_ok=True)
        dataset_dir = root_dir / "datasets" / config.dataset_path
        assert config.tokenized_dataset_path is not None, "tokenized_dataset_path is None. Set this parameter to save tokenized dataset!"
        tokenized_dataset_dir = root_dir / "datasets" / config.tokenized_dataset_path
    
        if not os.path.exists(tokenized_dataset_dir):
            dataset = load_from_disk(str(dataset_dir))
            vocab_fast, merges_fast = tokenizer_fast_train(
                data=dataset["text"], 
                vocab_size=tokenizer_config.vocab_size,
                special_tokens=tokenizer_config.special_tokens,
            )

            save_tokenizer_files(path=tokenizer_cache_dir_full, vocab=vocab_fast, merges=merges_fast)
            tokenizer = ByteLevelBPETokenizer.from_pretrained(tokenizer_cache_dir_full)

            tokenize_dataset(
                texts=dataset["text"], 
                max_seq_len=train_config.max_seq_len, 
                out_path=root_dir / "datasets" / config.tokenized_dataset_path,
                tokenizer=tokenizer
            )
        else:
            tokenizer = ByteLevelBPETokenizer.from_pretrained(tokenizer_cache_dir_full)

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
            vocab, merges = tokenizer_train(data=dataset["train"]["jokes"], **tokenizer_config.model_dump())
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
