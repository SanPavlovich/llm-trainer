import json
import datetime
import os
import argparse
from pathlib import Path
import logging

from tqdm.notebook import tqdm
import numpy as np
import multiprocessing as mp
import torch
from datasets import load_dataset, load_from_disk

from src.dataset import (
    TextDataset,
    TokenIdsDataset,
    VisualTextDataset,
    create_dataloader,
    create_token_ids_dataloader,
    multimodal_data_collator,
)
from src.tokenizer.bpe_tokenizer import ByteLevelBPETokenizer
from src.tokenizer.bpe_tokenizer import train as tokenizer_train
from src.tokenizer.bpe_tokenizer_fast import FastByteLevelBPETokenizer
from src.tokenizer.bpe_tokenizer_fast import train as tokenizer_fast_train
from src.tokenizer.mp_worker import _init_fast, tokenize_batch_fast
from src.model import TransformerForCausalLM, TransformerForVisualCausalLM
from src.trainer import Trainer
from src.schemas import TokenizerConfig, TransformerConfig, TrainerConfig, RunConfig
from src.utils import timeit, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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


def get_valid_images(split):
    # Fix 5 random validation images once; we caption these every eval so
    # progress is visually comparable across steps.
    rng = np.random.default_rng(config.seed)
    n_fixed = min(5, len(split["test"]))
    fixed_idx = rng.choice(len(split["test"]), size=n_fixed, replace=False).tolist()

    fixed_pixel_values, fixed_display, fixed_captions = [], [], []
    for j in fixed_idx:
        row = split["test"][int(j)]
        pil_img = row[config.image_field]
        proc = image_processor(pil_img, return_tensors="pt")["pixel_values"][0]  # [3,H,W]
        fixed_pixel_values.append(proc)
        # Display copy: original image resized/normalized to 0..1 for TensorBoard.
        disp = image_processor(pil_img, return_tensors="pt", do_normalize=False)["pixel_values"][0]
        fixed_display.append(disp.clamp(0, 1))
        cap = row[config.text_field]
        fixed_captions.append(cap if isinstance(cap, str) else str(cap))

    return {
        "pixel_values": torch.stack(fixed_pixel_values),
        "display_images": torch.stack(fixed_display),
        "captions": fixed_captions,
    }


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
    checkpoint_dir = root_dir / "model_checkpoint" / config.run_name / f"{config.exp_name}_{time_str_format}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    tensorboard_dir = exp_subdir / "tensorboard"
    profiler_dir = exp_subdir / "pytorch_profiler"
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    save_config(exp_subdir, config)

    if config.tokenizer.cache_dir is None:
        raise ValueError("tokenizer cache_dir must not be None!")
    tokenizer_cache_dir = root_dir / "tokenizer_cache"
    tokenizer_cache_dir_full = tokenizer_cache_dir / config.tokenizer.cache_dir
    tokenizer_cache_dir_full.mkdir(parents=True, exist_ok=True)
    pretrained_tokenizer_cache_dir = root_dir / "tokenizer_cache" / train_config.pretrained_tokenizer_path

    # 169K samples - 561_063_303 tokens
    # 300K samples - 749_678_477 tokens
    valid_images = None  # populated only for dataset_type == "vision"
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
            logger.info(f"start training tokenizer on dataset: {dataset_dir}")
            dataset = load_from_disk(str(dataset_dir))
            vocab_fast, merges_fast = tokenizer_fast_train(
                data=dataset["text"], 
                vocab_size=tokenizer_config.vocab_size,
                special_tokens=tokenizer_config.special_tokens,
            )

            save_tokenizer_files(path=tokenizer_cache_dir_full, vocab=vocab_fast, merges=merges_fast)
            tokenizer = ByteLevelBPETokenizer.from_pretrained(tokenizer_cache_dir_full)

            logger.info(f"start dataset tokenization: {dataset_dir}")
            tokenize_dataset(
                texts=dataset["text"], 
                max_seq_len=train_config.max_seq_len, 
                out_path=root_dir / "datasets" / config.tokenized_dataset_path,
                tokenizer=tokenizer
            )
            logger.info(f"tokenized dataset saved at {root_dir / "datasets" / config.tokenized_dataset_path}")
        else:
            logger.info(f"pretrained tokenizer loaded from {pretrained_tokenizer_cache_dir}")
            tokenizer = ByteLevelBPETokenizer.from_pretrained(pretrained_tokenizer_cache_dir)

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
    elif config.dataset_type == "vision":
        # Image + text SFT. Load a pretrained tokenizer, add image special
        # tokens, then build image/text pairs for the multimodal collator.
        from functools import partial

        assert config.vision_adapter is not None, "vision dataset_type requires a vision_adapter config"
        va = config.vision_adapter

        tokenizer = ByteLevelBPETokenizer.from_pretrained(pretrained_tokenizer_cache_dir)
        tokenizer.add_image_tokens(
            image_start_token=va.image_start_token,
            image_token=va.image_token,
            image_end_token=va.image_end_token,
        )
        logger.info(f"tokenizer loaded from {pretrained_tokenizer_cache_dir}")
        logger.info(f"vocab grown to {len(tokenizer.token2id)} with image tokens")

        # use_fast=True -> torchvision-backed CLIPImageProcessorFast: ~1.7x
        # faster resize/normalize than the default PIL path, and it returns
        # torch tensors directly.
        from transformers import AutoImageProcessor
        image_processor = AutoImageProcessor.from_pretrained(
            va.vision_model_repo_id, use_fast=True
        )

        dataset = load_dataset(config.dataset_path)
        split = dataset["train"] if "train" in dataset else dataset[list(dataset.keys())[0]]
        split = split.train_test_split(test_size=config.test_size, seed=config.seed)

        train_dataset = VisualTextDataset(
            split["train"], config.text_field, config.image_field,
            tokenizer, image_processor, num_image_patches=va.num_image_patches,
        )
        test_dataset = VisualTextDataset(
            split["test"], config.text_field, config.image_field,
            tokenizer, image_processor, num_image_patches=va.num_image_patches,
        )

        collate = partial(
            multimodal_data_collator,
            pad_token_id=tokenizer.eos_token_id,
            max_seq_len=train_config.max_seq_len,
        )
        # With workers, image decode / CLIP resize / tokenization run in parallel
        # background processes and are prefetched, so the GPU doesn't stall on
        # the dataset. persistent_workers avoids re-spawning them every epoch.
        loader_kwargs = dict(collate_fn=collate, pin_memory=True, num_workers=train_config.num_workers)
        if train_config.num_workers > 0:
            loader_kwargs.update(
                persistent_workers=True,
                prefetch_factor=train_config.prefetch_factor,
            )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=train_config.batch_size, shuffle=config.shuffle_train,
            drop_last=True, **loader_kwargs,
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=train_config.batch_size, shuffle=False,
            drop_last=False, **loader_kwargs,
        )
        valid_images = get_valid_images(split)
    else:
        dataset = load_dataset("json", data_files=config.dataset_path)
        dataset = dataset["train"].train_test_split(test_size=config.test_size, seed=config.seed)

        if os.path.exists(pretrained_tokenizer_cache_dir / "vocabulary.json") and os.path.exists(pretrained_tokenizer_cache_dir / "merges.json"):
            tokenizer = ByteLevelBPETokenizer.from_pretrained(pretrained_tokenizer_cache_dir)
            logger.info(f"pretrained tokenizer loaded from {pretrained_tokenizer_cache_dir}")
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

    if config.dataset_type == "vision":
        model = TransformerForVisualCausalLM(model_config, config.vision_adapter)

        # Load the text-pretrained transformer weights into the base model. The
        # checkpoint has the OLD vocab size, so we resize the (randomly-init'd)
        # embedding/lm_head to the pretrain size, load, then grow for image tokens.
        if train_config.pretrained_model_path is not None:
            pretrained_model_dir = root_dir / "model_checkpoint" / Path(train_config.pretrained_model_path)
            pretrain_checkpoint = torch.load(pretrained_model_dir, map_location="cpu")
            pretrain_state = pretrain_checkpoint["model"]
            pretrain_vocab = pretrain_state["token_emb.weight"].shape[0]

            model.resize_token_embeddings(pretrain_vocab)
            # Non-strict: the checkpoint has no vision_adapter weights.
            missing, unexpected = model.load_state_dict(pretrain_state, strict=False)
            logger.info(
                f"pretrained checkpoint loaded from {pretrained_model_dir} "
                f"(missing={len(missing)} keys, unexpected={len(unexpected)})"
            )

        # Grow to fit the image special tokens appended to the tokenizer.
        if len(tokenizer.token2id) != model.vocab_size:
            model.resize_token_embeddings(len(tokenizer.token2id))
            logger.info(f"resized token embeddings to {model.vocab_size} (image tokens)")
    else:
        model = TransformerForCausalLM(model_config)
        if train_config.pretrained_model_path is not None:
            pretrained_model_dir = root_dir / "model_checkpoint" / Path(train_config.pretrained_model_path)
            pretrain_checkpoint = torch.load(pretrained_model_dir)
            model.load_state_dict(pretrain_checkpoint["model"])
            logger.info(f"pretrained checkpoint loaded from {pretrained_model_dir}")

    logger.info(f"Number of parameters: {model.n_params / 1e6:.2f}M")
    logger.info(f"Number of trainable parameters: {model.n_trainable_params / 1e6:.2f}M")
    logger.info(f"train_loader iterations count: {len(train_dataloader):_}")
    logger.info(f"val_loader   iterations count: {len(test_dataloader):_}")

    trainer = Trainer(
        tokenizer=tokenizer,
        model=model,
        train_config=train_config,
        valid_texts=config.valid_texts,
        tensorboard_dir=tensorboard_dir,
        checkpoint_dir=checkpoint_dir,
        profiler_dir=profiler_dir,
        profiler_config=config.profiler,
        valid_images=valid_images,
    )
    logger.info(f"Start training...")
    trainer.train(
        train_loader=train_dataloader, 
        val_loader=test_dataloader
    )
    logger.info(f"Training finised")
