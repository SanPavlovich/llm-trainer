from typing import Any, Callable
import numpy as np
import datasets
import torch
from torch.utils.data import Dataset, DataLoader
from functools import partial


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        texts = self.texts[idx]
        tokenized_sequence = self.tokenizer.encode(texts)
        return tokenized_sequence


class TokenIdsDataset(torch.utils.data.Dataset):
    def __init__(self, path, max_seq_len: int | None = None):
        # mmap_mode='r' keeps the corpus on disk; rows are read lazily.
        self.data = np.load(path, mmap_mode="r")
        if self.data.ndim != 2:
            raise ValueError(
                f"expected a 2D (n_blocks, block_len) array, got shape {self.data.shape}"
            )
        block_len = self.data.shape[1]
        # Allow using a shorter context than the stored block length.
        self.seq_len = block_len if max_seq_len is None else min(max_seq_len, block_len)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        block = np.asarray(self.data[idx, : self.seq_len], dtype=np.int64)
        input_ids = torch.from_numpy(block)
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

        return {
            "input_ids": input_ids, 
            "attention_mask": attention_mask
        }


class VisualTextDataset(torch.utils.data.Dataset):
    """Image-conditioned text dataset.

    Each sample becomes the sequence:

        [IMG_START] [IMG] x num_patches [IMG_END] <text tokens ...> [EOS]

    The [IMG] placeholders occupy the slots whose embeddings the model replaces
    with the vision-adapter patch features. ``image_ids_mask`` is True exactly
    on those placeholder positions, so the model (and the loss) can tell image
    slots from text slots.
    """

    def __init__(
        self,
        dataset: datasets.Dataset,
        text_field: str,
        image_field: str,
        tokenizer: Any,
        image_processor: Any,
        num_image_patches: int = 49,
    ):
        self.dataset = dataset
        self.text_field = text_field
        self.image_field = image_field
        assert self.text_field in self.dataset.features and self.image_field in self.dataset.features
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.num_image_patches = num_image_patches

        assert tokenizer.image_token_id is not None, (
            "tokenizer has no image tokens; call tokenizer.add_image_tokens() first"
        )

        # [IMG_START] [IMG]*P [IMG_END]
        self.image_tokens_prefix = (
            [tokenizer.image_start_token_id]
            + [tokenizer.image_token_id] * num_image_patches
            + [tokenizer.image_end_token_id]
        )
        # True only on the P patch placeholders; the START/END markers are real
        # (text) tokens that the model embeds and predicts normally.
        self.image_ids_mask_prefix = [False] + [True] * num_image_patches + [False]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Single row fetch: indexing the HF dataset once decodes the row (incl.
        # the image) exactly one time; pulling fields off the dict is then free.
        row = self.dataset[idx]

        tokenized_sequence = self.tokenizer.encode(row[self.text_field])

        # image_processor is the fast (torchvision) CLIP processor; it returns a
        # torch tensor directly. The heavy resize/normalize runs in the worker
        # process, off the training hot path.
        pixel_values = self.image_processor(
            row[self.image_field], return_tensors="pt"
        )["pixel_values"][0]  # torch.Tensor [3, H, W]

        input_ids = self.image_tokens_prefix + tokenized_sequence
        image_ids_mask = self.image_ids_mask_prefix + [False] * len(tokenized_sequence)

        return {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "image_ids_mask": image_ids_mask,
        }


def flatten_collator(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    input_ids = []
    attention_mask =[]
    for b in batch:
        input_ids.append(b["input_ids"])
        attention_mask.append(b["attention_mask"])

    return {
        "input_ids": torch.stack(input_ids), 
        "attention_mask": torch.stack(attention_mask)
    }

def pad_sequences(
    tokenized_sequences: list[list[int]], pad_token_id: int, max_seq_len: int = None
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = len(tokenized_sequences)
    max_batch_seq_len = min(max_seq_len, max((len(it) for it in tokenized_sequences)))

    input_ids = torch.full((batch_size, max_batch_seq_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_batch_seq_len), dtype=torch.bool)

    for i, tok_seq in enumerate(tokenized_sequences):
        cur_len = min(len(tok_seq), max_batch_seq_len)
        input_ids[i, :cur_len] = torch.tensor(tok_seq[:cur_len], dtype=torch.long)
        attention_mask[i, :cur_len] = 1

    return {
        "input_ids": input_ids, 
        "attention_mask": attention_mask
    }


def data_collator(
    tokenized_sequences: list[list[int]],
    pad_token_id: int,
    max_seq_len: int | None = None,
):
    padded = pad_sequences(
        tokenized_sequences,
        pad_token_id=pad_token_id,
        max_seq_len=max_seq_len,
    )
    return padded["input_ids"], padded["attention_mask"]


def pad_mask(
    masks: list[list[bool]],
    target_len: int,
) -> torch.Tensor:
    """Pad a list of per-sample bool masks to ``target_len`` (right-padded False).

    Used for image_ids_mask so it stays aligned with the padded input_ids.
    """
    batch_size = len(masks)
    out = torch.zeros((batch_size, target_len), dtype=torch.bool)
    for i, m in enumerate(masks):
        cur_len = min(len(m), target_len)
        out[i, :cur_len] = torch.tensor(m[:cur_len], dtype=torch.bool)
    return out


def multimodal_data_collator(
    batch,
    pad_token_id: int,
    max_seq_len: int | None = None,
):
    padded = pad_sequences(
        [sample["input_ids"] for sample in batch],
        pad_token_id=pad_token_id,
        max_seq_len=max_seq_len,
    )
    input_ids = padded["input_ids"]
    attention_mask = padded["attention_mask"]
    target_len = input_ids.size(1)

    # image_ids_mask must line up with the (possibly truncated) padded input_ids.
    image_ids_mask = pad_mask(
        [sample["image_ids_mask"] for sample in batch], target_len
    )

    # One image per sample: [B, 3, H, W]. pixel_values arrive as [3, H, W] tensors.
    pixel_values = torch.stack([sample["pixel_values"] for sample in batch])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "image_ids_mask": image_ids_mask,
    }


def create_dataloader(
    dataset: Dataset,
    collator: Callable,
    pad_token_id: int,
    max_seq_len: int,
    batch_size: int,
    drop_last: bool,
    shuffle: bool | None = None,
) -> DataLoader:
    collate_fn = partial(collator, pad_token_id=pad_token_id, max_seq_len=max_seq_len)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, collate_fn=collate_fn, pin_memory=True
    )


def create_token_ids_dataloader(
    dataset: Dataset,
    batch_size: int,
    drop_last: bool,
    shuffle: bool | None = None,
) -> DataLoader:
    """DataLoader for :class:`TokenIdsDataset`.

    Blocks are already fixed-length, so there is nothing to collate: PyTorch's
    default collation stacks the per-item ``(input_ids, attention_mask)`` tuples
    into batched tensors of shape ``[batch_size, seq_len]`` — the same contract
    the training loop expects from :func:`create_dataloader`.
    """
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, collate_fn=flatten_collator, pin_memory=True
    )