import numpy as np
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
    """Dataset over a pre-tokenized corpus stored as fixed-length blocks.

    Expects a ``.npy`` file of shape ``(n_blocks, block_len)`` produced by the
    offline tokenization step (see eda/pretrain_dataset.ipynb): each row is one
    already-packed block of token ids of length ``block_len``.

    Because every block is already the right length and fully populated with
    real tokens, there is nothing to pad or truncate at load time — no collator
    is needed. ``__getitem__`` returns ``(input_ids, attention_mask)`` directly,
    matching the tuple that :func:`data_collator` produces, so the training loop
    is unchanged. The attention mask is all-ones (no padding in packed blocks).

    The array is memory-mapped, so the full corpus is never loaded into RAM;
    only the requested rows are read from disk.
    """

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
        # Copy out of the memmap into an owned int64 tensor for the model.
        block = np.asarray(self.data[idx, : self.seq_len], dtype=np.int64)
        input_ids = torch.from_numpy(block)
        # No padding inside a packed block -> every position attends.
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        return input_ids, attention_mask
    

def data_collator(
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

    return input_ids, attention_mask


def create_dataloader(
    dataset: Dataset,
    pad_token_id: int,
    max_seq_len: int,
    batch_size: int,
    drop_last: bool,
    shuffle: bool | None = None,
) -> DataLoader:
    collate_fn = partial(data_collator, pad_token_id=pad_token_id, max_seq_len=max_seq_len)
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
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, pin_memory=True
    )