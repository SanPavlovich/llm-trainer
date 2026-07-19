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