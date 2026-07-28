import sys
sys.path.append("C:/vscode_projects/complete/LLM")

import time
from functools import wraps

from tqdm.notebook import tqdm
import numpy as np
from datasets import load_from_disk
from pathlib import Path
import multiprocessing as mp

from token_count_worker import _init_fast, tokenize_batch_fast
from src.tokenizer import ByteLevelBPETokenizer
from src.tokenizer_fast import FastByteLevelBPETokenizer


MAX_SEQ_LEN = 192
BATCH_SIZE = 1024
OUT_PATH = Path(
    "C:/vscode_projects/complete/LLM/datasets/"
    f"NotEvilAI_ruwiki_169K_tokenized_msl{MAX_SEQ_LEN}.npy"
)

cache_dir = Path("C:/vscode_projects/complete/LLM/cache")
tokenizer = ByteLevelBPETokenizer.from_pretrained(cache_dir)
assert len(tokenizer.token2id) <= np.iinfo(np.uint16).max + 1, \
    "vocab не влезает в uint16 — поменяйте dtype в tokenize_batch_fast"


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        duration = end_time - start_time
        print(f"Function '{func.__name__}' took {duration:.4f} seconds to run.")
        return result
    return wrapper


@timeit
def tokenize_dataset(texts: list[str], max_seq_len: int, out_path: Path) -> np.ndarray:
    n_procs = min(16, mp.cpu_count())
    batches = [texts[i : i + BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]

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
    return blocks


dataset_dir = "C:/vscode_projects/complete/LLM/datasets/NotEvilAI_ruwiki_169K_samples"
dataset = load_from_disk(dataset_dir)
blocks = tokenize_dataset(dataset["text"], MAX_SEQ_LEN, OUT_PATH)