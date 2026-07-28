"""Helper module for parallel token counting.

Lives in a separate importable file because on Windows `multiprocessing` uses
the `spawn` start method: worker functions must be importable by their qualified
name (functions defined in a notebook cell are not picklable).
"""

_tokenizer = None


def _init(tokenizer):
    """Pool initializer: stash the tokenizer once per worker process.

    Passing it via the initializer (instead of as an argument to every task)
    means it is pickled and sent one time per worker, not per sample.
    """
    global _tokenizer
    _tokenizer = tokenizer


def count_tokens(text: str) -> int:
    """Count tokens in a single sample without materializing the id list."""
    return len(_tokenizer.encode(text))


def count_tokens_batch(batch: list[str]) -> int:
    """Count tokens over a batch of samples (fewer IPC round-trips)."""
    enc = _tokenizer.encode
    return sum(len(enc(text)) for text in batch)


# --- Fast (numba) variant -------------------------------------------------
# The FastByteLevelBPETokenizer holds numpy lookup matrices and a JIT-compiled
# kernel. We can't just pickle a warmed-up instance across the spawn boundary,
# so each worker receives the small picklable base tokenizer and builds +
# warms up its own fast tokenizer once, in the initializer. That pays the JIT
# compile cost one time per worker (cached on disk by numba across runs).

_fast_tokenizer = None


def _init_fast(base_tokenizer):
    """Pool initializer: build and warm up a fast tokenizer per worker."""
    global _fast_tokenizer
    import sys
    from pathlib import Path

    # Ensure the project root is importable inside spawned worker processes.
    project_root = str(Path(__file__).resolve().parents[1])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from src.tokenizer.bpe_tokenizer_fast import FastByteLevelBPETokenizer

    _fast_tokenizer = FastByteLevelBPETokenizer.from_tokenizer(base_tokenizer).warmup()


def count_tokens_batch_fast(batch: list[str]) -> int:
    """Count tokens over a batch using the numba-accelerated tokenizer."""
    count = _fast_tokenizer.count
    return sum(count(text) for text in batch)


def tokenize_batch_fast(batch: list[str]):
    """Tokenize a batch of documents into one flat uint16 array of token ids.

    Each document is encoded with a trailing EOS token (encode's default), so
    the concatenation carries document boundaries. Returned as uint16 because
    the tokenizer vocab fits in 16 bits (checked once on the parent side).
    """
    import numpy as np

    encode = _fast_tokenizer.encode
    ids = []
    for text in batch:
        ids.extend(encode(text))  # add_eos_token=True by default
    return np.asarray(ids, dtype=np.uint16)
