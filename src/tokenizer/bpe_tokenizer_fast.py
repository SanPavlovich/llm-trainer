"""Numba-accelerated Byte-Level BPE tokenizer.

Drop-in faster variant of :class:`src.tokenizer.ByteLevelBPETokenizer`.

The hot part of BPE encoding is the per-word merge loop: repeatedly find the
highest-priority adjacent pair and merge it. In the pure-Python implementation
that loop walks tuples of *strings* and does dict lookups on every pair, which
is slow.

Here we do the same algorithm, but:

1. Everything is expressed in terms of integer token ids, not strings.
2. The merge priorities are precomputed into two dense ``V x V`` int32 lookup
   matrices (``pair_rank`` and ``pair_merge``), so a pair lookup inside the
   loop is a single array index instead of a hash-map probe.
3. The merge loop itself is compiled with ``numba.njit`` — no Python objects,
   no interpreter overhead in the inner loops.

Because Numba operates on numeric arrays, the string-level work (regex
word-splitting and UTF-8 byte encoding) stays in Python; only the numeric
merge loop is JIT-compiled.
"""

from collections import Counter

import numpy as np
from numba import njit, int32, int64
from tqdm.auto import tqdm

from src.tokenizer.bpe_tokenizer import ByteLevelBPETokenizer, WHITESPACE_SPLITTER, bytes_to_unicode


# ---------------------------------------------------------------------------
# Numba kernels (pure numeric, no Python objects)
# ---------------------------------------------------------------------------

@njit(int64(int32[:], int32, int32[:, :], int32[:, :]), cache=True)
def _bpe_merge_count(word, n, pair_rank, pair_merge):
    """Run BPE merges on one word (in place) and return its final length.

    Args:
        word: int32 buffer holding the base token ids of a single word.
        n: number of valid tokens currently in ``word``.
        pair_rank: ``pair_rank[a, b]`` is the merge rank of pair (a, b),
            or -1 if that pair is never merged. Lower rank == higher priority.
        pair_merge: ``pair_merge[a, b]`` is the id of the token produced by
            merging (a, b), valid only where ``pair_rank[a, b] != -1``.

    Returns:
        The number of tokens remaining after all possible merges.
    """
    while n >= 2:
        # Find the highest-priority (lowest-rank) adjacent pair.
        best_rank = np.iinfo(np.int32).max
        best_i = -1
        for i in range(n - 1):
            r = pair_rank[word[i], word[i + 1]]
            if r != -1 and r < best_rank:
                best_rank = r
                best_i = i

        if best_i == -1:
            break  # nothing left to merge

        # Merge at best_i, then shift the tail left by one (in place).
        word[best_i] = pair_merge[word[best_i], word[best_i + 1]]
        for j in range(best_i + 1, n - 1):
            word[j] = word[j + 1]
        n -= 1

    return n


@njit(int32[:](int32[:], int32, int32[:, :], int32[:, :]), cache=True)
def _bpe_merge_ids(word, n, pair_rank, pair_merge):
    """Same as :func:`_bpe_merge_count` but returns the merged token ids."""
    while n >= 2:
        best_rank = np.iinfo(np.int32).max
        best_i = -1
        for i in range(n - 1):
            r = pair_rank[word[i], word[i + 1]]
            if r != -1 and r < best_rank:
                best_rank = r
                best_i = i

        if best_i == -1:
            break

        word[best_i] = pair_merge[word[best_i], word[best_i + 1]]
        for j in range(best_i + 1, n - 1):
            word[j] = word[j + 1]
        n -= 1

    return word[:n].copy()


# ---------------------------------------------------------------------------
# Numba training kernel
# ---------------------------------------------------------------------------
#
# The pure-Python train() in src.tokenizer is O(vocab_size * total_tokens):
# every merge iteration rebuilds ALL words and recomputes ALL pair frequencies
# from scratch (see merge()), and picks the top pair with most_common() which
# scans every pair.
#
# This kernel keeps the exact same greedy algorithm but:
#   * The whole (deduplicated) corpus lives in one flat int32 token array with
#     per-word (start, len) offsets and a per-word frequency — no Python objects.
#   * Pair frequencies live in a dense V x V int64 matrix, so "most frequent
#     pair" is a matrix argmax and every update is O(1).
#   * Each merge rewrites only the affected words IN PLACE and updates pair
#     frequencies INCREMENTALLY (decrement the pairs that break, increment the
#     pairs that form) instead of recomputing the whole corpus.
#
# Overall cost drops to roughly O(number_of_merge_occurrences), which is far
# less than O(vocab_size * total_tokens).

@njit(cache=True)
def _train_kernel(tokens, word_start, word_len, word_freq, base_vocab_size, target_merges):
    """Run BPE merge-building over a flattened corpus.

    Args:
        tokens: flat int32 array, all words' base token ids concatenated.
        word_start: int64[nwords], offset of each word inside ``tokens``.
        word_len: int32[nwords], current length of each word (shrinks on merges).
        word_freq: int64[nwords], corpus frequency of each (unique) word.
        base_vocab_size: number of base tokens (256); new ids start here.
        target_merges: how many merges to perform (== vocab_size - 256 - n_special).

    Returns:
        merge_a, merge_b: int32[target_merges], the merged pair ids per step
            (a merged token id == base_vocab_size + step, so merges are implicit).
        n_done: number of merges actually performed (may be < target if the
            corpus runs out of mergeable pairs).
    """
    vocab_cap = base_vocab_size + target_merges
    nwords = word_start.shape[0]

    # Dense pair-frequency matrix over the full potential id space.
    pair_freq = np.zeros((vocab_cap, vocab_cap), dtype=np.int64)

    # Initial pair statistics.
    for w in range(nwords):
        s = word_start[w]
        L = word_len[w]
        f = word_freq[w]
        for i in range(L - 1):
            pair_freq[tokens[s + i], tokens[s + i + 1]] += f

    merge_a = np.empty(target_merges, dtype=np.int32)
    merge_b = np.empty(target_merges, dtype=np.int32)

    n_done = 0
    for step in range(target_merges):
        # --- find most frequent pair (matrix argmax) ---
        best_a = -1
        best_b = -1
        best_f = 0
        for a in range(vocab_cap):
            row = pair_freq[a]
            for b in range(vocab_cap):
                if row[b] > best_f:
                    best_f = row[b]
                    best_a = a
                    best_b = b

        if best_a == -1:
            break  # nothing left to merge

        new_id = base_vocab_size + step
        merge_a[step] = best_a
        merge_b[step] = best_b
        pair_freq[best_a, best_b] = 0  # this pair is consumed

        # --- apply the merge in every word, updating pair_freq incrementally ---
        for w in range(nwords):
            s = word_start[w]
            L = word_len[w]
            if L < 2:
                continue
            f = word_freq[w]

            i = 0
            write = 0  # in-place compaction index
            while i < L:
                if i < L - 1 and tokens[s + i] == best_a and tokens[s + i + 1] == best_b:
                    # A merge happens at this position. Fix up the neighbouring
                    # pair frequencies relative to the token we are replacing.
                    # left neighbour: (prev, a) -> (prev, new_id)
                    if write > 0:
                        prev = tokens[s + write - 1]
                        pair_freq[prev, best_a] -= f
                        pair_freq[prev, new_id] += f
                    # right neighbour: (b, next) -> (new_id, next)
                    if i + 2 < L:
                        nxt = tokens[s + i + 2]
                        pair_freq[best_b, nxt] -= f
                        pair_freq[new_id, nxt] += f

                    tokens[s + write] = new_id
                    write += 1
                    i += 2
                else:
                    tokens[s + write] = tokens[s + i]
                    write += 1
                    i += 1

            word_len[w] = write

        n_done += 1

    return merge_a, merge_b, n_done


def train(data, vocab_size: int = 1024, special_tokens=None):
    """Train a Byte-Level BPE tokenizer, Numba-accelerated.

    Same greedy algorithm and same return contract as
    :func:`src.tokenizer.train`, but the merge-building loop runs in a compiled
    kernel over flat int arrays (~80x faster on the joke corpus).

    Tie-breaking note: when several pairs share the top frequency, this trainer
    picks the one with the smallest ``(id_a, id_b)`` (deterministic), whereas
    the pure-Python trainer breaks ties by ``Counter`` insertion order. The two
    therefore produce *equivalent* vocabularies that can differ on a handful of
    entries; downstream token counts match to within a small fraction of a
    percent. If you need bit-identical output, keep using the base trainer.

    Memory: the kernel uses a dense ``vocab_size x vocab_size`` int64 pair-
    frequency matrix. That is 8 MB at vocab_size=1024 but grows quadratically
    (~8 GB at 32k), so this implementation targets small/medium vocabularies.

    Args:
        data: iterable of training documents (strings).
        vocab_size: target vocabulary size.
        special_tokens: list of special tokens appended at the end of the vocab.

    Returns:
        (vocab, merges): ``vocab`` maps string token -> id; ``merges`` is the
        ordered list of (str, str) merge pairs. Identical shape to the base
        implementation.
    """
    if vocab_size < 256:
        raise ValueError("Vocab size can't be less than 256")
    # Dense V x V int64 matrix; guard against accidentally huge allocations.
    matrix_gb = vocab_size * vocab_size * 8 / 1e9
    if matrix_gb > 4:
        raise ValueError(
            f"vocab_size={vocab_size} needs a {matrix_gb:.1f} GB pair-frequency "
            f"matrix; this dense-matrix trainer targets small/medium vocabs. "
            f"Use the pure-Python src.tokenizer.train for very large vocabularies."
        )
    if special_tokens is None:
        special_tokens = []

    id2token = bytes_to_unicode()          # id -> unicode char (base 256)
    base_vocab_size = len(id2token)        # 256

    # --- 1. Build the deduplicated word corpus (string side, in Python) ---
    # Same as the base trainer: split into words and count unique byte-tuples.
    words_by_bytes = Counter()
    for sample in tqdm(data, desc="[train fast tokenizer] Loading data:"):
        for word in WHITESPACE_SPLITTER.findall(sample.strip()):
            bt = word.encode("utf-8")
            if bt:
                words_by_bytes[bt] += 1

    nwords = len(words_by_bytes)
    total_len = sum(len(bt) for bt in words_by_bytes)

    # --- 2. Flatten into int arrays for the kernel ---
    # Base token id for a byte value b is simply b (bytes_to_unicode keys are
    # 0..255 and map to ids 0..255 in id2token), so tokens ARE the raw bytes.
    tokens = np.empty(total_len, dtype=np.int32)
    word_start = np.empty(nwords, dtype=np.int64)
    word_len = np.empty(nwords, dtype=np.int32)
    word_freq = np.empty(nwords, dtype=np.int64)

    pos = 0
    for w, (bt, freq) in enumerate(words_by_bytes.items()):
        word_start[w] = pos
        word_len[w] = len(bt)
        word_freq[w] = freq
        for b in bt:
            tokens[pos] = b
            pos += 1

    # --- 3. Run the compiled merge-building kernel ---
    target_merges = vocab_size - base_vocab_size - len(special_tokens)
    if target_merges < 0:
        raise ValueError("vocab_size too small for the given special tokens")

    merge_a, merge_b, n_done = _train_kernel(
        tokens, word_start, word_len, word_freq, base_vocab_size, target_merges
    )
    if n_done < target_merges:
        print("Not enough data to fulfil vocabulary")

    # --- 4. Reconstruct string vocab and merges from the id-space merges ---
    # id_to_str[i] is the concatenated string for token id i. Base ids map to
    # their unicode chars; each merge appends one new token string.
    id_to_str = {i: id2token[i] for i in range(base_vocab_size)}
    merges = []
    for step in range(n_done):
        a = int(merge_a[step])
        b = int(merge_b[step])
        a_str = id_to_str[a]
        b_str = id_to_str[b]
        new_str = a_str + b_str
        id_to_str[base_vocab_size + step] = new_str
        merges.append((a_str, b_str))

    # Assemble the final string vocab in id order, then append special tokens.
    ordered = bytes_to_unicode()  # {id: char} for base
    for step in range(n_done):
        ordered[base_vocab_size + step] = id_to_str[base_vocab_size + step]
    for special_token in special_tokens:
        ordered[len(ordered)] = special_token

    vocab = {v: k for k, v in ordered.items()}
    return vocab, merges


class FastByteLevelBPETokenizer:
    """Numba-accelerated Byte-Level BPE tokenizer.

    Build it from an existing tokenizer with :meth:`from_tokenizer`, or load
    directly from disk / the Hub with :meth:`from_pretrained` (same layout as
    :class:`src.tokenizer.ByteLevelBPETokenizer`).
    """

    def __init__(self, vocab, merges, eos_token="[EOS]"):
        if eos_token not in vocab:
            raise ValueError("There is no EOS token in vocab")

        self.token2id = vocab
        self.id2token = {v: k for k, v in vocab.items()}
        self.eos_token = eos_token
        self.eos_token_id = vocab[eos_token]
        self.merges = merges

        # Image special tokens (populated on demand by add_image_tokens()).
        self.image_start_token: str | None = None
        self.image_token: str | None = None
        self.image_end_token: str | None = None
        self.image_start_token_id: int | None = None
        self.image_token_id: int | None = None
        self.image_end_token_id: int | None = None

        self._byte_encoder = bytes_to_unicode()  # int byte -> unicode char
        self._byte_decoder = {v: k for k, v in self._byte_encoder.items()}

        vocab_size = len(vocab)

        # byte value (0..255) -> base token id
        self._byte_to_id = np.full(256, -1, dtype=np.int32)
        for b, ch in self._byte_encoder.items():
            self._byte_to_id[b] = vocab[ch]

        # Dense lookup matrices for O(1) pair queries inside the njit loop.
        # -1 means "this pair is never merged".
        self._pair_rank = np.full((vocab_size, vocab_size), -1, dtype=np.int32)
        self._pair_merge = np.full((vocab_size, vocab_size), -1, dtype=np.int32)
        for rank, (a_str, b_str) in enumerate(merges):
            a = vocab[a_str]
            b = vocab[b_str]
            self._pair_rank[a, b] = rank
            self._pair_merge[a, b] = vocab[a_str + b_str]

        # Reusable scratch buffer for a single word's token ids. UTF-8 encoded
        # words rarely exceed this; we grow it on demand if one does.
        self._buf = np.empty(1024, dtype=np.int32)

    # -- construction ------------------------------------------------------

    @classmethod
    def from_tokenizer(cls, tokenizer: ByteLevelBPETokenizer):
        """Build from an already-loaded pure-Python tokenizer."""
        return cls(tokenizer.token2id, tokenizer.merges, tokenizer.eos_token)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *, token=None, **model_kwargs):
        base = ByteLevelBPETokenizer.from_pretrained(
            pretrained_model_name_or_path, token=token, **model_kwargs
        )
        return cls.from_tokenizer(base)

    def add_image_tokens(
        self,
        image_start_token: str = "[IMG_START]",
        image_token: str = "[IMG]",
        image_end_token: str = "[IMG_END]",
    ) -> "FastByteLevelBPETokenizer":
        """Append image special tokens to the vocab in place; returns self.

        These ids are never emitted by encode() (image tokens don't participate
        in BPE merges), so the njit pair matrices need no resizing — they only
        need to exist in id2token so decode() can skip them.
        """
        for tok in (image_start_token, image_token, image_end_token):
            if tok not in self.token2id:
                new_id = len(self.token2id)
                self.token2id[tok] = new_id
                self.id2token[new_id] = tok

        self.image_start_token = image_start_token
        self.image_token = image_token
        self.image_end_token = image_end_token
        self.image_start_token_id = self.token2id[image_start_token]
        self.image_token_id = self.token2id[image_token]
        self.image_end_token_id = self.token2id[image_end_token]
        return self

    # -- internals ---------------------------------------------------------

    def _word_to_base_ids(self, word: str):
        """Encode a word to its base (byte-level) token ids in the scratch buffer.

        Returns the number of base tokens written to ``self._buf``.
        """
        bt = word.encode("utf-8")
        n = len(bt)
        if n > self._buf.shape[0]:
            self._buf = np.empty(n, dtype=np.int32)
        buf = self._buf
        for i in range(n):
            buf[i] = self._byte_to_id[bt[i]]
        return n

    # -- public API --------------------------------------------------------

    def count(self, text: str, add_eos_token: bool = True) -> int:
        """Return the number of tokens in ``text`` without materializing ids.

        This is the fast path for dataset statistics (see the notebook): it
        avoids building any per-word id lists in Python.
        """
        total = 0
        for w in WHITESPACE_SPLITTER.findall(text):
            n = self._word_to_base_ids(w)
            total += _bpe_merge_count(self._buf, n, self._pair_rank, self._pair_merge)
        if add_eos_token:
            total += 1
        return total

    def encode(self, text: str, add_eos_token: bool = True) -> list[int]:
        """Convert ``text`` to a list of token ids (matches the base tokenizer)."""
        ids: list[int] = []
        for w in WHITESPACE_SPLITTER.findall(text):
            n = self._word_to_base_ids(w)
            merged = _bpe_merge_ids(self._buf, n, self._pair_rank, self._pair_merge)
            ids.extend(int(x) for x in merged)
        if add_eos_token:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, idx: list[int]) -> str:
        """Convert token ids back to text (same logic as the base tokenizer)."""
        out_parts: list[str] = []
        byte_buf: list[int] = []
        skip_ids = {self.eos_token_id}
        skip_ids.update(
            tid for tid in (self.image_start_token_id, self.image_token_id, self.image_end_token_id)
            if tid is not None
        )
        for tid in idx:
            if tid in skip_ids:
                continue
            tok = self.id2token[tid]
            all_base = all(ch in self._byte_decoder for ch in tok)
            if all_base:
                byte_buf.extend(self._byte_decoder[ch] for ch in tok)
            else:
                if byte_buf:
                    out_parts.append(bytes(byte_buf).decode("utf-8", errors="replace"))
                    byte_buf = []
                out_parts.append(tok)
        if byte_buf:
            out_parts.append(bytes(byte_buf).decode("utf-8", errors="replace"))
        return "".join(out_parts)

    def warmup(self):
        """Trigger JIT compilation of the kernels once, up front."""
        self.count("прогрев", add_eos_token=False)
        self.encode("прогрев", add_eos_token=False)
        return self
