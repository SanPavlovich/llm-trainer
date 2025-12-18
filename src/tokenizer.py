import json
import os
import regex
from collections import Counter
from pathlib import Path
from functools import lru_cache, partial
from tqdm.auto import tqdm, trange
from huggingface_hub import HfApi, PyTorchModelHubMixin, interpreter_login, snapshot_download
from huggingface_hub.utils import SoftTemporaryDirectory


WHITESPACE_SPLITTER = regex.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


def bytes_to_unicode() -> dict[int, str]:
    """The original dictionary consists of 256 bytes and their corresponding Unicode characters.
    For example, chr(33) is '!'. However, not all bytes have a visually appealing representation,
    so such characters are skipped and replaced with the first available ones, i.e. shifted by 256.
    """
    initial_bytes = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    initial_chars = [chr(it) for it in initial_bytes]
    n = 0
    for byte in range(2**8):
        if byte not in initial_bytes:
            initial_bytes.append(byte)
            initial_chars.append(chr(2**8 + n))
            n += 1
    return dict(sorted(zip(initial_bytes, initial_chars)))


def merge(merge_pair: tuple[str, str], pair_frequences: Counter[tuple[str, str]], words_by_tokens: Counter[tuple[str]]):
    """Merges a given pair of tokens and update corresponding stats

    Args:
        merge_pair: The pair of tokens to be merged.
        pair_frequences: A counter tracking the frequency of token pairs in the dataset.
        words_by_tokens: A counter mapping tokenized words to their frequencies.

    Returns:
        Updated pair frequences and word tokenization w.r.t. to new token.
    """
    a, b = merge_pair
    n_tok = a + b

    # n_words: dict[tuple(str), int] - обновленная версия words_by_tokens со статистиками после слияния токенов merge_pair.
    # хранит в себе частоту встречаемости токенизированных слов: 
    # word_i: встречается 'x' раз --> (token_1, ..., token_n): встречается 'x' раз
    n_words = Counter() 
    for toks, freq in words_by_tokens.items():
        i = 0
        n = len(toks)
        merged = []
        while i < n:
            if i < n - 1 and toks[i] == a and toks[i + 1] == b:
                merged.append(n_tok)
                i += 2
            else:
                merged.append(toks[i])
                i += 1
        n_words[tuple(merged)] += freq
    
    # n_freqs - обновленная версия pair_frequences попарной совстречаемости токенов, пересчитанная по n_words!
    # Из него будем потом снова брать most_common и делать merge
    n_freqs = Counter() 
    for toks, freq in n_words.items():
        # в какой-то момент внутри слова все токены могут смерджиться  в один
        # поэтому мы пропускаем такой случай при подсчете статистики
        # далее нас будут интересовать только попарные статистики по другим, не смердженным в 1 токен, словам
        if len(toks) < 2:
            continue
        for i in range(len(toks) - 1):
            n_freqs[(toks[i], toks[i + 1])] += freq

    return n_freqs, n_words


def train(data: list[str], vocab_size: int = 1024, special_tokens: list[str] = None):
    """Train BPE tokenizer on passed data

    Args:
        data: List of train documents
        vocab_size: Size of target vocabulary
        special_tokens: List of special tokens to add into vocabulary
    Returns:
        vocabulary: mapping from string token to id
        merges: list of merges, each one is tuple of string tokens
    """
    if vocab_size < 256:
        raise ValueError("Vocab size can't be less than 256")
    if special_tokens is None:
        special_tokens = []

    # 1. Initialize vocabulary (using inverse one during training)
    id2token = bytes_to_unicode()
    merges = []

    # 2. Load data
    words_by_tokens = Counter()
    for sample in tqdm(data, desc="Loading data"):
        # 2.1 Split into words
        words = WHITESPACE_SPLITTER.findall(sample.strip())
        for word in words:
            # 2.2 Tokenize with base vocabulary
            bt = word.encode("utf-8")
            toks = tuple(id2token[b] for b in bt)
            if toks:
                words_by_tokens[toks] += 1

    # 3. Calculate statistic of token's pairs
    pair_frequences = Counter()
    for toks, freq in words_by_tokens.items():
        if len(toks) < 2:
            continue
        for i in range(len(toks) - 1):
            pair_frequences[(toks[i], toks[i + 1])] += freq

    # 4. Build vocabulary
    pbar = trange(vocab_size, desc="Building vocabulary", initial=len(id2token) + len(special_tokens))
    while len(id2token) < vocab_size - len(special_tokens):
        if len(pair_frequences) == 0:
            # в предельном случае все токены во всех словах могут смерджиться в 1 токен
            # в функции merge() при пересчете статистик мы не учитываем такие слова
            # поэтому если все слова смерджаться в 1 токен, то мы не посчитаем ни одной попарной статистики и длина pair_frequences будет нулевой
            # Значит нам нечего больше мерджить и нужно заканчивать обучение токенайзера.
            print("Not enough data to fulfil vocabulary")
            break

        # 4.1 Find the most frequent pair and create new token
        top_pair = pair_frequences.most_common(1)[0][0]
        new_token = top_pair[0] + top_pair[1]
        del pair_frequences[top_pair]

        # 4.2 Add to vocabulary
        if new_token in id2token.values():
            continue
        id2token[len(id2token)] = new_token
        merges.append(top_pair)

        # 4.3 Update stats and merge the top pair in all tokens
        pair_frequences, words_by_tokens = merge(top_pair, pair_frequences, words_by_tokens)

        pbar.update()
    pbar.close()

    # 5. Add special tokens
    for special_token in special_tokens:
        id2token[len(id2token)] = special_token

    return {v: k for k, v in id2token.items()}, merges


class ByteLevelBPETokenizer:
    def __init__(self, vocab: dict[str, int], merges: list[tuple[str, str]], eos_token: str = "[EOS]"):
        """Byte-Level BPE Tokenizer

        Args:
            vocab: mapping from string token to id
            merges: list of merges in prioritized order
            eos_token: string representation of EOS token
        """
        super().__init__()
        if eos_token not in vocab:
            raise ValueError("There is no EOS token in vocab")
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.token2id = vocab
        self.id2token = {v: k for k, v in self.token2id.items()}
        self.eos_token = eos_token
        self.eos_token_id = self.token2id[eos_token]

        # The closer the pair is to the beginning, the higher the rank
        self.merges = merges
        self.bpe_ranks = {pair: i for i, pair in enumerate(merges)}

    @lru_cache
    def bpe(self, word: tuple[str]) -> tuple[str]:
        """Process word into tokenized representation.
        Word is a tuple of base tokens, i.e. bytes.

        Under the hood:
        1. Tracks the set of token pairs, bi-grams
        2. While possible, replaces the highest-ranking pair with its union

        Args:
            word: list of base string tokens
        Return:
            list of BPE tokens
        """
        if len(word) < 2:
            return word
        pairs = set((word[i], word[i + 1]) for i in range(len(word) - 1))

        # проходимся по всем попарным сочетаниям (word[i], word[i + 1]) внутри word
        # делаем наименее высокоранговое слияние (то есть более базовое) из всех возможных
        # пересчитываем новые попарные сочетания. 
        # Повторяем процесс, пока пары токенов не закончатся или когда не сможем больше смерджить 
        # в self.merges закончатся такие попарные сочетания и нужно будет выходить из функции
        while pairs:
            best_pair = None
            best_rank = float("inf")
            for p in pairs:
                r = self.bpe_ranks.get(p)
                if r is not None and r < best_rank:
                    best_rank = r
                    best_pair = p
            if best_pair is None:
                break
            a, b = best_pair
            ab = a + b
            i = 0
            new_word = []
            n = len(word)
            while i < n:
                if i < n - 1 and word[i] == a and word[i + 1] == b:
                    new_word.append(ab)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)

            if len(word) < 2:
                break
            pairs = set((word[i], word[i + 1]) for i in range(len(word) - 1))

        return word

    def encode(self, text: str, add_eos_token: bool = True) -> list[int]:
        """Convert string to list of token ids.

        Args:
            text: input string, may contain multiple words
            add_eos_token: whether to add eos token id at the end
        Return:
            list of ints, ids of tokenized text
        """
        words = WHITESPACE_SPLITTER.findall(text)
        ids: list[int] = []
        for w in words:
            bt = w.encode("utf-8")
            base_tokens = tuple(self.byte_encoder[b] for b in bt) # получаем базовый набор токенов из байт
            bpe_tokens = self.bpe(base_tokens) # пытаемся по максимуму их смерджить на основе наших self.merges, полученных после обучения
            for t in bpe_tokens:
                ids.append(self.token2id[t]) # превращаем смердженные токены в int на основе vocab, полученного после обучения (self.token2id = vocab)
        if add_eos_token:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, idx: list[int]) -> str:
        """Convert list of tokens' ids to text, opposite to encode method

        Args:
            idx: list of tokens' ids
        Return:
            string, decoded text
        """
        out_parts: list[str] = []
        byte_buf: list[int] = []
        
        for tid in idx:
            if tid == self.eos_token_id:
                continue
            tok = self.id2token[tid]
            all_base = True
            for ch in tok:
                if ch not in self.byte_decoder:
                    all_base = False
                    break

            if all_base:
                for ch in tok:
                    byte_buf.append(self.byte_decoder[ch])
            else:
                if byte_buf:
                    out_parts.append(bytes(byte_buf).decode("utf-8", errors="replace"))
                    byte_buf = []
                out_parts.append(tok)

        if byte_buf:
            out_parts.append(bytes(byte_buf).decode("utf-8", errors="replace"))

        return "".join(out_parts)


    def push_to_hub(self, repo_id, *, private=None, token=None):
        api = HfApi()
        repo_id = api.create_repo(repo_id=repo_id, token=token, private=private, exist_ok=True).repo_id

        # Push the files to the repo in a single commit
        with SoftTemporaryDirectory() as tmp:
            save_directory = Path(tmp) / repo_id
            save_directory.mkdir(parents=True)
            with open(save_directory / "vocabulary.json", "w") as f_out:
                print(json.dumps(self.token2id, indent=2), file=f_out)
            with open(save_directory / "merges.json", "w") as f_out:
                print(json.dumps({"merges": self.merges}), file=f_out)

            return api.upload_folder(repo_id=repo_id, folder_path=save_directory, token=token)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *, token=None, **model_kwargs):
        if not os.path.isdir(pretrained_model_name_or_path):
            storage_folder = snapshot_download(repo_id=pretrained_model_name_or_path, token=token)
        else:
            storage_folder = pretrained_model_name_or_path
        storage_folder = Path(storage_folder)
        with open(storage_folder / "vocabulary.json", "r") as f_in:
            vocab = json.load(f_in)
        with open(storage_folder / "merges.json", "r") as f_in:
            merges = [tuple(it) for it in json.load(f_in)["merges"]]
        return cls(vocab, merges, **model_kwargs)