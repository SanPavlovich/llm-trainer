import json
import datetime
import os
from pathlib import Path
from datasets import load_dataset

from src.dataset import TextDataset, create_dataloader
from src.tokenizer import ByteLevelBPETokenizer, train
from src.model import TransformerForCausalLM
from src.trainer import Trainer
from src.schemas import TokenizerConfig, TransformerConfig, TrainerConfig


MODEL_CONFIGS = {
    "nano": TransformerConfig(n_layer=3, n_head=4, n_kv_head=2, hidden_dim=96, intermediate_dim=256),
    "mini": TransformerConfig(n_layer=6, n_head=6, n_kv_head=3, hidden_dim=384, intermediate_dim=1024),
    "small": TransformerConfig(n_layer=12, n_head=12, n_kv_head=6, hidden_dim=768, intermediate_dim=2048, use_rope=True, rope_theta=10000.0),
}
MAX_SEQ_LEN = 128
BATCH_SIZE = 16
VOCAB_SIZE = 1024
VALID_TEXTS = [
    "Заходит в бар",
    "Штирлиц пришел домой",
]
SEED = 42
RUN_NAME = "llm_small"

def save_tokenizer_files(
    path: Path,
    vocab: dict[str, int], 
    merges: list[tuple[str, str]]
) -> None:
    with open(path / "vocabulary.json", "w") as f:
        json.dump(vocab, f, indent=2)
    with open(path / "merges.json", "w") as f:
        json.dump({"merges": merges}, f)
        
def save_configs(
    path: Path, 
    tokenizer_config: TokenizerConfig, 
    model_config: TransformerConfig,
    train_config: TrainerConfig, 
) -> None:
    with open(path / "tokenizer_config.json", "w") as f:
        json.dump(tokenizer_config.model_dump(), f, indent=4)
    with open(path / "train_config.json", "w") as f:
        json.dump(train_config.model_dump(), f, indent=4)
    with open(path / "model_config.json", "w") as f:
        json.dump(model_config.model_dump(), f, indent=4)


if __name__ == "__main__":
    time_str_format = datetime.datetime.now().strftime('%m-%d-%Y--%H-%M-%S')
    log_dir = Path(__file__).resolve().parent / "runs" / f"{RUN_NAME}_{time_str_format}"
    cache_dir = Path(__file__).resolve().parent / "cache"
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    if not os.path.exists(cache_dir):
        os.mkdir(log_dir)

    dataset = load_dataset("json", data_files="hf://datasets/IgorVolochay/russian_jokes/dataset.json")
    dataset = dataset["train"].train_test_split(test_size=0.1, seed=SEED)
    
    tokenizer_config = TokenizerConfig(vocab_size=VOCAB_SIZE)
    if os.path.exists(cache_dir / "vocabulary.json") and os.path.exists(cache_dir / "vocabulary.json"):
        tokenizer = ByteLevelBPETokenizer.from_pretrained(cache_dir)
    else:
        vocab, merges = train(data=dataset["train"]["jokes"], **tokenizer_config.model_dump())
        save_tokenizer_files(cache_dir, vocab, merges)
        tokenizer = ByteLevelBPETokenizer(vocab, merges)

    train_config = TrainerConfig(max_seq_len=MAX_SEQ_LEN, batch_size=BATCH_SIZE)
    
    train_dataset = TextDataset(dataset["train"]["jokes"], tokenizer)
    train_dataloader = create_dataloader(
        train_dataset, tokenizer.eos_token_id, max_seq_len=train_config.max_seq_len, batch_size=train_config.batch_size, is_train=True
    )
    test_dataset = TextDataset(dataset["test"]["jokes"], tokenizer)
    test_dataloader = create_dataloader(
        test_dataset, tokenizer.eos_token_id, max_seq_len=train_config.max_seq_len, batch_size=train_config.batch_size, is_train=False
    )

    model_config = MODEL_CONFIGS["small"]
    model = TransformerForCausalLM(model_config)
    
    trainer = Trainer(
        tokenizer=tokenizer,
        model=model,
        train_config=train_config,
        valid_texts=VALID_TEXTS,
        log_dir=log_dir,
    )
    save_configs(log_dir, tokenizer_config, model_config, train_config)

    trainer.train(train_dataloader, test_dataloader)