from pathlib import Path
from datasets import Dataset, load_dataset
from tqdm import tqdm


TARGET_SAMPLES_COUNT = 300_000
REPO_ID = "NotEvilAI/ruwiki-pretrain-20260102"

if __name__ == "__main__":
    dataset = load_dataset(
        REPO_ID,
        split="train",
        streaming=True,
    )

    samples = []

    for sample in tqdm(dataset, total=TARGET_SAMPLES_COUNT, desc="Загрузка"):
        samples.append(sample)

        if len(samples) >= TARGET_SAMPLES_COUNT:
            break

    print(f"Загружено примеров: {len(samples)}")

    dataset = Dataset.from_list(samples)
    save_dir = Path(f"C:/vscode_projects/complete/LLM/datasets/NotEvilAI_ruwiki_{TARGET_SAMPLES_COUNT // 1000}K_samples")
    dataset.save_to_disk(str(save_dir))