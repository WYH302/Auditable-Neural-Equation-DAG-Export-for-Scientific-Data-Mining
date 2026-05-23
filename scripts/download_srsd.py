from pathlib import Path

from datasets import load_dataset


OUT_DIR = Path("data/srsd_feynman_hard")
DATASET_NAME = "yoshitomo-matsubara/srsd-feynman_hard"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(DATASET_NAME)
    print(dataset)

    for split_name, split_data in dataset.items():
        split_path = OUT_DIR / f"{split_name}.csv"
        split_data.to_pandas().to_csv(split_path, index=False)
        print(f"Saved {split_name} to {split_path}")


if __name__ == "__main__":
    main()
