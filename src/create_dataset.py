from pathlib import Path

from src.data.ingestion import load_cmapss_fd001
from src.data.validation import validate_cmapss_data
from src.features.build_features import add_remaining_useful_life, cap_rul
from src.config import load_params, ROOT_DIR


def create_training_dataset():
    params = load_params()

    raw_path = ROOT_DIR / params["data"]["raw_path"]
    processed_path = ROOT_DIR / params["data"]["processed_path"]

    processed_path.mkdir(parents=True, exist_ok=True)

    df = load_cmapss_fd001(raw_path)

    validate_cmapss_data(df)

    df = add_remaining_useful_life(df)
    df = cap_rul(df, cap_value=params["features"]["rul_clip"])

    output_path = processed_path / "train_fd001_processed.csv"
    df.to_csv(output_path, index=False)

    print(f"Processed dataset saved to: {output_path}")
    print(f"Processed dataset shape: {df.shape}")


if __name__ == "__main__":
    create_training_dataset()