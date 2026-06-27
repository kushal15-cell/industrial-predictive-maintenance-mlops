import joblib
import pandas as pd

from src.config import ROOT_DIR, load_params


def load_model():
    model_path = ROOT_DIR / "models" / "random_forest_rul_model.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run `dvc repro` or train the model first."
        )

    model = joblib.load(model_path)
    return model


def predict_sample(sample_data: pd.DataFrame):
    model = load_model()
    predictions = model.predict(sample_data)
    return predictions


if __name__ == "__main__":
    params = load_params()

    data_path = ROOT_DIR / params["data"]["processed_path"] / "train_fd001_processed.csv"

    df = pd.read_csv(data_path)

    drop_columns = ["RUL", "unit_number"]
    X = df.drop(columns=drop_columns)

    sample = X.head(5)

    predictions = predict_sample(sample)

    print("Sample predictions:")
    for i, pred in enumerate(predictions, start=1):
        print(f"Sample {i}: Predicted RUL = {pred:.2f}")