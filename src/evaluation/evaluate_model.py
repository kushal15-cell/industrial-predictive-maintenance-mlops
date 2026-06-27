import json

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.config import ROOT_DIR, load_params


def evaluate_model():
    params = load_params()

    data_path = ROOT_DIR / params["data"]["processed_path"] / "train_fd001_processed.csv"
    model_path = ROOT_DIR / "models" / "random_forest_rul_model.pkl"
    reports_dir = ROOT_DIR / "reports"

    reports_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)

    target = "RUL"
    drop_columns = ["RUL", "unit_number"]

    X = df.drop(columns=drop_columns)
    y = df[target]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=params["model"]["test_size"],
        random_state=params["model"]["random_state"],
    )

    model = joblib.load(model_path)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "r2_score": r2,
    }

    metrics_path = reports_dir / "metrics.json"

    with open(metrics_path, "w") as file:
        json.dump(metrics, file, indent=4)

    print("Model evaluation completed")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    evaluate_model()