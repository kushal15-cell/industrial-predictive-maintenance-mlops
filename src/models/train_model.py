from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.config import ROOT_DIR, load_params


def train_model():
    params = load_params()

    processed_path = ROOT_DIR / params["data"]["processed_path"]
    data_path = processed_path / "train_fd001_processed.csv"

    model_dir = ROOT_DIR / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)

    target = "RUL"

    drop_columns = ["RUL", "unit_number"]

    X = df.drop(columns=drop_columns)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=params["model"]["test_size"],
        random_state=params["model"]["random_state"],
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=params["model"]["random_state"],
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)

    print("Model training completed")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")

    model_path = model_dir / "random_forest_rul_model.pkl"
    joblib.dump(model, model_path)

    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    train_model()