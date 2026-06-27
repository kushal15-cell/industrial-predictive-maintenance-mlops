import time
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from src.config import ROOT_DIR, load_params


def evaluate_regression_model(model, X_train, X_test, y_train, y_test):
    start_time = time.time()

    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)

    return {
        "mae": mae,
        "rmse": rmse,
        "r2_score": r2,
        "training_time_seconds": training_time,
    }


def compare_models():
    params = load_params()

    data_path = ROOT_DIR / params["data"]["processed_path"] / "train_fd001_processed.csv"
    reports_dir = ROOT_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

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

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(random_state=params["model"]["random_state"]),
        "Decision Tree": DecisionTreeRegressor(
            random_state=params["model"]["random_state"],
            max_depth=10,
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=params["model"]["n_estimators"],
            max_depth=params["model"]["max_depth"],
            random_state=params["model"]["random_state"],
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            random_state=params["model"]["random_state"],
        ),
    }

    results = []

    for model_name, model in models.items():
        print(f"Training {model_name}...")

        metrics = evaluate_regression_model(
            model,
            X_train,
            X_test,
            y_train,
            y_test,
        )

        results.append(
            {
                "model_name": model_name,
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2_score": metrics["r2_score"],
                "training_time_seconds": metrics["training_time_seconds"],
            }
        )

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by=["mae", "rmse"],
        ascending=[True, True],
    )

    output_path = reports_dir / "model_comparison.csv"
    results_df.to_csv(output_path, index=False)

    print("\nModel comparison completed.")
    print(results_df)
    print(f"\nModel comparison saved to: {output_path}")


if __name__ == "__main__":
    compare_models()