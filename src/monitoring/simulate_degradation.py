import pandas as pd

from src.config import ROOT_DIR, load_params


def main():
    params = load_params()

    production_path = (
        ROOT_DIR
        / params["monitoring"]["production_path"]
    )

    backup_path = (
        ROOT_DIR
        / "monitoring"
        / "production_data_backup.csv"
    )

    df = pd.read_csv(production_path)

    # Save original healthy production dataset once
    if not backup_path.exists():
        df.to_csv(backup_path, index=False)
        print(f"Backup saved to: {backup_path}")

    degraded = df.copy()

    # -----------------------------
    # Simulate sensor/data drift
    # -----------------------------
    drift_features = [
        "sensor_2",
        "sensor_3",
        "sensor_4",
        "sensor_11",
    ]

    for feature in drift_features:
        if feature in degraded.columns:
            degraded[feature] = (
                degraded[feature] * 1.20
            )

    # -----------------------------
    # Simulate prediction degradation
    # -----------------------------
    if "predicted_RUL" not in degraded.columns:
        raise ValueError(
            "predicted_RUL column missing. "
            "Run src.monitoring.monitor first."
        )

    degraded["predicted_RUL"] = (
        degraded["predicted_RUL"] + 25
    )

    degraded["prediction_error"] = (
        degraded["RUL"]
        - degraded["predicted_RUL"]
    )

    degraded["absolute_error"] = (
        degraded["prediction_error"].abs()
    )

    degraded.to_csv(
        production_path,
        index=False
    )

    print()
    print("Degraded production data generated.")
    print(f"Modified features: {drift_features}")
    print("Predicted RUL shifted by +25 cycles.")
    print()
    print(f"Saved to: {production_path}")


if __name__ == "__main__":
    main()