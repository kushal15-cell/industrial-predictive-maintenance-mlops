import joblib
import pandas as pd

from sklearn.model_selection import train_test_split

from src.config import ROOT_DIR, load_params


MODEL_PATH = ROOT_DIR / "models" / "random_forest_rul_model.pkl"
MONITORING_DIR = ROOT_DIR / "monitoring"


def load_model():
    """
    Load the currently deployed RUL model.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    print("Model loaded successfully")

    return model


def load_data():
    """
    Load the processed CMAPSS FD001 dataset.
    """

    params = load_params()

    processed_path = ROOT_DIR / params["data"]["processed_path"]

    data_path = processed_path / "train_fd001_processed.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Processed data not found: {data_path}"
        )

    df = pd.read_csv(data_path)

    print(f"Data loaded: {df.shape}")

    return df


def recreate_engine_split(df):
    """
    Recreate exactly the same engine-level split
    that was used during model training.

    Because train_model.py uses:

        train_test_split(
            units,
            test_size=params["model"]["test_size"],
            random_state=params["model"]["random_state"]
        )

    using the same parameters here gives us the
    same train and test engines.
    """

    params = load_params()

    units = df["unit_number"].unique()

    train_units, production_units = train_test_split(
        units,
        test_size=params["model"]["test_size"],
        random_state=params["model"]["random_state"],
    )

    reference_df = df[
        df["unit_number"].isin(train_units)
    ].copy()

    production_df = df[
        df["unit_number"].isin(production_units)
    ].copy()

    reference_df = reference_df.sort_values(
        ["unit_number", "time_in_cycles"]
    ).reset_index(drop=True)

    production_df = production_df.sort_values(
        ["unit_number", "time_in_cycles"]
    ).reset_index(drop=True)

    return (
        reference_df,
        production_df,
        train_units,
        production_units,
    )


def add_predictions(model, production_df):
    """
    Run the deployed model on unseen production engines
    and attach predictions to the production dataset.
    """

    feature_columns = [
        column
        for column in production_df.columns
        if column not in ["RUL", "unit_number"]
    ]

    X_production = production_df[feature_columns]

    predictions = model.predict(X_production)

    production_df = production_df.copy()

    production_df["predicted_RUL"] = predictions

    production_df["prediction_error"] = (
        production_df["RUL"]
        - production_df["predicted_RUL"]
    )

    production_df["absolute_error"] = (
        production_df["prediction_error"].abs()
    )

    return production_df


def add_operational_status(production_df):
    """
    Convert predicted RUL into an operational status.

    > 30 cycles       -> NORMAL
    15 to 30 cycles   -> WARNING
    < 15 cycles       -> CRITICAL
    """

    production_df = production_df.copy()

    def get_status(predicted_rul):

        if predicted_rul > 30:
            return "NORMAL"

        elif predicted_rul >= 15:
            return "WARNING"

        else:
            return "CRITICAL"

    production_df["status"] = (
        production_df["predicted_RUL"]
        .apply(get_status)
    )

    return production_df


def save_monitoring_data(
    reference_df,
    production_df,
):
    """
    Save monitoring datasets.

    reference_data.csv:
        Engines used during model training.

    production_data.csv:
        Completely unseen engines with:
        - actual RUL
        - predicted RUL
        - errors
        - operational status
    """

    MONITORING_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    reference_path = (
        MONITORING_DIR /
        "reference_data.csv"
    )

    production_path = (
        MONITORING_DIR /
        "production_data.csv"
    )

    reference_df.to_csv(
        reference_path,
        index=False
    )

    production_df.to_csv(
        production_path,
        index=False
    )

    print()
    print("Monitoring datasets saved")
    print("-" * 60)

    print(
        f"Reference data: {reference_path}"
    )

    print(
        f"Production data: {production_path}"
    )


def print_summary(
    reference_df,
    production_df,
    train_units,
    production_units,
):
    """
    Print monitoring dataset summary.
    """

    print()
    print("=" * 60)
    print("MONITORING DATASET SUMMARY")
    print("=" * 60)

    print(
        f"Reference engines: {len(train_units)}"
    )

    print(
        f"Production engines: {len(production_units)}"
    )

    print(
        f"Reference rows: {len(reference_df)}"
    )

    print(
        f"Production rows: {len(production_df)}"
    )

    print()
    print("Production status distribution:")

    print(
        production_df["status"]
        .value_counts()
    )

    print()
    print("Prediction examples:")

    print(
        production_df[
            [
                "unit_number",
                "time_in_cycles",
                "RUL",
                "predicted_RUL",
                "absolute_error",
                "status",
            ]
        ].head(10)
    )


def main():

    print("=" * 60)
    print("IPMIP PRODUCTION MONITOR")
    print("=" * 60)

    model = load_model()

    df = load_data()

    (
        reference_df,
        production_df,
        train_units,
        production_units,
    ) = recreate_engine_split(df)

    production_df = add_predictions(
        model,
        production_df,
    )

    production_df = add_operational_status(
        production_df
    )

    save_monitoring_data(
        reference_df,
        production_df,
    )

    print_summary(
        reference_df,
        production_df,
        train_units,
        production_units,
    )

    print()
    print(
        "Production monitoring data generation completed."
    )


if __name__ == "__main__":
    main()