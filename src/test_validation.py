from pathlib import Path

from src.data.ingestion import load_cmapss_fd001
from src.data.validation import validate_cmapss_data
from src.features.build_features import add_remaining_useful_life, cap_rul


raw_path = Path("data/raw/CMAPSSData")

df = load_cmapss_fd001(raw_path)

print("Data loaded successfully")
print(df.shape)

validate_cmapss_data(df)

df = add_remaining_useful_life(df)
df = cap_rul(df, cap_value=125)

print("Feature engineering completed")
print(df[["unit_number", "time_in_cycles", "RUL"]].head())
print(df[["unit_number", "time_in_cycles", "RUL"]].tail())
print(df["RUL"].describe())