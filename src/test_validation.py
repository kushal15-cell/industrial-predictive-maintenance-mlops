from pathlib import Path

from src.data.ingestion import load_cmapss_fd001
from src.data.validation import validate_cmapss_data


raw_path = Path("data/raw/CMAPSSData")

df = load_cmapss_fd001(raw_path)

print("Data loaded successfully")
print(df.shape)

validate_cmapss_data(df)