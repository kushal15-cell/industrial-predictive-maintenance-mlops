from pathlib import Path
from src.data.ingestion import load_cmapss_fd001


raw_path = Path("data/raw/CMAPSSData")

df = load_cmapss_fd001(raw_path)

print("Data loaded successfully")
print(df.head())
print(df.shape)
print(df.columns)