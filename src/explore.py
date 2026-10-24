"""Quick look at every raw file: python src/explore.py"""
import pandas as pd
from config import RAW

def load(path):
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)

if __name__ == "__main__":
    files = sorted(p for p in RAW.iterdir() if p.suffix in {".csv", ".parquet"})
    if not files:
        print(f"No csv/parquet files found in {RAW}. Download the data there first.")
    for p in files:
        df = load(p)
        print("=" * 80)
        print(p.name, df.shape)
        print(df.dtypes.to_string())
        print(df.head(3).to_string())
        print("null share (top 10):")
        print(df.isna().mean().sort_values(ascending=False).head(10).to_string())
