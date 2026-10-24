"""Validate predictions before upload:
python src/check_submission.py outputs/predictions.csv [expected_rows]"""
import sys
import numpy as np
import pandas as pd

def check(path, expected_rows=40101):
    df = pd.read_csv(path)
    problems = []
    if list(df.columns) != ["TRANSACTION_HASH", "score"]:
        problems.append(f"columns are {list(df.columns)}; expected ['TRANSACTION_HASH', 'score']")
    if len(df) != expected_rows:
        problems.append(f"{len(df)} rows; expected {expected_rows}")
    if df["TRANSACTION_HASH"].duplicated().any():
        problems.append("duplicate TRANSACTION_HASH values")
    if df["TRANSACTION_HASH"].isna().any():
        problems.append("missing TRANSACTION_HASH values")
    s = pd.to_numeric(df["score"], errors="coerce")
    if s.isna().any() or not np.isfinite(s).all():
        problems.append("non-finite or non-numeric scores")
    elif (s < 0).any() or (s > 1).any():
        problems.append("scores outside [0, 1]")
    if s.nunique() < 100:
        problems.append(f"only {s.nunique()} distinct scores (ties hurt AP)")
    size_mib = __import__("os").path.getsize(path) / 2**20
    if size_mib > 12:
        problems.append(f"file is {size_mib:.1f} MiB; limit 12 MiB")
    if problems:
        print("FAILED:"); [print(" -", p) for p in problems]; return False
    print(f"OK: {len(df)} rows, {size_mib:.2f} MiB, scores in [{s.min():.4f}, {s.max():.4f}]")
    return True

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "outputs/predictions.csv"
    rows = int(sys.argv[2]) if len(sys.argv) > 2 else 40101
    sys.exit(0 if check(path, rows) else 1)
