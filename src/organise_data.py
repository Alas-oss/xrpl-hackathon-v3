"""Sort the downloaded files into data/raw/train and data/raw/validation by SHA-256.

The two packages share file names, so a browser may have renamed some to
transactions_1.parquet etc. and mixed them up. This script identifies each file by its
hash (from checksums.json) and copies it to the right place under its right name.

    python src/organise_data.py <folder_with_downloaded_files> [checksums.json]
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path
from config import RAW, ROOT

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    src = Path(sys.argv[1])
    sums = Path(sys.argv[2]) if len(sys.argv) > 2 else src / "checksums.json"
    if not sums.exists():
        sums = ROOT / "checksums.json"
    by_hash = {v: k for k, v in json.loads(sums.read_text()).items()
               if k.startswith(("train/", "validation/"))}
    found = set()
    for p in sorted(src.rglob("*")):
        if p.suffix not in {".parquet", ".csv"}:
            continue
        rel = by_hash.get(sha256(p))
        if rel is None:
            print(f"unmatched (test data or unrelated): {p.name}")
            continue
        dest = RAW / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
        found.add(rel)
        print(f"{p.name:28s} -> {dest.relative_to(ROOT)}")
    missing = sorted(set(by_hash.values()) - found)
    if missing:
        print("\nNOT FOUND:", *missing, sep="\n  ")

if __name__ == "__main__":
    main()
