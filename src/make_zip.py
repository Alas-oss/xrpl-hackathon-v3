"""Build the code ZIP for upload (limit 8 MiB). Excludes data, outputs, secrets.
python src/make_zip.py"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDE_DIRS = {"data", "outputs", ".git", "__pycache__", ".ipynb_checkpoints", ".venv"}
EXCLUDE_SUFFIX = {".pyc", ".zip", ".env"}

def main():
    out = ROOT / "outputs" / "code_submission.zip"
    out.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in ROOT.rglob("*"):
            rel = p.relative_to(ROOT)
            if p.is_dir() or set(rel.parts) & EXCLUDE_DIRS:
                continue
            if p.suffix in EXCLUDE_SUFFIX or "access" in p.name.lower():
                continue
            z.write(p, rel)
    mib = out.stat().st_size / 2**20
    print(f"{out}  {mib:.2f} MiB  ({'OK' if mib <= 8 else 'TOO BIG, limit 8 MiB'})")

if __name__ == "__main__":
    main()
