# XRPL Decode & Detect - <TEAM NAME>

## Reproduce the predictions
1. Python 3.10+; `pip install -r requirements.txt`
2. Put the data in `data/raw/`:
   - Training package -> `data/raw/train/` and `data/raw/validation/`
     (if the files got mixed up/renamed, run `python src/organise_data.py <download_folder>`)
   - Test package (unzipped on its own) -> `data/raw/test-data/`
3. `python run_pipeline.py`  -> `outputs/predictions.csv`  (about 1-2 minutes on a laptop, CPU only)
4. `python src/make_zip.py` -> `outputs/code_submission.zip` (must be < 8 MiB)

## Approach
One row per transaction (sender, receiver, delivered value valued in reference USD, actual
first-leg cost, conversion legs). Features (src/features_*.py) are computed inside each split:
amounts/conversion spread, account behaviour and timing (relay/fan-in/fan-out/layering windows),
and graph/pair features. LightGBM, 3-seed average, rank-normalised scores. Validated with
GroupKFold by scenario (a random split overstates AP).

## Files
- src/organise_data.py : sorts downloaded files into data/raw by SHA-256
- src/base_table.py    : raw tables -> one row per transaction
- src/features_core.py / features_behaviour.py / features_graph.py : feature groups
- src/windows.py       : fast per-account time-window queries
- src/model.py         : grouped CV, LightGBM, rank scores
- src/check_submission.py : validates the CSV before upload
- DISCLOSURES.md       : libraries and AI tools used
