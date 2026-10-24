from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"

TRAIN_DIR = RAW / "train"          # transactions/transfers/blocks .parquet, market_rates.csv, labels.csv
VALID_DIR = RAW / "validation"     # unlabelled practice split
TEST_DIR = RAW / "test-data"       # separate download: the 40,101 transactions that are scored

ID_COL = "TRANSACTION_HASH"
LABEL_COL = "is_suspicious"
META_COLS = ["family", "difficulty", "scenario_id", "cohort_id"]

XRP_SENTINEL = "0x1111222211112222111122221111222211112222"
SEED = 42
N_FOLDS = 5

FEATURE_MODULES = ["features_core", "features_behaviour", "features_graph", "features_chain"]

LGB_PARAMS = dict(
    objective="binary",
    n_estimators=600,
    learning_rate=0.025,
    num_leaves=15,
    min_child_samples=100,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.7,
    reg_lambda=5.0,
    random_state=SEED,
    n_jobs=-1,
    verbose=-1,
)
