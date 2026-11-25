"""Central paths and constants for the Phase 4 pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
SCRIPTS = ROOT / "scripts"
OPEN_DATA_EVENTS = ROOT / "open-data" / "data" / "events"

TABULAR = DATA / "tabular.csv"
CORPUS = DATA / "text_corpus.jsonl"
PAIRS = DATA / "text_pairs.jsonl"
XG_SCORES = DATA / "xg_scores.csv"

XG_MODEL_DIR = ROOT / "models" / "xg_ann"
ENCODER_DIR = ROOT / "models" / "minilm_aug"
INDEX_DIR = ROOT / "models" / "index"
RESULTS_P4 = ROOT / "results" / "phase4"

MAX_SEQ_LENGTH = 128
NON_FEATURES = ["row_idx", "shot_id", "match_id", "team", "player", "period", "minute", "split", "goal"]

# Source pointer for traceability: StatsBomb event file for a match, then the event with this id.
SOURCE_TEMPLATE = "open-data/data/events/{match_id}.json"
