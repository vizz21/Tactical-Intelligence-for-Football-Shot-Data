"""The fixed Phase 5 eval set must never change (it is the only data used for the final comparison)."""
import hashlib
from pathlib import Path

D = Path(__file__).resolve().parents[1] / "data" / "processed"


def test_eval_inputs_sha_unchanged():
    expected = (D / "gen_inputs_eval.sha256").read_text().split()[0]
    assert hashlib.sha256((D / "gen_inputs_eval.jsonl").read_bytes()).hexdigest() == expected
