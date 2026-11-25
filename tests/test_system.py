"""Phase 6 system checks on the 50 independent questions (data/phase6/queries_50.csv): test-only retrieval, evidence
traceability (shot ids resolve to source events, xG equals the saved ANN ensemble, split labels) and unsupported warnings."""
import csv
from pathlib import Path

import pandas as pd
import pytest

from football_rag import config as C

Q = Path(__file__).resolve().parents[1] / "data" / "phase6" / "queries_50.csv"
ROWS = list(csv.DictReader(open(Q, encoding="utf-8")))


@pytest.fixture(scope="module")
def xg_saved():
    df = pd.read_csv(C.XG_SCORES)
    return dict(zip(df["shot_id"], df["xg"]))


@pytest.fixture(scope="module")
def results(pipeline):
    return {r["id"]: pipeline.search(r["question"], k=10, test_only=True) for r in ROWS}


def test_all_evidence_is_test_split_with_matching_xg(results, xg_saved):
    n = 0
    for res in results.values():
        for e in res["evidence"]:
            n += 1
            assert e["split"] == "test" and e["in_sample"] is False
            assert abs(e["xg"] - xg_saved[e["shot_id"]]) < 1e-6
    assert n > 300


def test_shot_ids_resolve_to_source_events(pipeline, results):
    seen = 0
    for res in list(results.values())[:12]:
        for e in res["evidence"][:3]:
            shot = pipeline.get_shot(e["shot_id"])
            assert shot is not None and shot["shot_id"] == e["shot_id"]
            assert shot["source_event"] is None or shot["source_event"]["id"] == e["shot_id"]
            assert abs(shot["xg"] - e["xg"]) < 1e-9
            seen += 1
    assert seen > 20


def test_unsupported_questions_get_a_warning(results):
    unsupported = [r for r in ROWS if r["category"] == "unsupported" and "minute" in r["question"].lower() or r["question"] in
                   ("Shots by Harry Kane", "What did Lewandowski do from inside the box?", "Messi's left-footed goals")]
    assert len(unsupported) >= 3
    for r in unsupported:
        assert any(m.startswith("Not supported") for m in results[r["id"]]["summary"]["messages"]), r["question"]
