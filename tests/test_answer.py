"""/answer endpoint: retrieval -> flan-t5 -> verify / fall back. Loads the fine-tuned model on CPU (a few seconds per call)."""
import json
from pathlib import Path

import pytest

from football_rag.generate import GEN, Generator, build_example
from football_rag.grounding import check_answer
from football_rag.verify import group_errors

D = Path(__file__).resolve().parents[1] / "data" / "processed"
pytestmark = pytest.mark.skipif(not Generator().available, reason="models/flan_t5_armC not present")


def test_input_matches_phase5_eval_inputs(pipeline):
    """The live input builder reproduces the stored eval inputs exactly (so the evaluated model sees the same text)."""
    rows = [json.loads(l) for l in (D / "gen_inputs_eval.jsonl").read_text(encoding="utf-8").splitlines()]
    sample = [r for r in rows if r["kind"] in ("template", "unsupported_minute", "off_topic")][::12]
    assert len(sample) >= 8
    for r in sample:
        ex = build_example(r["query"], pipeline.search(r["query"], 3, test_only=True))
        assert ex["input"] == r["input"] and ex["mode"] == r["mode"], r["id"]


def test_answer_contract_and_verification(client, pipeline):
    q = "Spain Women's left-footed shots from inside the box"
    r = client.post("/answer", json={"query": q})
    assert r.status_code == 200
    b = r.json()
    assert b["source"] in ("model", "template") and b["answer"] and len(b["evidence"]) == 3
    assert set(b["citations"]) == {"1", "2", "3"} and list(b["citations"].values()) == [e["shot_id"] for e in b["evidence"]]
    assert all(e["split"] == "test" for e in b["evidence"]) and "heuristic" in b["notice"]
    ex = build_example(q, pipeline.search(q, 3, test_only=True))
    assert check_answer(ex, b["answer"])["passed"] and group_errors(ex, b["answer"]) == []   # whatever is shown passes both checks
    if b["source"] == "model":
        assert b["fallback_reasons"] == [] and b["answer"] == b["raw_model_answer"]
    else:
        assert b["fallback_reasons"]


def test_answer_refuses_off_topic_without_evidence(client):
    b = client.post("/answer", json={"query": "what is the weather forecast for Berlin"}).json()
    assert b["evidence"] == [] and b["citations"] == {} and "[" not in b["answer"]
    assert any(w in b["answer"].lower() for w in ("can't", "cannot", "only answer", "only cover"))


def test_answer_validation_and_decoding_constants(client):
    assert client.post("/answer", json={"query": ""}).status_code == 422
    assert GEN == dict(num_beams=4, max_new_tokens=128, no_repeat_ngram_size=3, do_sample=False)
