import pytest

from football_rag.retrieval import is_relevant


def test_template_queries_return_attribute_correct_results(pipeline, val_template_queries):
    """Top min(3, n_relevant) of every sampled val template query satisfy ALL asked attributes (Phase 3 relevance;
    like nP@k, a query with fewer than 3 relevant documents in the corpus is only judged on those slots)."""
    total = correct = 0
    for q, attrs in val_template_queries:
        n_rel = sum(is_relevant(attrs, d["doc_attrs"]) for d in pipeline.corpus)
        for e in pipeline.search(q, 3)["evidence"][:min(3, n_rel)]:
            total += 1
            correct += is_relevant(attrs, e["doc_attrs"])
    assert total >= 85 and correct / total >= 0.98


def test_evidence_fields_and_source_pointer(pipeline):
    r = pipeline.search("Find a shot that was saved from the right side", 5)
    assert len(r["evidence"]) == 5
    scores = [e["score"] for e in r["evidence"]]
    assert scores == sorted(scores, reverse=True)
    for e in r["evidence"]:
        assert e["source"]["file"] == f"open-data/data/events/{e['match_id']}.json"
        assert e["source"]["event_id"] == e["shot_id"]
        assert 0 <= e["xg"] <= 1 and e["goal"] in (0, 1)
        assert e["in_sample"] == (e["split"] == "train")
        ev = pipeline.load_source_event(e["match_id"], e["shot_id"])
        if ev is not None:                       # open-data/ may be absent on another machine
            assert ev["type"]["name"] == "Shot"


def test_summary_fields(pipeline):
    r = pipeline.search("goals from outside the box", 10)
    s = r["summary"]
    assert s["n_results"] == len(r["evidence"]) == 10
    assert s["n_goals"] == sum(e["goal"] for e in r["evidence"])
    assert s["mean_xg"] == pytest.approx(sum(e["xg"] for e in r["evidence"]) / 10)
    assert s["n_above_threshold"] >= 1 and s["mean_xg_above_threshold"] is not None
    assert s["low_confidence"] is False


def test_unsupported_and_low_confidence_are_explained(pipeline):
    r = pipeline.search("free kicks taken in minute 85", 5)           # minute cue, no recognised attribute -> withheld
    assert r["summary"]["unsupported"] and r["evidence"] == [] and any("minute" in m for m in r["summary"]["messages"])
    r = pipeline.search("all shots by Lionel Messi", 5)               # player cue, no attribute -> withheld
    assert any("player" in n for n in r["summary"]["unsupported"]) and r["evidence"] == []
    r = pipeline.search("Messi left foot goals", 5)                   # attributes present -> served, with a warning
    assert r["evidence"] and r["summary"]["unsupported"] and any("ignored" in m for m in r["summary"]["messages"])
    r = pipeline.search("what is the weather in Paris tomorrow", 5)
    assert r["summary"]["low_confidence"] and r["summary"]["messages"]


@pytest.mark.parametrize("query", ["shots", "headers", "shots by Barcelona", "goals", "efforts"])
def test_short_queries_are_in_scope_and_return_results(pipeline, query):
    r = pipeline.search(query, 5)
    assert r["summary"]["low_confidence"] is False and r["summary"]["unsupported"] == []
    assert len(r["evidence"]) == 5


def test_short_queries_are_understood_by_lexicon(pipeline):
    assert pipeline.search("shots by Barcelona", 3)["summary"]["understood"] == {"team": "barcelona"}
    s = pipeline.search("left-footed goals from outside the box", 3)["summary"]
    assert set(s["understood"]) == {"body", "area", "outcome"} and s["not_specified"] == ["team", "side"]
    assert pipeline.search("shots", 3)["summary"]["understood"] == {}


def test_team_short_query_returns_that_team(pipeline):
    for e in pipeline.search("shots by Barcelona", 5)["evidence"]:
        assert e["team"] == "Barcelona"


def test_test_only_filter_fills_topk_from_test_split(pipeline):
    for q in ("shots", "goals from outside the box"):
        r = pipeline.search(q, 20, test_only=True)
        assert len(r["evidence"]) == 20 and {e["split"] for e in r["evidence"]} == {"test"}
        assert not any(e["in_sample"] for e in r["evidence"]) and r["summary"]["test_only"] is True
    full = pipeline.search("goals from outside the box", 20)
    assert {e["split"] for e in full["evidence"]} > {"test"}               # unfiltered search mixes splits
    a = pipeline.search("goals from outside the box", 5, test_only=True)["summary"]
    assert a["n_above_threshold"] <= full["summary"]["n_above_threshold"]


def test_detector_has_no_false_positives_on_template_queries(pipeline, val_template_queries):
    assert [q for q, _ in val_template_queries if pipeline.detect_unsupported(q)] == []


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_internal_splits_filter(pipeline, split):
    r = pipeline.search("goals from outside the box", 5, splits={split})
    assert len(r["evidence"]) == 5 and {e["split"] for e in r["evidence"]} == {split}
    assert r["summary"]["searched_splits"] == [split]


def test_splits_filter_keeps_public_behaviour(pipeline):
    a = pipeline.search("goals from outside the box", 5, test_only=True)
    b = pipeline.search("goals from outside the box", 5, splits={"test"})
    assert [e["shot_id"] for e in a["evidence"]] == [e["shot_id"] for e in b["evidence"]]
    assert a["summary"]["test_only"] is True and b["summary"]["test_only"] is True
    assert pipeline.search("shots", 3)["summary"]["searched_splits"] == ["train", "val", "test"]
