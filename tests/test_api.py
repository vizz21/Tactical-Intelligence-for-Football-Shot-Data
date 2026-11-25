def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    b = r.json()
    assert b["status"] == "ok" and b["n_documents"] == b["n_shots"] == 10233 and 0 < b["low_confidence_threshold"] < 1


def test_search_contract(client):
    r = client.post("/search", json={"query": "headers from the left that were saved", "k": 4})
    assert r.status_code == 200
    b = r.json()
    assert b["k"] == 4 and len(b["evidence"]) == 4
    e = b["evidence"][0]
    for key in ("rank", "score", "shot_id", "match_id", "team", "player", "minute", "period", "document", "doc_attrs",
                "split", "in_sample", "xg", "goal", "source"):
        assert key in e
    assert e["rank"] == 1 and set(e["source"]) == {"file", "event_id"}
    for key in ("n_results", "mean_xg", "n_goals", "n_above_threshold", "mean_xg_above_threshold", "low_confidence", "messages"):
        assert key in b["summary"]


def test_search_validation(client):
    assert client.post("/search", json={"query": "", "k": 5}).status_code == 422
    assert client.post("/search", json={"query": "shots", "k": 0}).status_code == 422
    assert client.post("/search", json={"k": 5}).status_code == 422


def test_shot_detail_and_404(client):
    sid = client.post("/search", json={"query": "goals from inside the box", "k": 1}).json()["evidence"][0]["shot_id"]
    r = client.get(f"/shot/{sid}")
    assert r.status_code == 200
    b = r.json()
    assert b["shot_id"] == sid and "features" in b and 0 <= b["xg"] <= 1
    assert b["source_event"] is None or b["source_event"]["id"] == sid
    assert client.get("/shot/not-a-real-id").status_code == 404


def test_static_page(client):
    assert client.get("/").status_code == 200


def test_accented_names_are_not_mojibake(client):
    r = client.post("/search", json={"query": "left-footed goals from outside the box", "k": 50})
    assert r.status_code == 200
    names = " ".join(e["player"] or "" for e in r.json()["evidence"])
    assert "Ã" not in names and "é" in names   # e.g. "Lionel Andrés Messi Cuccittini"


def test_search_test_only_parameter(client):
    r = client.post("/search", json={"query": "shots by Barcelona", "k": 15, "test_only": True})
    assert r.status_code == 200
    b = r.json()
    assert len(b["evidence"]) == 15 and {e["split"] for e in b["evidence"]} == {"test"} and b["summary"]["test_only"] is True
    assert client.post("/search", json={"query": "shots", "k": 5}).json()["summary"]["test_only"] is False   # default: all splits


def test_summary_understood_fields_and_short_query(client):
    b = client.post("/search", json={"query": "headers", "k": 3}).json()
    assert len(b["evidence"]) == 3 and b["summary"]["low_confidence"] is False
    assert b["summary"]["understood"] == {"body": "headers"} and "team" in b["summary"]["not_specified"]


def test_evaluation_endpoint(client):
    b = client.get("/evaluation").json()
    ids = {s["id"] for s in b["sections"]}
    assert {"xg", "retr", "gen", "faith"} <= ids and b["cautions"]
    gen = next(s for s in b["sections"] if s["id"] == "gen")
    by_label = {i["label"][0]: i for i in gen["items"]}
    assert by_label["C"]["detail"] == "128/143" and by_label["A"]["value"] == 0.0 and by_label["D"]["value"] == 1.0
    for s in b["sections"]:
        if s["kind"] == "bars":
            assert all(0 <= i["lo"] <= i["value"] <= i["hi"] <= 1 for i in s["items"])
        else:
            assert s["rows"] and all(len(r) == len(s["columns"]) for r in s["rows"])
    assert client.get("/").text.count("tab-e") >= 1


def test_cnn_comparison_and_spatial_tensor(client):
    d = client.get("/evaluation/cnn").json()
    cal = d["calibration"]
    assert set(cal) == {"logistic", "ANN", "CNN"} and all(len(v) == 6 and sum(b["n"] for b in v) == 1480 for v in cal.values())
    rows = {r[0]: r for r in d["summary"]["rows"]}
    assert rows["logistic"][1] == "0.2799"                              # the logistic refit reproduces the Phase 2 number
    assert rows["ANN"][1].startswith("0.2802") and rows["CNN"][1].startswith("0.2855")   # mean over seeds, as reported in the Phase 2 notes
    assert len(d["cnn_search"]["rows"]) == 7 and len(d["channels"]) == 6
    t = client.get(f"/spatial/{d['sample_shots'][0]['shot_id']}").json()
    assert t["shape"] == [6, 40, 60] and len(t["tensor"]) == 6 and set(t["probabilities"]) == {"logistic", "ANN", "CNN"}
    assert all(0 < p < 1 for p in t["probabilities"].values())
    assert client.get("/spatial/not-a-shot").status_code == 404
