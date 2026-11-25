"""Unit tests for football_rag.verify: each rule must flag a failing example and accept a correct one."""
import json
from pathlib import Path

import pytest

from football_rag.grounding import check_answer
from football_rag.verify import choose_answer, group_errors, template_answer

D = Path(__file__).resolve().parents[1] / "data" / "processed"


def _ex(outcomes, xgs, bodies=None, areas=None, sides=None, names=None):
    ev = []
    for i in range(3):
        ev.append({"document_model": f"Spain, minute {i}.", "team_model": "Spain", "player_model": (names or ["Ana Lopez", "Eva Ruiz", "Mia Diaz"])[i],
                   "team": "Spain", "xg": xgs[i], "outcome": outcomes[i], "goal": int(outcomes[i] == "goal"), "minute": i, "split": "test",
                   "doc_attrs": {"team": "Spain", "body": (bodies or ["right foot"] * 3)[i], "area": (areas or ["inside the box"] * 3)[i],
                                 "side": (sides or ["central"] * 3)[i], "outcome": outcomes[i]}})
    return {"query": "Spain shots", "mode": "answer", "evidence": ev, "summary": {"mean_xg": sum(xgs) / 3, "n_results": 3},
            "retrieval": {"unsupported": [], "low_confidence": False}}


BASE = _ex(["saved", "goal", "blocked"], [0.10, 0.30, 0.05])


def test_ordering_and_highest():
    assert "ordering" in group_errors(BASE, "[2] has the highest xG at 0.30, followed by [1] at 0.05 and [3] at 0.10.")
    assert "highest_value" in group_errors(BASE, "[1] has the highest xG at 0.10, followed by [2] at 0.30.")
    assert group_errors(BASE, "[2] has the highest xG at 0.30, followed by [1] at 0.10 and [3] at 0.05.") == []


def test_all_claims():
    assert "all_outcome" in group_errors(BASE, "All three Spain shots were goals. [1] by Ana Lopez (xG 0.10).")
    assert "all_foot" in group_errors(_ex(["saved"] * 3, [0.1] * 3, bodies=["right foot", "left foot", "right foot"]), "All three shots were right-footed.")
    assert "all_area" in group_errors(_ex(["saved"] * 3, [0.1] * 3, areas=["inside the box", "outside the box", "inside the box"]), "All three came from inside the box.")
    assert "all_side" in group_errors(_ex(["saved"] * 3, [0.1] * 3, sides=["central", "left", "central"]), "All three shots were central.")
    assert group_errors(_ex(["saved"] * 3, [0.1] * 3), "All three shots were saved and came from inside the box.") == []


def test_goal_count():
    assert "goal_count" in group_errors(BASE, "Two of the three were goals: [2] and [3].")
    assert "goal_count" in group_errors(BASE, "None of the three was a goal.")
    assert group_errors(BASE, "One goal among these shots is [2] by Eva Ruiz (xG 0.30).") == []


def test_attribution():
    assert "attribution_player" in group_errors(BASE, "[2] by Ana Lopez has an xG of 0.30.")
    assert "attribution_player" in group_errors(BASE, "Ana Lopez scored [2] with an xG of 0.30.")
    assert "attribution_team" in group_errors(BASE, "[2] by Eva Ruiz of Brazil has an xG of 0.30.")
    assert group_errors(BASE, "[2] by Eva Ruiz of Spain has an xG of 0.30, and [1] by Ana Lopez has 0.10.") == []


def test_garbled():
    assert "garbled" in group_errors(BASE, "[1] was saved (xG 0.10). Result [2].")
    assert "garbled" in group_errors(BASE, "[1] was saved, [3.] was blocked.")


def test_choose_answer_falls_back_and_keeps():
    ex = {json.loads(l)["id"]: json.loads(l) for l in (D / "gen_inputs_eval.jsonl").read_text(encoding="utf-8").splitlines()}["test-0010"]
    good = ("Norway Women's three right-footed shots from inside the box all went off target. [1] by Amalie Vevle Eikeland has the highest xG "
            "at 0.15, followed by [3] by Maren Naevdal Mjelde at 0.11 and [2] by Vilde Boe Risa at 0.08.")
    text, src, why = choose_answer(ex, good)
    assert src == "model" and text == good and not why
    bad = good.replace("all went off target", "were all goals").replace("[2] by Vilde Boe Risa", "[2] by Amalie Vevle Eikeland")
    text, src, why = choose_answer(ex, bad)
    assert src == "template" and why and text == template_answer(ex)


def test_template_always_passes_both_checks():
    rows = [json.loads(l) for l in (D / "gen_inputs_val.jsonl").read_text(encoding="utf-8").splitlines()]
    for ex in rows:
        t = template_answer(ex)
        assert check_answer(ex, t)["passed"] and group_errors(ex, t) == [], ex["id"]
