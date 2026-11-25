import pytest

from football_rag.gen_format import fold_text
from football_rag.grounding import check_answer, citation_groups, split_sentences


def _ev(team, minute, player, body, side, area, yards, outcome, xg, goal=0, other="Jeremie Frimpong"):
    doc = (f"{team}, minute {minute}. Build-up: {player} passes to {other}. {other} shoots with the {body} from {yards} yards "
           f"({area}, {side}). Outcome: {outcome}.")
    cat = {"goal": "ended in a goal", "saved": "was saved", "blocked": "was blocked"}.get(outcome, "missed the target")
    return {"document_model": doc, "team_model": team, "player_model": player, "minute": minute, "xg": xg, "outcome": outcome, "goal": goal,
            "doc_attrs": {"team": team, "body": body, "side": side, "area": area, "outcome": cat, "yards": yards}}


EVIDENCE = [_ev("Bayer Leverkusen", 44, "Jonas Hofmann", "right foot", "central", "inside the box", 20, "blocked", 0.05),
            _ev("Bayer Leverkusen", 34, "Jonas Hofmann", "right foot", "central", "inside the box", 12, "goal", 0.27, goal=1),
            _ev("Bayer Leverkusen", 72, "Adam Hlozek", "head", "left", "outside the box", 22, "saved", 0.04, other="Adam Hlozek")]


def example(mode="answer", evidence=None, query="Find a shot by Bayer Leverkusen", notes=()):
    ev = EVIDENCE if evidence is None else evidence
    mean = sum(e["xg"] for e in ev) / len(ev) if ev else None
    return {"mode": mode, "query": query, "evidence": ev, "summary": {"mean_xg": mean, "n_goals": sum(e["goal"] for e in ev)},
            "retrieval": {"unsupported": list(notes)}}


GOOD = ("Shot [1] was a blocked right-foot effort by Jonas Hofmann in minute 44 from 20 yards, with an xG of 0.05. "
        "Shot [2] ended as a goal (xG 0.27) from 12 yards inside the box. The three shots average an xG of 0.12.")


def test_good_answer_passes():
    r = check_answer(example(), GOOD)
    assert r["passed"], r["reasons"]
    assert r["xg_ok"] and r["outcome_ok"] and r["unsupported"] == []


def test_split_and_citation_groups():
    assert len(split_sentences(GOOD)) == 3
    assert citation_groups("Shots [1] and [2], and [3] were saved")[0]["items"] == [1, 2, 3]
    assert fold_text("Rodríguez Ødegaard ć") == "Rodriguez Odegaard c"


@pytest.mark.parametrize("answer,what", [
    ("Shot [1] was taken by Harry Kane in minute 44 and has an xG of 0.05. Shot [2] was a goal with an xG of 0.27.", "name"),
    ("Shot [1] was blocked and has an xG of 0.50. Shot [2] was a goal with an xG of 0.27.", "xg"),
    ("Shot [1] was blocked in minute 89 with an xG of 0.05. Shot [2] was a goal with an xG of 0.27.", "minute"),
    ("Shot [1] ended in a goal with an xG of 0.05. Shot [2] was a goal with an xG of 0.27.", "outcome"),
    ("Shot [1] was blocked with an xG of 0.05 for Bayern Munich. Shot [2] was a goal with an xG of 0.27.", "team"),
    ("Shot [1] was blocked from 30 yards with an xG of 0.05. Shot [2] was a goal with an xG of 0.27.", "distance"),
    ("Shot [1] was a blocked header with an xG of 0.05. Shot [2] was a goal with an xG of 0.27.", "body part"),
    ("Shot [1] was blocked with an xG of 0.05. Shot [2] was a goal from the left side with an xG of 0.27.", "side"),
    ("Shot [1] was blocked from outside the box with an xG of 0.05. Shot [2] was a goal with an xG of 0.27.", "area"),
    ("Shots [1] and [2] were both blocked, with xG of 0.05 and 0.27.", "outcome"),
    ("Shot [9] was blocked with an xG of 0.05. Shot [2] was a goal with an xG of 0.27.", "citation"),
    ("Shot [1] was blocked with a goal probability of 5%. Shot [2] was a goal with an xG of 0.27.", "xg"),
])
def test_failing_answers_are_caught(answer, what):
    r = check_answer(example(), answer)
    assert not r["passed"]
    if what == "xg":
        assert not r["xg_ok"]
    elif what == "outcome":
        assert not r["outcome_ok"]
    elif what == "citation":
        assert not r["citations_ok"]
    else:
        assert any(u["type"] == what for u in r["unsupported"]), r


def test_negation_and_goal_phrases_are_not_false_alarms():
    ev = [e for e in EVIDENCE if e["goal"] == 0]
    ans = "None of these shots was a goal: [1] was blocked (xG 0.05) and [2] was saved (xG 0.04). The expected goals total is small."
    assert check_answer(example(evidence=ev), ans)["outcome_ok"]


def test_refusal_checks():
    ex = example(mode="refuse", evidence=[], query="what is the weather in Paris tomorrow")
    good = ("I can only answer questions about shots, so I cannot help with the weather. "
            "You can ask about the team, body part, side, area or outcome of a shot.")
    assert check_answer(ex, good)["passed"]
    assert not check_answer(ex, "I cannot help with the weather. Ask about something else.")["passed"]       # no attribute list
    assert not check_answer(ex, good + " See shot [1].")["passed"]                                         # citation without evidence


def test_partial_answer_must_mention_ignored_part():
    ex = example(mode="partial", query="Messi left foot goals", notes=["player names are not supported"])
    ok = ("I cannot filter by player name, so I searched only left-footed goals. Shot [2] was a goal with an xG of 0.27, "
          "and shot [1] was blocked with an xG of 0.05.")
    assert check_answer(ex, ok)["passed"], check_answer(ex, ok)["reasons"]
    bad = "Shot [2] was a goal with an xG of 0.27. Shot [1] was blocked with an xG of 0.05."
    assert not check_answer(ex, bad)["passed"]


def test_sentence_count_is_enforced():
    one = "Shot [1] was blocked with an xG of 0.05."
    assert not check_answer(example(), one)["format_ok"]
